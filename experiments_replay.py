from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from evaluate_v2 import evaluate_model
from experiments_v2 import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EVAL_BATCH_SIZE,
    DEFAULT_MAX_EPOCHS,
    DEFAULT_MIN_DELTA,
    DEFAULT_PATIENCE,
    evaluate_objective_loss,
    load_initialization_checkpoint,
)
from model import WordleGPT
from tokenizer_v2 import VOCABULARY_SIZE
from train import DEFAULT_LEARNING_RATE, IGNORE_INDEX
from train_v2 import V2SplitData, load_v2_split
from wordle import DEFAULT_WORDS, load_words

DEFAULT_EXPERT_DATA_DIR = Path("data/wordle-v2")
DEFAULT_CONSISTENCY_DATA_DIR = Path("data/wordle-v2-consistency")
DEFAULT_SPLITS_PATH = Path("data/wordle-100k/secret-splits.json")
OBJECTIVES = ("expert", "mechanics", "consistency")


@dataclass(frozen=True)
class ReplayRecord:
    epoch: int
    step: int
    learning_rate: float
    batch_counts: dict[str, int]
    train_losses: dict[str, float]
    expert_validation_loss: float
    mechanics_validation_loss: float
    consistency_validation_loss: float | None
    wins: int
    games: int
    win_rate: float
    average_guesses: float
    invalid_guesses: int
    improved: bool


def replay_batch_counts(
    expert_batches: int,
    ratios: Mapping[str, float],
) -> dict[str, int]:
    """Convert requested proportions into replay counts per full expert epoch."""
    if expert_batches < 1:
        raise ValueError("expert_batches must be positive")
    if set(ratios) != set(OBJECTIVES):
        raise ValueError("ratios must define expert, mechanics, and consistency")
    if any(ratio < 0 for ratio in ratios.values()):
        raise ValueError("replay ratios cannot be negative")
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=1e-9):
        raise ValueError("replay ratios must sum to one")
    expert_ratio = ratios["expert"]
    if expert_ratio <= 0:
        raise ValueError("expert ratio must be positive")
    counts = {"expert": expert_batches}
    for objective in OBJECTIVES[1:]:
        counts[objective] = round(
            expert_batches * ratios[objective] / expert_ratio
        )
    return counts


def even_replay_schedule(counts: Mapping[str, int]) -> tuple[str, ...]:
    """Spread exact objective batch counts evenly with deterministic tie-breaking."""
    if set(counts) != set(OBJECTIVES):
        raise ValueError("counts must define every objective")
    if counts["expert"] < 1 or any(count < 0 for count in counts.values()):
        raise ValueError("objective batch counts are invalid")
    total = sum(counts.values())
    emitted = {objective: 0 for objective in OBJECTIVES}
    schedule: list[str] = []
    for position in range(1, total + 1):
        available = [
            objective
            for objective in OBJECTIVES
            if emitted[objective] < counts[objective]
        ]
        objective = max(
            available,
            key=lambda name: (
                position * counts[name] / total - emitted[name],
                -OBJECTIVES.index(name),
            ),
        )
        schedule.append(objective)
        emitted[objective] += 1
    return tuple(schedule)


def _sample_batches(
    data: V2SplitData,
    count: int,
    batch_size: int,
    generator: torch.Generator,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(count):
        selected = torch.randint(
            len(data.inputs),
            (batch_size,),
            generator=generator,
        )
        batches.append((data.inputs[selected], data.targets[selected]))
    return batches


def _sample_expert_epoch(
    data: V2SplitData,
    *,
    target_tokens: int,
    tokens_seen: int,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[list[tuple[torch.Tensor, torch.Tensor]], int]:
    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    while tokens_seen < target_tokens:
        batch = _sample_batches(data, 1, batch_size, generator)[0]
        batches.append(batch)
        tokens_seen += int((batch[1] != IGNORE_INDEX).sum().item())
    return batches, tokens_seen


def _save_checkpoint(
    path: Path,
    model: WordleGPT,
    record: ReplayRecord,
    *,
    ratios: Mapping[str, float],
    seed: int,
    initialization: str,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "vocabulary_size": VOCABULARY_SIZE,
            "epoch": record.epoch,
            "step": record.step,
            "learning_rate": record.learning_rate,
            "expert_validation_loss": record.expert_validation_loss,
            "mechanics_validation_loss": record.mechanics_validation_loss,
            "consistency_validation_loss": record.consistency_validation_loss,
            "gameplay": {
                "wins": record.wins,
                "games": record.games,
                "win_rate": record.win_rate,
                "average_guesses": record.average_guesses,
                "invalid_guesses": record.invalid_guesses,
            },
            "ratios": dict(ratios),
            "seed": seed,
            "initialization": initialization,
        },
        temporary,
    )
    temporary.replace(path)


def train_with_replay(
    train_data: Mapping[str, V2SplitData],
    validation_data: Mapping[str, V2SplitData],
    output_dir: str | Path,
    *,
    initialization_checkpoint: str | Path,
    ratios: Mapping[str, float],
    secrets: Sequence[str],
    allowed_words: Sequence[str],
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> tuple[Path, list[ReplayRecord]]:
    """Fine-tune on full expert epochs with deterministic interleaved replay."""
    if patience < 1 or max_epochs < 1:
        raise ValueError("patience and max_epochs must be positive")
    if min_delta < 0 or learning_rate < 0:
        raise ValueError("min_delta and learning_rate cannot be negative")
    if batch_size < 1 or eval_batch_size < 1:
        raise ValueError("batch sizes must be positive")
    if not secrets:
        raise ValueError("gameplay evaluation requires at least one secret")
    if set(train_data) != {name for name in OBJECTIVES if ratios[name] > 0}:
        raise ValueError("training data must match objectives with nonzero ratios")
    if not {"expert", "mechanics"} <= set(validation_data):
        raise ValueError("expert and mechanics validation data are required")

    nominal_expert_batches = math.ceil(len(train_data["expert"].inputs) / batch_size)
    nominal_batch_counts = replay_batch_counts(nominal_expert_batches, ratios)
    run_dir = Path(output_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(seed)
    model = WordleGPT(vocab_size=VOCABULARY_SIZE).to(selected_device)
    load_initialization_checkpoint(
        model,
        initialization_checkpoint,
        device=selected_device,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generators = {
        objective: torch.Generator().manual_seed(
            seed + OBJECTIVES.index(objective) * 1_000_003
        )
        for objective in OBJECTIVES
    }
    best_path = checkpoints_dir / "best.pt"
    records: list[ReplayRecord] = []
    best_validation = float("inf")
    patience_reference = float("inf")
    checks_without_progress = 0
    step = 0
    expert_epoch_tokens = train_data["expert"].supervised_token_count
    expert_tokens_seen = 0

    realized_total = sum(nominal_batch_counts.values())
    run_config = {
        "initialization": str(initialization_checkpoint),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "vocabulary_size": VOCABULARY_SIZE,
        "requested_ratios": dict(ratios),
        "nominal_batches_per_expert_epoch": nominal_batch_counts,
        "nominal_realized_ratios": {
            objective: nominal_batch_counts[objective] / realized_total
            for objective in OBJECTIVES
        },
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "min_delta": min_delta,
        "max_epochs": max_epochs,
        "seed": seed,
        "gameplay_secrets": len(secrets),
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for epoch in range(max_epochs + 1):
        train_losses: dict[str, float] = {}
        if epoch:
            expert_sampled, expert_tokens_seen = _sample_expert_epoch(
                train_data["expert"],
                target_tokens=epoch * expert_epoch_tokens,
                tokens_seen=expert_tokens_seen,
                batch_size=batch_size,
                generator=generators["expert"],
            )
            batch_counts = replay_batch_counts(len(expert_sampled), ratios)
            sampled = {"expert": expert_sampled}
            for objective in OBJECTIVES[1:]:
                if batch_counts[objective]:
                    sampled[objective] = _sample_batches(
                        train_data[objective],
                        batch_counts[objective],
                        batch_size,
                        generators[objective],
                    )
            schedule = even_replay_schedule(batch_counts)
            cursors = {objective: 0 for objective in sampled}
            loss_sums = {
                objective: torch.zeros((), device=selected_device)
                for objective in sampled
            }
            token_counts = {
                objective: torch.zeros((), dtype=torch.long, device=selected_device)
                for objective in sampled
            }
            model.train()
            for objective in schedule:
                inputs, targets = sampled[objective][cursors[objective]]
                cursors[objective] += 1
                inputs = inputs.to(selected_device)
                targets = targets.to(selected_device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(inputs)
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=IGNORE_INDEX,
                )
                loss.backward()
                optimizer.step()
                supervised = (targets != IGNORE_INDEX).sum()
                loss_sums[objective] += loss.detach() * supervised
                token_counts[objective] += supervised
                step += 1
            train_losses = {
                objective: loss_sums[objective].item() / token_counts[objective].item()
                for objective in sampled
            }

        expert_validation = evaluate_objective_loss(
            model,
            validation_data["expert"],
            batch_size=eval_batch_size,
        )
        mechanics_validation = evaluate_objective_loss(
            model,
            validation_data["mechanics"],
            batch_size=eval_batch_size,
        )
        consistency_validation = (
            evaluate_objective_loss(
                model,
                validation_data["consistency"],
                batch_size=eval_batch_size,
            )
            if "consistency" in validation_data
            else None
        )
        gameplay = evaluate_model(model, secrets, allowed_words)
        improved = expert_validation < best_validation
        record = ReplayRecord(
            epoch=epoch,
            step=step,
            learning_rate=learning_rate,
            batch_counts=dict(batch_counts) if epoch else {name: 0 for name in OBJECTIVES},
            train_losses=train_losses,
            expert_validation_loss=expert_validation,
            mechanics_validation_loss=mechanics_validation,
            consistency_validation_loss=consistency_validation,
            wins=gameplay.wins,
            games=gameplay.games,
            win_rate=gameplay.win_rate,
            average_guesses=gameplay.average_guesses,
            invalid_guesses=gameplay.invalid_guesses,
            improved=improved,
        )
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as output:
            output.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        if improved:
            best_validation = expert_validation
            _save_checkpoint(
                best_path,
                model,
                record,
                ratios=ratios,
                seed=seed,
                initialization=str(initialization_checkpoint),
            )
        if expert_validation < patience_reference - min_delta:
            patience_reference = expert_validation
            checks_without_progress = 0
        elif epoch:
            checks_without_progress += 1
        print(
            f"epoch={epoch} step={step} expert={expert_validation:.4f} "
            f"mechanics={mechanics_validation:.4f} "
            f"consistency={consistency_validation if consistency_validation is not None else 'n/a'} "
            f"wins={gameplay.wins}/{gameplay.games} avg={gameplay.average_guesses:.3f} "
            f"invalid={gameplay.invalid_guesses} patience={checks_without_progress}/{patience}"
        )
        if checks_without_progress >= patience:
            break
    (run_dir / "best.json").write_text(
        json.dumps(asdict(min(records, key=lambda item: item.expert_validation_loss)), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return best_path, records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run expert SFT with deterministic mechanics/consistency replay."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--expert-data-dir", type=Path, default=DEFAULT_EXPERT_DATA_DIR)
    parser.add_argument(
        "--consistency-data-dir", type=Path, default=DEFAULT_CONSISTENCY_DATA_DIR
    )
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--expert-ratio", type=float, default=1.0)
    parser.add_argument("--mechanics-ratio", type=float, default=0.0)
    parser.add_argument("--consistency-ratio", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--eval-batch-size", type=int, default=DEFAULT_EVAL_BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--min-delta", type=float, default=DEFAULT_MIN_DELTA)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    ratios = {
        "expert": args.expert_ratio,
        "mechanics": args.mechanics_ratio,
        "consistency": args.consistency_ratio,
    }
    replay_batch_counts(1, ratios)
    train_data = {
        "expert": load_v2_split(args.expert_data_dir, "train", example_type="expert")
    }
    if ratios["mechanics"]:
        train_data["mechanics"] = load_v2_split(
            args.expert_data_dir,
            "train",
            example_type="mechanics",
        )
    if ratios["consistency"]:
        train_data["consistency"] = load_v2_split(
            args.consistency_data_dir,
            "train",
            example_type="consistency",
        )
    validation_data = {
        "expert": load_v2_split(
            args.expert_data_dir,
            "validation",
            example_type="expert",
        ),
        "mechanics": load_v2_split(
            args.expert_data_dir,
            "validation",
            example_type="mechanics",
        ),
    }
    if ratios["consistency"]:
        validation_data["consistency"] = load_v2_split(
            args.consistency_data_dir,
            "validation",
            example_type="consistency",
        )
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    best_path, records = train_with_replay(
        train_data,
        validation_data,
        args.output_dir,
        initialization_checkpoint=args.initial_checkpoint,
        ratios=ratios,
        secrets=splits["test"],
        allowed_words=load_words(args.words),
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        max_epochs=args.max_epochs,
        device=args.device,
        seed=args.seed,
    )
    best = min(records, key=lambda item: item.expert_validation_loss)
    print(
        f"best checkpoint: {best_path} epoch={best.epoch} "
        f"expert={best.expert_validation_loss:.6f} "
        f"mechanics={best.mechanics_validation_loss:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
