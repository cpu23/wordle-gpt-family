from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from model import WordleGPT
from tokenizer_v2 import VOCABULARY_SIZE
from train import DEFAULT_LEARNING_RATE, IGNORE_INDEX, calculate_loss
from train_v2 import V2SplitData, load_v2_split

DEFAULT_DATA_DIR = Path("data/wordle-v2")
DEFAULT_RUNS_DIR = Path("runs")
DEFAULT_BATCH_SIZE = 32
DEFAULT_EVAL_BATCH_SIZE = 256
DEFAULT_PATIENCE = 4
DEFAULT_MIN_DELTA = 1e-4
DEFAULT_MAX_EPOCHS = 100


@dataclass(frozen=True)
class StageRecord:
    epoch: int
    step: int
    tokens_seen: int
    token_epochs: float
    learning_rate: float
    train_loss: float
    validation_loss: float
    improved: bool


def evaluate_objective_loss(
    model: WordleGPT,
    data: V2SplitData,
    *,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
) -> float:
    """Calculate exact token-weighted loss over one masked v2 objective split."""
    device = next(model.parameters()).device
    total_loss = torch.zeros((), device=device)
    total_tokens = torch.zeros((), dtype=torch.long, device=device)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(data.inputs), batch_size):
                inputs = data.inputs[start : start + batch_size].to(device)
                targets = data.targets[start : start + batch_size].to(device)
                token_count = (targets != IGNORE_INDEX).sum()
                loss = calculate_loss(model(inputs), targets)
                total_loss += loss * token_count
                total_tokens += token_count
    finally:
        model.train(was_training)
    return total_loss.item() / total_tokens.item()


def _save_best_checkpoint(
    path: Path,
    *,
    model: WordleGPT,
    objective: str,
    initialization: str,
    record: StageRecord,
    seed: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "vocabulary_size": VOCABULARY_SIZE,
            "objective": objective,
            "initialization": initialization,
            "epoch": record.epoch,
            "step": record.step,
            "tokens_seen": record.tokens_seen,
            "token_epochs": record.token_epochs,
            "learning_rate": record.learning_rate,
            "train_loss": record.train_loss,
            "validation_loss": record.validation_loss,
            "seed": seed,
        },
        temporary,
    )
    temporary.replace(path)


def _record_payload(record: StageRecord) -> dict[str, object]:
    return {
        "epoch": record.epoch,
        "step": record.step,
        "tokens_seen": record.tokens_seen,
        "token_epochs": record.token_epochs,
        "learning_rate": record.learning_rate,
        "train_loss": record.train_loss,
        "validation_loss": record.validation_loss,
        "improved": record.improved,
    }


def train_stage(
    train_data: V2SplitData,
    validation_data: V2SplitData,
    output_dir: str | Path,
    *,
    objective: str,
    initialization_checkpoint: str | Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    patience: int = DEFAULT_PATIENCE,
    min_delta: float = DEFAULT_MIN_DELTA,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    device: str | torch.device | None = None,
    seed: int = 0,
) -> tuple[Path, list[StageRecord]]:
    """Train one v2 objective until validation fails to improve for patience checks."""
    if objective not in ("mechanics", "expert"):
        raise ValueError("objective must be mechanics or expert")
    if patience < 1:
        raise ValueError("patience must be positive")
    if min_delta < 0:
        raise ValueError("min_delta cannot be negative")
    if max_epochs < 1:
        raise ValueError("max_epochs must be positive")
    if learning_rate < 0:
        raise ValueError("learning_rate cannot be negative")

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
    initialization = "random"
    if initialization_checkpoint is not None:
        checkpoint = torch.load(
            initialization_checkpoint,
            map_location=selected_device,
            weights_only=True,
        )
        if checkpoint.get("vocabulary_size") != VOCABULARY_SIZE:
            raise ValueError("initialization checkpoint has an incompatible vocabulary")
        model.load_state_dict(checkpoint["model_state_dict"])
        initialization = str(initialization_checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    epoch_tokens = train_data.supervised_token_count
    tokens_seen = 0
    step = 0
    best_validation = float("inf")
    patience_reference = float("inf")
    checks_without_progress = 0
    best_path = checkpoints_dir / "best.pt"
    records: list[StageRecord] = []

    run_config = {
        "objective": objective,
        "initialization": initialization,
        "vocabulary_size": VOCABULARY_SIZE,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "learning_rate": learning_rate,
        "patience": patience,
        "min_delta": min_delta,
        "max_epochs": max_epochs,
        "seed": seed,
        "train_examples": len(train_data.inputs),
        "validation_examples": len(validation_data.inputs),
        "train_supervised_tokens": epoch_tokens,
    }
    (run_dir / "run.json").write_text(
        json.dumps(run_config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for epoch in range(max_epochs + 1):
        train_loss = evaluate_objective_loss(
            model,
            train_data,
            batch_size=eval_batch_size,
        )
        validation_loss = evaluate_objective_loss(
            model,
            validation_data,
            batch_size=eval_batch_size,
        )
        improved = validation_loss < best_validation
        record = StageRecord(
            epoch=epoch,
            step=step,
            tokens_seen=tokens_seen,
            token_epochs=tokens_seen / epoch_tokens,
            learning_rate=optimizer.param_groups[0]["lr"],
            train_loss=train_loss,
            validation_loss=validation_loss,
            improved=improved,
        )
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(_record_payload(record), sort_keys=True) + "\n")
        if improved:
            best_validation = validation_loss
            _save_best_checkpoint(
                best_path,
                model=model,
                objective=objective,
                initialization=initialization,
                record=record,
                seed=seed,
            )
            (run_dir / "best.json").write_text(
                json.dumps(_record_payload(record), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        if validation_loss < patience_reference - min_delta:
            patience_reference = validation_loss
            checks_without_progress = 0
        else:
            checks_without_progress += 1
        print(
            f"epoch={epoch} step={step} tokens={tokens_seen} "
            f"lr={record.learning_rate:.2e} train={train_loss:.4f} "
            f"validation={validation_loss:.4f} best={best_validation:.4f} "
            f"patience={checks_without_progress}/{patience}"
        )
        if epoch == max_epochs or checks_without_progress >= patience:
            break

        target_tokens = (epoch + 1) * epoch_tokens
        while tokens_seen < target_tokens:
            indices = torch.randint(
                len(train_data.inputs),
                (batch_size,),
                generator=generator,
            )
            inputs = train_data.inputs.index_select(0, indices).to(selected_device)
            targets = train_data.targets.index_select(0, indices).to(selected_device)
            batch_tokens = int((targets != IGNORE_INDEX).sum().item())
            model.train()
            optimizer.zero_grad(set_to_none=True)
            loss = calculate_loss(model(inputs), targets)
            loss.backward()
            optimizer.step()
            tokens_seen += batch_tokens
            step += 1

    return best_path, records


def _default_output(runs_dir: Path, experiment: str) -> Path:
    names = {
        "a": "v2-experiment-a-expert-only",
        "b-mechanics": "v2-experiment-b-mechanics",
        "b-expert": "v2-experiment-b-expert",
    }
    return runs_dir / names[experiment]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run early-stopped Wordle dataset v2 experiments A and B."
    )
    parser.add_argument(
        "--experiment",
        choices=("a", "b-mechanics", "b-expert"),
        required=True,
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--initial-checkpoint", type=Path)
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
    objective = "mechanics" if args.experiment == "b-mechanics" else "expert"
    if args.experiment == "b-expert" and args.initial_checkpoint is None:
        raise ValueError("b-expert requires the best mechanics checkpoint")
    if args.experiment != "b-expert" and args.initial_checkpoint is not None:
        raise ValueError("only b-expert accepts an initialization checkpoint")
    train_data = load_v2_split(args.data_dir, "train", example_type=objective)
    validation_data = load_v2_split(
        args.data_dir,
        "validation",
        example_type=objective,
    )
    output_dir = args.output_dir or _default_output(args.runs_dir, args.experiment)
    best_path, records = train_stage(
        train_data,
        validation_data,
        output_dir,
        objective=objective,
        initialization_checkpoint=args.initial_checkpoint,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        patience=args.patience,
        min_delta=args.min_delta,
        max_epochs=args.max_epochs,
        device=args.device,
        seed=args.seed,
    )
    best = min(records, key=lambda record: record.validation_loss)
    print(
        f"best checkpoint: {best_path} epoch={best.epoch} "
        f"validation={best.validation_loss:.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
