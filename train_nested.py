from __future__ import annotations

import argparse
import gzip
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from model import WordleGPT
from tokenizer import (
    END_TOKEN,
    FEEDBACK_SYMBOLS,
    FEEDBACK_TOKEN,
    GUESS_TOKEN,
    LETTERS,
    TOKEN_TO_ID,
    encode,
    serialize_trajectory,
)
from train import DEFAULT_LEARNING_RATE, IGNORE_INDEX, calculate_loss, create_shifted_pairs

DEFAULT_DATA_DIR = Path("data/wordle-100k")
DEFAULT_BATCH_SIZE = 32
DEFAULT_EVAL_BATCH_SIZE = 256
DEFAULT_STEPS = 1000
DEFAULT_CHECKPOINTS = (0, 100, 500, 1000)
STRATEGIES = ("clever", "simple", "random", "partly-random")
STRATEGY_TO_ID = {strategy: index for index, strategy in enumerate(STRATEGIES)}
GUESS_LETTER_TYPE = 0
FEEDBACK_TYPE = 1
STRUCTURE_TYPE = 2
END_TYPE = 3
TOKEN_TYPE_NAMES = {
    GUESS_LETTER_TYPE: "guess_letter",
    FEEDBACK_TYPE: "feedback",
    STRUCTURE_TYPE: "structure",
    END_TYPE: "end",
}


@dataclass(frozen=True)
class TokenizedState:
    tokens: list[int]
    strategy: str
    current_guess_target_start: int


@dataclass(frozen=True)
class SplitData:
    inputs: Tensor
    targets: Tensor
    token_types: Tensor
    strategy_ids: Tensor
    current_guess_mask: Tensor

    @property
    def target_token_count(self) -> int:
        return int((self.targets != IGNORE_INDEX).sum().item())


@dataclass(frozen=True)
class TrainingRecord:
    step: int
    tokens_seen: int
    epochs: float
    learning_rate: float
    train_loss: float
    validation_losses: dict[str, float]


def nested_sizes(data_dir: str | Path) -> tuple[int, ...]:
    """Read the ordered nested-prefix sizes declared by the dataset."""
    manifest_path = Path(data_dir) / "manifest.json"
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)
    return tuple(sorted(int(size) for size in manifest["nested_datasets"]))


def load_nested_examples(
    data_dir: str | Path,
    prefix_size: int,
) -> dict[str, list[TokenizedState]]:
    """Tokenize train and validation state trajectories in one nested prefix."""
    if prefix_size < 1:
        raise ValueError("prefix_size must be positive")

    examples: dict[str, list[TokenizedState]] = {"train": [], "validation": []}
    states_path = Path(data_dir) / "states.jsonl.gz"
    records_read = 0
    with gzip.open(states_path, "rt", encoding="utf-8") as source:
        for state_index, line in enumerate(source):
            if state_index >= prefix_size:
                break
            state = json.loads(line)
            records_read += 1
            split = state["split"]
            if split not in examples:
                continue
            history = state["history"]
            turns = [
                *history,
                {"guess": state["action"], "feedback": state["feedback"]},
            ]
            examples[split].append(
                TokenizedState(
                    tokens=encode(serialize_trajectory(turns)),
                    strategy=state["strategy"],
                    current_guess_target_start=len(history) * 12,
                )
            )

    if records_read != prefix_size:
        raise ValueError(
            f"requested prefix of {prefix_size} states, found {records_read}"
        )
    for split, split_examples in examples.items():
        if not split_examples:
            raise ValueError(f"nested prefix has no {split} examples")
    return examples


def create_split_data(examples: Sequence[TokenizedState]) -> SplitData:
    """Collate token sequences and aligned token-type/strategy metadata."""
    if not examples:
        raise ValueError("at least one tokenized state is required")
    invalid_strategies = {example.strategy for example in examples} - set(STRATEGIES)
    if invalid_strategies:
        raise ValueError(f"unknown strategies: {sorted(invalid_strategies)}")

    inputs, targets = create_shifted_pairs([example.tokens for example in examples])
    token_types = torch.full_like(targets, -1, dtype=torch.int8)
    token_types[(targets >= 0) & (targets < len(LETTERS))] = GUESS_LETTER_TYPE
    feedback_start = TOKEN_TO_ID[FEEDBACK_SYMBOLS[0]]
    feedback_end = feedback_start + len(FEEDBACK_SYMBOLS)
    token_types[(targets >= feedback_start) & (targets < feedback_end)] = FEEDBACK_TYPE
    token_types[
        (targets == TOKEN_TO_ID[GUESS_TOKEN])
        | (targets == TOKEN_TO_ID[FEEDBACK_TOKEN])
    ] = STRUCTURE_TYPE
    token_types[targets == TOKEN_TO_ID[END_TOKEN]] = END_TYPE

    current_guess_mask = torch.zeros_like(targets, dtype=torch.bool)
    for row, example in enumerate(examples):
        start = example.current_guess_target_start
        current_guess_mask[row, start : start + 5] = True
    strategy_ids = torch.tensor(
        [STRATEGY_TO_ID[example.strategy] for example in examples],
        dtype=torch.int8,
    )
    return SplitData(
        inputs=inputs,
        targets=targets,
        token_types=token_types,
        strategy_ids=strategy_ids,
        current_guess_mask=current_guess_mask,
    )


def evaluate_losses(
    model: WordleGPT,
    data: SplitData,
    *,
    batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    breakdowns: bool = True,
) -> dict[str, float]:
    """Calculate exact token-weighted overall and categorized losses."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    metric_names = ["overall"]
    if breakdowns:
        metric_names.extend(TOKEN_TYPE_NAMES.values())
        metric_names.extend(f"trajectory_{strategy}" for strategy in STRATEGIES)
        metric_names.extend(f"next_guess_{strategy}" for strategy in STRATEGIES)
    device = next(model.parameters()).device
    sums = {name: torch.zeros((), device=device) for name in metric_names}
    counts = {
        name: torch.zeros((), dtype=torch.long, device=device) for name in metric_names
    }
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(data.inputs), batch_size):
                stop = start + batch_size
                inputs = data.inputs[start:stop].to(device)
                targets = data.targets[start:stop].to(device)
                losses = F.cross_entropy(
                    model(inputs).transpose(1, 2),
                    targets,
                    reduction="none",
                    ignore_index=IGNORE_INDEX,
                )
                valid = targets != IGNORE_INDEX
                sums["overall"] += losses.masked_select(valid).sum()
                counts["overall"] += valid.sum()
                if not breakdowns:
                    continue

                token_types = data.token_types[start:stop].to(device)
                current_guess = data.current_guess_mask[start:stop].to(device)
                strategy_ids = data.strategy_ids[start:stop].to(device)
                for token_type, name in TOKEN_TYPE_NAMES.items():
                    mask = token_types == token_type
                    sums[name] += losses.masked_select(mask).sum()
                    counts[name] += mask.sum()
                for strategy, strategy_id in STRATEGY_TO_ID.items():
                    rows = (strategy_ids == strategy_id).unsqueeze(1)
                    trajectory_mask = rows & valid
                    next_guess_mask = rows & current_guess
                    trajectory_name = f"trajectory_{strategy}"
                    next_guess_name = f"next_guess_{strategy}"
                    sums[trajectory_name] += losses.masked_select(trajectory_mask).sum()
                    counts[trajectory_name] += trajectory_mask.sum()
                    sums[next_guess_name] += losses.masked_select(next_guess_mask).sum()
                    counts[next_guess_name] += next_guess_mask.sum()
    finally:
        model.train(was_training)

    metrics: dict[str, float] = {}
    for name in metric_names:
        count = counts[name].item()
        metrics[name] = sums[name].item() / count if count else float("nan")
    return metrics


def _print_record(record: TrainingRecord) -> None:
    validation = record.validation_losses
    print(
        f"{record.step:<8}{record.tokens_seen:>14}{record.epochs:>10.3f}"
        f"{record.learning_rate:>12.2e}{record.train_loss:>12.4f}"
        f"{validation['overall']:>12.4f}{validation['guess_letter']:>12.4f}"
        f"{validation['feedback']:>12.4f}{validation['structure']:>12.4f}"
        f"{validation['end']:>12.4f}"
    )
    print(
        "  validation trajectory: "
        + "  ".join(
            f"{strategy}={validation[f'trajectory_{strategy}']:.4f}"
            for strategy in STRATEGIES
        )
    )
    print(
        "  validation next guess: "
        + "  ".join(
            f"{strategy}={validation[f'next_guess_{strategy}']:.4f}"
            for strategy in STRATEGIES
        )
    )


def train_nested_prefix(
    train_data: SplitData,
    validation_data: SplitData,
    *,
    num_steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str | torch.device | None = None,
    seed: int = 0,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
) -> tuple[WordleGPT, dict[int, TrainingRecord]]:
    """Train one fresh model and record detailed train/validation metrics."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = WordleGPT().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    monitored = {step for step in checkpoints if 0 <= step <= num_steps}
    monitored.update((0, num_steps))
    records: dict[int, TrainingRecord] = {}
    tokens_seen = 0
    epoch_tokens = train_data.target_token_count

    print(
        f"{'step':<8}{'tokens seen':>14}{'epochs':>10}{'lr':>12}"
        f"{'train':>12}{'validation':>12}{'guess':>12}{'feedback':>12}"
        f"{'structure':>12}{'end':>12}"
    )
    for step in range(num_steps + 1):
        if step in monitored:
            train_loss = evaluate_losses(
                model,
                train_data,
                batch_size=eval_batch_size,
                breakdowns=False,
            )["overall"]
            validation_losses = evaluate_losses(
                model,
                validation_data,
                batch_size=eval_batch_size,
            )
            record = TrainingRecord(
                step=step,
                tokens_seen=tokens_seen,
                epochs=tokens_seen / epoch_tokens,
                learning_rate=optimizer.param_groups[0]["lr"],
                train_loss=train_loss,
                validation_losses=validation_losses,
            )
            records[step] = record
            _print_record(record)
        if step == num_steps:
            break

        indices = torch.randint(
            len(train_data.inputs),
            (batch_size,),
            generator=generator,
        )
        inputs = train_data.inputs.index_select(0, indices).to(selected_device)
        targets = train_data.targets.index_select(0, indices).to(selected_device)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = calculate_loss(model(inputs), targets)
        loss.backward()
        optimizer.step()
        tokens_seen += int((targets != IGNORE_INDEX).sum().item())

    return model, records


def train_all_nested(
    data_dir: str | Path,
    *,
    sizes: Sequence[int] | None = None,
    num_steps: int = DEFAULT_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str | torch.device | None = None,
    seed: int = 0,
    checkpoints: Sequence[int] = DEFAULT_CHECKPOINTS,
) -> dict[int, dict[int, TrainingRecord]]:
    """Train a fresh, comparable model on every requested nested prefix."""
    available_sizes = nested_sizes(data_dir)
    selected_sizes = tuple(sizes) if sizes is not None else available_sizes
    invalid_sizes = set(selected_sizes) - set(available_sizes)
    if invalid_sizes:
        raise ValueError(f"sizes are not declared nested prefixes: {sorted(invalid_sizes)}")

    all_records: dict[int, dict[int, TrainingRecord]] = {}
    for prefix_size in selected_sizes:
        examples = load_nested_examples(data_dir, prefix_size)
        train_data = create_split_data(examples["train"])
        validation_data = create_split_data(examples["validation"])
        print(
            f"\ndataset {prefix_size} states "
            f"({len(examples['train'])} train, "
            f"{len(examples['validation'])} validation, "
            f"{train_data.target_token_count} train tokens)"
        )
        _, records = train_nested_prefix(
            train_data,
            validation_data,
            num_steps=num_steps,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            learning_rate=learning_rate,
            device=device,
            seed=seed,
            checkpoints=checkpoints,
        )
        all_records[prefix_size] = records
    return all_records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train WordleGPT on nested state prefixes with detailed losses."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--sizes", type=int, nargs="+")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    train_all_nested(
        args.data_dir,
        sizes=args.sizes,
        num_steps=args.steps,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        checkpoints=args.checkpoints,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
