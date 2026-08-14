from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

import torch

from model import WordleGPT
from train import DEFAULT_LEARNING_RATE, IGNORE_INDEX, calculate_loss
from train_nested import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_DATA_DIR,
    DEFAULT_EVAL_BATCH_SIZE,
    STRATEGIES,
    SplitData,
    TrainingRecord,
    create_split_data,
    evaluate_losses,
    load_nested_examples,
)

DEFAULT_TARGET_EPOCHS = 20.0
DEFAULT_SAVE_EPOCHS = (0.0, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0)
DEFAULT_OUTPUT_DIR = Path("runs/v1-full-20epochs")


def _record_payload(record: TrainingRecord, checkpoint: Path) -> dict[str, object]:
    return {
        "step": record.step,
        "tokens_seen": record.tokens_seen,
        "epochs": record.epochs,
        "learning_rate": record.learning_rate,
        "train_loss": record.train_loss,
        "validation_losses": record.validation_losses,
        "checkpoint": str(checkpoint),
    }


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_checkpoint(
    path: Path,
    *,
    model: WordleGPT,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    step: int,
    tokens_seen: int,
    epoch_tokens: int,
    seed: int,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "batch_generator_state": generator.get_state(),
            "step": step,
            "tokens_seen": tokens_seen,
            "epoch_tokens": epoch_tokens,
            "epochs": tokens_seen / epoch_tokens,
            "seed": seed,
        },
        temporary,
    )
    temporary.replace(path)


def _evaluate_record(
    model: WordleGPT,
    optimizer: torch.optim.Optimizer,
    train_data: SplitData,
    validation_data: SplitData,
    *,
    step: int,
    tokens_seen: int,
    eval_batch_size: int,
) -> TrainingRecord:
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
    return TrainingRecord(
        step=step,
        tokens_seen=tokens_seen,
        epochs=tokens_seen / train_data.target_token_count,
        learning_rate=optimizer.param_groups[0]["lr"],
        train_loss=train_loss,
        validation_losses=validation_losses,
    )


def _print_record(record: TrainingRecord, checkpoint: Path) -> None:
    validation = record.validation_losses
    print(
        f"step={record.step} tokens={record.tokens_seen} epochs={record.epochs:.4f} "
        f"lr={record.learning_rate:.2e} train={record.train_loss:.4f} "
        f"validation={validation['overall']:.4f} checkpoint={checkpoint.name}"
    )
    print(
        "  token losses: "
        f"guess={validation['guess_letter']:.4f} "
        f"feedback={validation['feedback']:.4f} "
        f"structure={validation['structure']:.4f} end={validation['end']:.4f}"
    )
    print(
        "  trajectory losses: "
        + " ".join(
            f"{strategy}={validation[f'trajectory_{strategy}']:.4f}"
            for strategy in STRATEGIES
        )
    )
    print(
        "  next-guess losses: "
        + " ".join(
            f"{strategy}={validation[f'next_guess_{strategy}']:.4f}"
            for strategy in STRATEGIES
        )
    )


def train_to_epochs(
    train_data: SplitData,
    validation_data: SplitData,
    output_dir: str | Path,
    *,
    target_epochs: float = DEFAULT_TARGET_EPOCHS,
    save_epochs: Sequence[float] = DEFAULT_SAVE_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    eval_batch_size: int = DEFAULT_EVAL_BATCH_SIZE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str | torch.device | None = None,
    seed: int = 0,
    resume: str | Path | None = None,
) -> list[TrainingRecord]:
    """Train until a token-equivalent epoch target, saving resumable checkpoints."""
    if target_epochs <= 0:
        raise ValueError("target_epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if eval_batch_size < 1:
        raise ValueError("eval_batch_size must be positive")

    run_dir = Path(output_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.jsonl"
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    torch.manual_seed(seed)
    model = WordleGPT().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    step = 0
    tokens_seen = 0
    epoch_tokens = train_data.target_token_count

    if resume is not None:
        checkpoint = torch.load(resume, map_location=selected_device, weights_only=True)
        if checkpoint["epoch_tokens"] != epoch_tokens:
            raise ValueError("checkpoint training-token count does not match the dataset")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        generator.set_state(checkpoint["batch_generator_state"].cpu())
        step = checkpoint["step"]
        tokens_seen = checkpoint["tokens_seen"]
    else:
        metrics_path.write_text("", encoding="utf-8")

    current_epochs = tokens_seen / epoch_tokens
    thresholds = sorted(
        {
            float(epoch)
            for epoch in (*save_epochs, target_epochs)
            if current_epochs <= epoch <= target_epochs
        }
    )
    if resume is not None and thresholds and thresholds[0] == current_epochs:
        thresholds.pop(0)
    records: list[TrainingRecord] = []

    _write_json(
        run_dir / "run.json",
        {
            "batch_size": batch_size,
            "eval_batch_size": eval_batch_size,
            "learning_rate": learning_rate,
            "seed": seed,
            "target_epochs": target_epochs,
            "save_epochs": list(thresholds),
            "train_examples": len(train_data.inputs),
            "validation_examples": len(validation_data.inputs),
            "train_tokens": epoch_tokens,
        },
    )

    next_threshold = 0
    while tokens_seen / epoch_tokens < target_epochs:
        current_epochs = tokens_seen / epoch_tokens
        if next_threshold < len(thresholds) and current_epochs >= thresholds[next_threshold]:
            threshold = thresholds[next_threshold]
            record = _evaluate_record(
                model,
                optimizer,
                train_data,
                validation_data,
                step=step,
                tokens_seen=tokens_seen,
                eval_batch_size=eval_batch_size,
            )
            checkpoint_path = checkpoints_dir / (
                f"epoch-{threshold:05.1f}-step-{step:06d}.pt"
            )
            _save_checkpoint(
                checkpoint_path,
                model=model,
                optimizer=optimizer,
                generator=generator,
                step=step,
                tokens_seen=tokens_seen,
                epoch_tokens=epoch_tokens,
                seed=seed,
            )
            with metrics_path.open("a", encoding="utf-8") as metrics_file:
                metrics_file.write(
                    json.dumps(_record_payload(record, checkpoint_path), sort_keys=True)
                    + "\n"
                )
            _write_json(
                run_dir / "latest.json",
                _record_payload(record, checkpoint_path),
            )
            records.append(record)
            _print_record(record, checkpoint_path)
            next_threshold += 1
            continue

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
        step += 1
        tokens_seen += batch_tokens

    while next_threshold < len(thresholds):
        threshold = thresholds[next_threshold]
        record = _evaluate_record(
            model,
            optimizer,
            train_data,
            validation_data,
            step=step,
            tokens_seen=tokens_seen,
            eval_batch_size=eval_batch_size,
        )
        checkpoint_path = checkpoints_dir / f"epoch-{threshold:05.1f}-step-{step:06d}.pt"
        _save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            generator=generator,
            step=step,
            tokens_seen=tokens_seen,
            epoch_tokens=epoch_tokens,
            seed=seed,
        )
        with metrics_path.open("a", encoding="utf-8") as metrics_file:
            metrics_file.write(
                json.dumps(_record_payload(record, checkpoint_path), sort_keys=True) + "\n"
            )
        _write_json(run_dir / "latest.json", _record_payload(record, checkpoint_path))
        records.append(record)
        _print_record(record, checkpoint_path)
        next_threshold += 1

    return records


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the full v1 dataset to a token-equivalent epoch target."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target-epochs", type=float, default=DEFAULT_TARGET_EPOCHS)
    parser.add_argument(
        "--save-epochs",
        type=float,
        nargs="+",
        default=DEFAULT_SAVE_EPOCHS,
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=DEFAULT_EVAL_BATCH_SIZE,
    )
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--resume", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    examples = load_nested_examples(args.data_dir, 100_000)
    train_data = create_split_data(examples["train"])
    validation_data = create_split_data(examples["validation"])
    train_to_epochs(
        train_data,
        validation_data,
        args.output_dir,
        target_epochs=args.target_epochs,
        save_epochs=args.save_epochs,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
