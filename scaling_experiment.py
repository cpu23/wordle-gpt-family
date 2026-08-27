from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from benchmark_cv import _aggregate_comparison
from cross_validation import (
    aggregate_seed_summaries,
    combine_fold_predictions,
    compare_paired_predictions,
    load_mode,
)
from evaluate_v2 import evaluate_checkpoint
from experiments_replay import train_with_replay
from experiments_v2 import train_stage
from model import WordleGPT
from tokenizer_v2 import VOCABULARY_SIZE
from train_v2 import load_v2_split
from wordle import DEFAULT_WORDS, load_words

MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "3.2m": {"context_length": 96, "embedding_size": 256, "num_layers": 4, "num_heads": 8, "mlp_size": 1024},
    "7.2m": {"context_length": 96, "embedding_size": 384, "num_layers": 4, "num_heads": 12, "mlp_size": 1536},
    "12.7m": {"context_length": 96, "embedding_size": 512, "num_layers": 4, "num_heads": 16, "mlp_size": 2048},
}
RATIOS = {"expert": 0.95, "mechanics": 0.05, "consistency": 0.0}
DECODE_MODES = ("raw", "constrained")


def parameter_counts() -> dict[str, int]:
    return {
        name: sum(p.numel() for p in WordleGPT(vocab_size=VOCABULARY_SIZE, **config).parameters())
        for name, config in MODEL_CONFIGS.items()
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _source_manifest(data_root: Path, mode_path: Path) -> dict[str, object]:
    files = sorted(path for path in data_root.rglob("*") if path.is_file())
    return {
        "mode": str(mode_path),
        "mode_sha256": sha256(mode_path),
        "corpus_files": [{"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in files],
    }


def run_scaling_experiment(
    mode_path: str | Path,
    data_dir: str | Path,
    runs_dir: str | Path,
    words: Sequence[str],
    *,
    device: str,
    batch_size: int = 128,
    accumulation_steps: int = 1,
    eval_batch_size: int = 256,
    max_epochs: int = 100,
) -> dict[str, object]:
    if batch_size < 1 or accumulation_steps < 1 or batch_size * accumulation_steps != 128:
        raise ValueError("physical batch size times accumulation steps must equal 128")
    mode_path, data_root, output_root = Path(mode_path), Path(data_dir), Path(runs_dir)
    mode = load_mode(mode_path)
    counts = parameter_counts()
    manifest = {
        "experiment": "1m-corpus-model-scaling",
        "architectures": MODEL_CONFIGS,
        "parameter_counts": counts,
        "physical_batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "effective_batch_size": batch_size * accumulation_steps,
        "eval_batch_size": eval_batch_size,
        "ratios": RATIOS,
        "learning_rate": 3e-4,
        "patience": 4,
        "max_epochs": max_epochs,
        "model_seeds": list(mode.model_seeds),
        "decode_modes": list(DECODE_MODES),
        "sources": _source_manifest(data_root, mode_path),
    }
    write_json(output_root / "manifest.json", manifest)
    summaries = {name: {decode: [] for decode in DECODE_MODES} for name in MODEL_CONFIGS}
    comparisons: dict[str, list[dict[str, object]]] = {}
    pairs = (("3.2m", "7.2m"), ("7.2m", "12.7m"), ("3.2m", "12.7m"))
    for a, b in pairs:
        for decode in DECODE_MODES:
            comparisons[f"{a}-vs-{b}-{decode}"] = []

    for seed in mode.model_seeds:
        for fold in mode.runs:
            fold_data = data_root / f"fold-{fold.run}"
            for name, config in MODEL_CONFIGS.items():
                run_dir = output_root / f"seed-{seed}" / f"fold-{fold.run}" / name
                mechanics_dir = run_dir / "mechanics"
                mechanics_checkpoint = mechanics_dir / "checkpoints" / "best.pt"
                mechanics_complete = mechanics_dir / "training-complete.json"
                replay_started = run_dir / "run.json"
                if not mechanics_complete.exists() and not replay_started.exists():
                    mechanics_checkpoint, mechanics_records = train_stage(
                        load_v2_split(fold_data / "mechanics", "train", example_type="mechanics"),
                        load_v2_split(fold_data / "mechanics", "validation", example_type="mechanics"),
                        mechanics_dir,
                        objective="mechanics", batch_size=batch_size,
                        eval_batch_size=eval_batch_size, learning_rate=3e-4,
                        patience=4, max_epochs=max_epochs, device=device, seed=seed,
                        model_config=config,
                    )
                    write_json(mechanics_complete, {"epochs_evaluated": len(mechanics_records), "checkpoint": str(mechanics_checkpoint)})
                checkpoint = run_dir / "checkpoints" / "best.pt"
                training_complete = run_dir / "training-complete.json"
                best_metadata = run_dir / "best.json"
                held_out_complete = all((run_dir / f"held-out-{decode}.json").exists() for decode in DECODE_MODES)
                retrained = False
                if not training_complete.exists() and (not held_out_complete or not best_metadata.exists()):
                    checkpoint, replay_records = train_with_replay(
                        {"expert": load_v2_split(fold_data / "expert-1m", "train", example_type="expert"),
                         "mechanics": load_v2_split(fold_data / "mechanics", "train", example_type="mechanics")},
                        {"expert": load_v2_split(fold_data / "expert-1m", "validation", example_type="expert"),
                         "mechanics": load_v2_split(fold_data / "mechanics", "validation", example_type="mechanics")},
                        run_dir, initialization_checkpoint=mechanics_checkpoint,
                        ratios=RATIOS, secrets=fold.validation, allowed_words=words,
                        batch_size=batch_size, gradient_accumulation_steps=accumulation_steps,
                        eval_batch_size=eval_batch_size, learning_rate=3e-4,
                        patience=4, max_epochs=max_epochs, device=device, seed=seed,
                    )
                    retrained = True
                    write_json(training_complete, {"epochs_evaluated": len(replay_records), "checkpoint": str(checkpoint)})
                for decode in DECODE_MODES:
                    result_path = run_dir / f"held-out-{decode}.json"
                    if retrained or not result_path.exists():
                        result = evaluate_checkpoint(checkpoint, fold.test, words, device=device, decode=decode)
                        write_json(result_path, asdict(result))
                if device == "cuda":
                    torch.cuda.empty_cache()

        combined: dict[str, dict[str, dict[str, object]]] = {}
        for name in MODEL_CONFIGS:
            for decode in DECODE_MODES:
                paths = [output_root / f"seed-{seed}" / f"fold-{fold.run}" / name / f"held-out-{decode}.json" for fold in mode.runs]
                result = combine_fold_predictions(mode, paths)
                combined.setdefault(name, {})[decode] = result
                summaries[name][decode].append(result)
                write_json(output_root / f"seed-{seed}" / f"{name}-{decode}-combined.json", result)
        for a, b in pairs:
            for decode in DECODE_MODES:
                label = f"{a}-vs-{b}-{decode}"
                paired = compare_paired_predictions(combined[a][decode], combined[b][decode])
                comparisons[label].append(paired)
                write_json(output_root / f"seed-{seed}" / f"paired-{label}.json", paired)

    aggregate = {
        **manifest,
        "summaries": {name: {decode: aggregate_seed_summaries(values) for decode, values in modes.items()} for name, modes in summaries.items()},
        "paired_comparisons": {label: _aggregate_comparison(values) for label, values in comparisons.items()},
    }
    write_json(output_root / "aggregate.json", aggregate)
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the controlled 1M-corpus model-scaling experiment.")
    parser.add_argument("--mode", type=Path, default=Path("data/wordle-development.json"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/wordle-dev-1m"))
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/scaling-dev-1m"))
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--print-parameters", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.print_parameters:
        print(json.dumps(parameter_counts(), indent=2, sort_keys=True))
        return 0
    aggregate = run_scaling_experiment(
        args.mode, args.data_dir, args.runs_dir, load_words(args.words), device=args.device,
        batch_size=args.batch_size, accumulation_steps=args.gradient_accumulation_steps,
        eval_batch_size=args.eval_batch_size, max_epochs=args.max_epochs,
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
