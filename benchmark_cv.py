from __future__ import annotations

import argparse
import json
from dataclasses import asdict
import statistics
from pathlib import Path

import torch

from cross_validation import (
    aggregate_seed_summaries,
    combine_fold_predictions,
    compare_paired_predictions,
    load_mode,
    materialize_expert_fold,
)
from evaluate_v2 import evaluate_checkpoint
from experiments_replay import train_with_replay
from experiments_v2 import train_stage
from train_v2 import load_v2_split
from wordle import DEFAULT_WORDS, load_words

MODEL_CONFIG = {
    "context_length": 96,
    "embedding_size": 256,
    "num_layers": 4,
    "num_heads": 8,
    "mlp_size": 1024,
}
VARIANTS = {"expert-100k": 100_000, "expert-200k": 200_000}
RATIOS = {"expert": 0.95, "mechanics": 0.05, "consistency": 0.0}
DEFAULT_MODE = Path("data/wordle-cv5.json")
DEFAULT_EXPERT_POOL = Path("data/wordle-v2-diverse-cv")
DEFAULT_MECHANICS_POOL = Path("data/wordle-v2-mechanics-cv")
DEFAULT_DATA_DIR = Path("data/wordle-cv5")
DEFAULT_RUNS_DIR = Path("runs/cv5-diverse")


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def prepare_fold_datasets(
    mode_path: str | Path,
    expert_pool: str | Path,
    mechanics_pool: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Materialize train/validation data while omitting every held-out source."""
    mode = load_mode(mode_path)
    root = Path(output_dir)
    manifests: dict[str, object] = {}
    for run in mode.runs:
        fold_dir = root / f"fold-{run.run}"
        manifests[f"fold-{run.run}"] = {
            variant: materialize_expert_fold(
                expert_pool,
                fold_dir / variant,
                run,
                prefix=prefix,
            )
            for variant, prefix in VARIANTS.items()
        }
        manifests[f"fold-{run.run}"]["mechanics"] = materialize_expert_fold(
            mechanics_pool,
            fold_dir / "mechanics",
            run,
        )
    _write_json(root / "manifest.json", manifests)
    return manifests


def _evaluation_payload(summary) -> dict[str, object]:
    return asdict(summary)


def run_benchmark(
    mode_path: str | Path,
    data_dir: str | Path,
    runs_dir: str | Path,
    words: list[str],
    *,
    device: str,
    max_epochs: int,
) -> dict[str, object]:
    """Train nested variants for every fold/seed and combine held-out predictions."""
    mode = load_mode(mode_path)
    data_root = Path(data_dir)
    output_root = Path(runs_dir)
    seed_summaries: dict[str, list[dict[str, object]]] = {
        variant: [] for variant in VARIANTS
    }
    seed_comparisons: list[dict[str, object]] = []

    for seed in mode.model_seeds:
        evaluation_paths: dict[str, list[Path]] = {
            variant: [] for variant in VARIANTS
        }
        for run in mode.runs:
            fold_data = data_root / f"fold-{run.run}"
            fold_run = output_root / f"seed-{seed}" / f"fold-{run.run}"
            mechanics_run = fold_run / "mechanics"
            mechanics_checkpoint = mechanics_run / "checkpoints" / "best.pt"
            if not mechanics_checkpoint.exists():
                mechanics_checkpoint, _ = train_stage(
                    load_v2_split(fold_data / "mechanics", "train", example_type="mechanics"),
                    load_v2_split(fold_data / "mechanics", "validation", example_type="mechanics"),
                    mechanics_run,
                    objective="mechanics",
                    batch_size=32,
                    eval_batch_size=256,
                    learning_rate=3e-4,
                    patience=4,
                    max_epochs=max_epochs,
                    device=device,
                    seed=seed,
                    model_config=MODEL_CONFIG,
                )
            for variant in VARIANTS:
                variant_run = fold_run / variant
                evaluation_path = variant_run / "held-out.json"
                evaluation_paths[variant].append(evaluation_path)
                if evaluation_path.exists():
                    continue
                expert_data = fold_data / variant
                best_path, _ = train_with_replay(
                    {
                        "expert": load_v2_split(expert_data, "train", example_type="expert"),
                        "mechanics": load_v2_split(fold_data / "mechanics", "train", example_type="mechanics"),
                    },
                    {
                        "expert": load_v2_split(expert_data, "validation", example_type="expert"),
                        "mechanics": load_v2_split(fold_data / "mechanics", "validation", example_type="mechanics"),
                    },
                    variant_run,
                    initialization_checkpoint=mechanics_checkpoint,
                    ratios=RATIOS,
                    secrets=run.validation,
                    allowed_words=words,
                    batch_size=32,
                    eval_batch_size=256,
                    learning_rate=3e-4,
                    patience=4,
                    max_epochs=max_epochs,
                    device=device,
                    seed=seed,
                )
                held_out = evaluate_checkpoint(
                    best_path,
                    run.test,
                    words,
                    device=device,
                )
                _write_json(evaluation_path, _evaluation_payload(held_out))
                if device == "cuda":
                    torch.cuda.empty_cache()

        combined_by_variant: dict[str, dict[str, object]] = {}
        for variant, paths in evaluation_paths.items():
            combined = combine_fold_predictions(mode, paths)
            combined_by_variant[variant] = combined
            seed_summaries[variant].append(combined)
            _write_json(
                output_root / f"seed-{seed}" / f"{variant}-combined.json",
                combined,
            )
        comparison = compare_paired_predictions(
            combined_by_variant["expert-100k"],
            combined_by_variant["expert-200k"],
        )
        seed_comparisons.append(comparison)
        _write_json(
            output_root / f"seed-{seed}" / "paired-comparison.json",
            comparison,
        )

    paired_deltas = [
        comparison["mean_guess_delta_b_minus_a"]
        for comparison in seed_comparisons
    ]
    outcome_names = (
        "a_loses_b_wins",
        "a_wins_b_loses",
        "both_win",
        "both_lose",
    )
    paired_outcomes = {
        outcome: {
            "mean": statistics.fmean(
                comparison["outcomes"][outcome]
                for comparison in seed_comparisons
            ),
            "standard_deviation": statistics.stdev(
                comparison["outcomes"][outcome]
                for comparison in seed_comparisons
            ),
            "values": [
                comparison["outcomes"][outcome]
                for comparison in seed_comparisons
            ],
        }
        for outcome in outcome_names
    }
    aggregate = {
        "mode": mode.name,
        "model_seeds": list(mode.model_seeds),
        "architecture": MODEL_CONFIG,
        "ratios": RATIOS,
        "variants": {
            variant: aggregate_seed_summaries(summaries)
            for variant, summaries in seed_summaries.items()
        },
        "paired_guess_delta_b_minus_a": {
            "mean": statistics.fmean(paired_deltas),
            "standard_deviation": statistics.stdev(paired_deltas),
            "values": paired_deltas,
        },
        "paired_outcomes": paired_outcomes,
    }
    _write_json(output_root / "aggregate.json", aggregate)
    return aggregate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict five-fold Wordle CV.")
    parser.add_argument("--mode", type=Path, default=DEFAULT_MODE)
    parser.add_argument("--expert-pool", type=Path, default=DEFAULT_EXPERT_POOL)
    parser.add_argument("--mechanics-pool", type=Path, default=DEFAULT_MECHANICS_POOL)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.skip_prepare:
        prepare_fold_datasets(
            args.mode,
            args.expert_pool,
            args.mechanics_pool,
            args.data_dir,
        )
    if args.prepare_only:
        return 0
    aggregate = run_benchmark(
        args.mode,
        args.data_dir,
        args.runs_dir,
        load_words(args.words),
        device=args.device,
        max_epochs=args.max_epochs,
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
