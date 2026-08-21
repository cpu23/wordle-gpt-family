from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
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
from evaluate_v2 import DECODE_MODES, evaluate_checkpoint
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
# Nested expert pools: variant name = state-count prefix of one universal pool.
# The default reproduces the original 100K/200K benchmark bit-for-bit.
DEFAULT_VARIANTS = "expert-100k=100000,expert-200k=200000"
DEFAULT_DECODE_MODES = "raw"
RATIOS = {"expert": 0.95, "mechanics": 0.05, "consistency": 0.0}
DEFAULT_MODE = Path("data/wordle-cv5.json")
DEFAULT_EXPERT_POOL = Path("data/wordle-v2-diverse-cv")
DEFAULT_MECHANICS_POOL = Path("data/wordle-v2-mechanics-cv")
DEFAULT_DATA_DIR = Path("data/wordle-cv5")
DEFAULT_RUNS_DIR = Path("runs/cv5-diverse")
OUTCOME_NAMES = (
    "a_loses_b_wins",
    "a_wins_b_loses",
    "both_win",
    "both_lose",
)


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def parse_variants(spec: str) -> dict[str, int]:
    """Parse ``name=prefix`` pairs into an ordered variant table."""
    variants: dict[str, int] = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, prefix = part.partition("=")
        if not name or not prefix.isdigit():
            raise ValueError(f"invalid variant specification: {part!r}")
        prefix = int(prefix)
        if prefix < 1 or name in variants:
            raise ValueError(f"invalid variant specification: {part!r}")
        variants[name] = prefix
    if not variants:
        raise ValueError("at least one variant is required")
    return variants


def parse_decode_modes(spec: str) -> tuple[str, ...]:
    """Parse a comma-separated list of evaluation decode modes."""
    modes: list[str] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if part not in DECODE_MODES or part in modes:
            raise ValueError(f"invalid decode mode: {part!r}")
        modes.append(part)
    if not modes:
        raise ValueError("at least one decode mode is required")
    return tuple(modes)


def prepare_fold_datasets(
    mode_path: str | Path,
    expert_pool: str | Path,
    mechanics_pool: str | Path,
    output_dir: str | Path,
    variants: Mapping[str, int],
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
            for variant, prefix in variants.items()
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


def _aggregate_comparison(
    comparisons: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Report mean and sample standard deviation of paired outcomes across seeds."""
    if not comparisons:
        raise ValueError("at least one seed comparison is required")
    deltas = [float(c["mean_guess_delta_b_minus_a"]) for c in comparisons]
    return {
        "seeds": len(comparisons),
        "outcomes": {
            name: {
                "mean": statistics.fmean(
                    float(c["outcomes"].get(name, 0)) for c in comparisons
                ),
                "standard_deviation": (
                    statistics.stdev(
                        float(c["outcomes"].get(name, 0)) for c in comparisons
                    )
                    if len(comparisons) > 1
                    else 0.0
                ),
                "values": [
                    int(c["outcomes"].get(name, 0)) for c in comparisons
                ],
            }
            for name in OUTCOME_NAMES
        },
        "mean_guess_delta_b_minus_a": {
            "mean": statistics.fmean(deltas),
            "standard_deviation": (
                statistics.stdev(deltas) if len(deltas) > 1 else 0.0
            ),
            "values": deltas,
        },
    }


def run_benchmark(
    mode_path: str | Path,
    data_dir: str | Path,
    runs_dir: str | Path,
    words: list[str],
    *,
    device: str,
    max_epochs: int,
    variants: Mapping[str, int],
    decode_modes: Sequence[str],
) -> dict[str, object]:
    """Train every nested variant for every fold/seed and combine held-out predictions."""
    mode = load_mode(mode_path)
    data_root = Path(data_dir)
    output_root = Path(runs_dir)
    variant_names = tuple(variants)
    seed_summaries: dict[str, dict[str, list[dict[str, object]]]] = {
        variant: {decode: [] for decode in decode_modes} for variant in variant_names
    }
    seed_comparisons: dict[str, list[dict[str, object]]] = {}
    variant_pairs = [
        (first, second)
        for position, first in enumerate(variant_names)
        for second in variant_names[position + 1 :]
    ]
    for first, second in variant_pairs:
        for decode in decode_modes:
            seed_comparisons[f"{first}-vs-{second}-{decode}"] = []
    if "raw" in decode_modes and "constrained" in decode_modes:
        for variant in variant_names:
            seed_comparisons[f"{variant}-raw-vs-constrained"] = []

    for seed in mode.model_seeds:
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
            for variant in variant_names:
                variant_run = fold_run / variant
                best_path = variant_run / "checkpoints" / "best.pt"
                missing_decodes = [
                    decode
                    for decode in decode_modes
                    if not (variant_run / f"held-out-{decode}.json").exists()
                ]
                if not missing_decodes:
                    continue
                if not best_path.exists():
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
                    if device == "cuda":
                        torch.cuda.empty_cache()
                for decode in missing_decodes:
                    evaluation_path = variant_run / f"held-out-{decode}.json"
                    held_out = evaluate_checkpoint(
                        best_path,
                        run.test,
                        words,
                        device=device,
                        decode=decode,
                    )
                    _write_json(evaluation_path, _evaluation_payload(held_out))
                    if device == "cuda":
                        torch.cuda.empty_cache()

        combined: dict[str, dict[str, dict[str, object]]] = {}
        for variant in variant_names:
            for decode in decode_modes:
                paths = [
                    output_root
                    / f"seed-{seed}"
                    / f"fold-{run.run}"
                    / variant
                    / f"held-out-{decode}.json"
                    for run in mode.runs
                ]
                combined.setdefault(variant, {})[decode] = combine_fold_predictions(
                    mode, paths
                )
                seed_summaries[variant][decode].append(
                    combined[variant][decode]
                )
                _write_json(
                    output_root
                    / f"seed-{seed}"
                    / f"{variant}-{decode}-combined.json",
                    combined[variant][decode],
                )
        for first, second in variant_pairs:
            for decode in decode_modes:
                label = f"{first}-vs-{second}-{decode}"
                comparison = compare_paired_predictions(
                    combined[first][decode],
                    combined[second][decode],
                )
                seed_comparisons[label].append(comparison)
                _write_json(
                    output_root / f"seed-{seed}" / f"paired-{label}.json",
                    comparison,
                )
        for variant in variant_names:
            if "raw" in decode_modes and "constrained" in decode_modes:
                label = f"{variant}-raw-vs-constrained"
                comparison = compare_paired_predictions(
                    combined[variant]["raw"],
                    combined[variant]["constrained"],
                )
                seed_comparisons[label].append(comparison)
                _write_json(
                    output_root / f"seed-{seed}" / f"paired-{label}.json",
                    comparison,
                )

    aggregate: dict[str, object] = {
        "mode": mode.name,
        "model_seeds": list(mode.model_seeds),
        "architecture": MODEL_CONFIG,
        "ratios": RATIOS,
        "variants": dict(variants),
        "decode_modes": list(decode_modes),
        "variant_summaries": {
            variant: {
                decode: aggregate_seed_summaries(summaries)
                for decode, summaries in by_decode.items()
            }
            for variant, by_decode in seed_summaries.items()
        },
        "comparisons": {
            label: _aggregate_comparison(seeds)
            for label, seeds in seed_comparisons.items()
            if seeds
        },
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
    parser.add_argument(
        "--variants",
        default=DEFAULT_VARIANTS,
        help=(
            "Comma-separated name=state-prefix pairs, e.g. "
            "expert-200k=200000,expert-500k=500000"
        ),
    )
    parser.add_argument(
        "--decode",
        default=DEFAULT_DECODE_MODES,
        help="Comma-separated decode modes: raw and/or constrained.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-prepare", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--max-epochs", type=int, default=100)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args()
    variants = parse_variants(args.variants)
    decode_modes = parse_decode_modes(args.decode)
    if not args.skip_prepare:
        prepare_fold_datasets(
            args.mode,
            args.expert_pool,
            args.mechanics_pool,
            args.data_dir,
            variants,
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
        variants=variants,
        decode_modes=decode_modes,
    )
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
