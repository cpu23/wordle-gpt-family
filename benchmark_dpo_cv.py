from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from action_regret import analyze_action_regret
from benchmark_cv import _aggregate_comparison
from cross_validation import (
    aggregate_seed_summaries,
    combine_fold_predictions,
    compare_paired_predictions,
    load_mode,
)
from evaluate_v2 import evaluate_checkpoint
from train_dpo import PreferenceData, load_preferences, train_dpo
from train_v2 import load_v2_split
from wordle import DEFAULT_WORDS, load_words

BETA = 0.20
LAMBDA_SFT = 1.0
LEARNING_RATE = 1e-6
EVALUATION_PASSES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)
DECODE_MODES = ("raw", "constrained")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def aggregate_regret_summaries(summaries: Sequence[Mapping[str, object]]) -> dict[str, object]:
    metrics = (
        "mean_action_regret",
        "median_action_regret",
        "rank_1_fraction",
        "top_3_fraction",
        "top_8_fraction",
    )
    aggregate: dict[str, object] = {"seeds": len(summaries)}
    for metric in metrics:
        values = [float(summary[metric]) for summary in summaries]
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return aggregate


def run_dpo_cv_benchmark(
    mode_path: str | Path = "data/wordle-cv5.json",
    preferences_dir: str | Path = "data/wordle-dpo-dev",
    sft_runs_dir: str | Path = "runs/scaling-cv5-1m",
    mechanics_dir: str | Path = "data/wordle-cv5-1m",
    output_dir: str | Path = "runs/dpo-cv5-7.2m",
    words_path: str | Path = DEFAULT_WORDS,
    *,
    device: str = "cuda",
    beta: float = BETA,
    lambda_sft: float = LAMBDA_SFT,
    learning_rate: float = LEARNING_RATE,
    target_seeds: Sequence[int] | None = None,
    target_folds: Sequence[int] | None = None,
) -> dict[str, object]:
    mode_path = Path(mode_path)
    preferences_dir = Path(preferences_dir)
    sft_runs_dir = Path(sft_runs_dir)
    mechanics_dir = Path(mechanics_dir)
    output_dir = Path(output_dir)
    words = list(load_words(words_path))
    mode = load_mode(mode_path)

    print(f"Loading preference datasets from {preferences_dir}...")
    t0 = time.perf_counter()
    train_preferences = load_preferences(preferences_dir / "train.jsonl.gz")
    validation_preferences = load_preferences(preferences_dir / "validation.jsonl.gz")
    print(f"Loaded {len(train_preferences):,} train pairs and {len(validation_preferences):,} val pairs in {time.perf_counter() - t0:.2f}s")

    selected_seeds = list(mode.model_seeds) if target_seeds is None else list(target_seeds)
    selected_folds = [r.run for r in mode.runs] if target_folds is None else list(target_folds)

    manifest = {
        "benchmark": "dpo-cv5-7.2m",
        "base_model": "7.2m SFT",
        "beta": beta,
        "lambda_sft": lambda_sft,
        "learning_rate": learning_rate,
        "evaluation_passes": list(EVALUATION_PASSES),
        "physical_batch_size": 64,
        "gradient_accumulation_steps": 2,
        "effective_batch_size": 128,
        "eval_batch_size": 256,
        "mode": str(mode_path),
        "mode_sha256": sha256(mode_path),
        "preferences_dir": str(preferences_dir),
        "train_preferences_sha256": sha256(preferences_dir / "train.jsonl.gz"),
        "validation_preferences_sha256": sha256(preferences_dir / "validation.jsonl.gz"),
        "sft_runs_dir": str(sft_runs_dir),
        "model_seeds": selected_seeds,
        "folds": selected_folds,
    }
    write_json(output_dir / "manifest.json", manifest)

    total_runs = len(selected_seeds) * len(selected_folds)
    current_run = 0

    for seed in selected_seeds:
        for fold_run in selected_folds:
            current_run += 1
            fold = next(r for r in mode.runs if r.run == fold_run)
            run_dir = output_dir / f"seed-{seed}" / f"fold-{fold.run}"
            checkpoints_dir = run_dir / "checkpoints"
            best_checkpoint = checkpoints_dir / "best.pt"
            complete_file = run_dir / "training-complete.json"

            base_checkpoint = sft_runs_dir / f"seed-{seed}" / f"fold-{fold.run}" / "7.2m" / "checkpoints" / "best.pt"
            if not base_checkpoint.exists():
                raise FileNotFoundError(f"Base SFT checkpoint not found: {base_checkpoint}")

            sft_best_json = sft_runs_dir / f"seed-{seed}" / f"fold-{fold.run}" / "7.2m" / "best.json"
            if sft_best_json.exists():
                sft_best_data = json.loads(sft_best_json.read_text(encoding="utf-8"))
                sft_val_wins = sft_best_data.get("wins", 50)
            else:
                sft_val_wins = 50
            collapse_wins = max(0, sft_val_wins - 10)

            print(f"\n[{current_run}/{total_runs}] Starting Seed {seed}, Fold {fold.run} (base: {base_checkpoint}, sft_val_wins: {sft_val_wins})")

            if not complete_file.exists() or not best_checkpoint.exists():
                mechanics_data = load_v2_split(
                    mechanics_dir / f"fold-{fold.run}" / "mechanics",
                    "validation",
                    example_type="mechanics",
                )

                checkpoint_path, records = train_dpo(
                    train_preferences,
                    validation_preferences,
                    mechanics_data,
                    run_dir,
                    base_checkpoint=base_checkpoint,
                    validation_secrets=fold.validation,
                    allowed_words=words,
                    beta=beta,
                    lambda_sft=lambda_sft,
                    learning_rate=learning_rate,
                    physical_batch_size=64,
                    gradient_accumulation_steps=2,
                    eval_batch_size=256,
                    evaluation_passes=EVALUATION_PASSES,
                    collapse_wins=collapse_wins,
                    seed=seed,
                    device=device,
                )
                write_json(
                    complete_file,
                    {
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": sha256(checkpoint_path),
                        "evaluations": len(records),
                        "base_checkpoint": str(base_checkpoint),
                        "base_checkpoint_sha256": sha256(base_checkpoint),
                    },
                )
                if device == "cuda":
                    torch.cuda.empty_cache()
            else:
                print(f"Seed {seed}, Fold {fold.run} DPO training already complete. Using existing {best_checkpoint}")

            # Evaluate on held-out test secrets (held-out test set was invisible during training/selection)
            for decode in DECODE_MODES:
                held_out_path = run_dir / f"held-out-{decode}.json"
                if not held_out_path.exists():
                    print(f"Evaluating {decode} on held-out test secrets ({len(fold.test)} secrets)...")
                    eval_res = evaluate_checkpoint(best_checkpoint, fold.test, words, device=device, decode=decode)
                    write_json(held_out_path, asdict(eval_res))

            held_out_regret_path = run_dir / "held-out-action-regret.json"
            if not held_out_regret_path.exists():
                constrained_data = json.loads((run_dir / "held-out-constrained.json").read_text(encoding="utf-8"))
                regret_res = analyze_action_regret(constrained_data, words)
                write_json(held_out_regret_path, regret_res)

    # Combining and comparisons
    print("\n--- Aggregating Benchmark Results Across Folds & Seeds ---")
    summaries: dict[str, list[dict[str, object]]] = {decode: [] for decode in DECODE_MODES}
    regret_summaries: list[dict[str, object]] = []
    comparisons: dict[str, list[dict[str, object]]] = {decode: [] for decode in DECODE_MODES}
    sft_regret_summaries: list[dict[str, object]] = []

    for seed in selected_seeds:
        seed_dir = output_dir / f"seed-{seed}"
        # Check if all folds are present
        all_folds_present = all(
            (seed_dir / f"fold-{f.run}" / "held-out-constrained.json").exists() and
            (seed_dir / f"fold-{f.run}" / "held-out-raw.json").exists()
            for f in mode.runs
        )
        if not all_folds_present:
            print(f"Seed {seed} has incomplete folds, skipping seed-level aggregation.")
            continue

        for decode in DECODE_MODES:
            fold_paths = [seed_dir / f"fold-{f.run}" / f"held-out-{decode}.json" for f in mode.runs]
            combined = combine_fold_predictions(mode, fold_paths)
            write_json(seed_dir / f"dpo-{decode}-combined.json", combined)
            summaries[decode].append(combined)

            # Paired comparison with SFT baseline
            sft_combined_path = sft_runs_dir / f"seed-{seed}" / f"7.2m-{decode}-combined.json"
            if sft_combined_path.exists():
                sft_combined = json.loads(sft_combined_path.read_text(encoding="utf-8"))
                paired = compare_paired_predictions(sft_combined, combined)
                write_json(seed_dir / f"paired-sft-vs-dpo-{decode}.json", paired)
                comparisons[decode].append(paired)

        # Combined action regret for DPO
        dpo_constrained_combined = json.loads((seed_dir / "dpo-constrained-combined.json").read_text(encoding="utf-8"))
        dpo_regret = analyze_action_regret(dpo_constrained_combined, words)
        write_json(seed_dir / "dpo-constrained-action-regret.json", dpo_regret)
        regret_summaries.append(dpo_regret["summary"])

        # SFT action regret (compute on SFT combined if not already done)
        sft_regret_path = seed_dir / "sft-constrained-action-regret.json"
        sft_combined_path = sft_runs_dir / f"seed-{seed}" / "7.2m-constrained-combined.json"
        if sft_combined_path.exists():
            sft_combined = json.loads(sft_combined_path.read_text(encoding="utf-8"))
            sft_regret = analyze_action_regret(sft_combined, words)
            write_json(sft_regret_path, sft_regret)
            sft_regret_summaries.append(sft_regret["summary"])

    if summaries["constrained"]:
        aggregate = {
            **manifest,
            "dpo_summaries": {decode: aggregate_seed_summaries(summaries[decode]) for decode in DECODE_MODES},
            "dpo_action_regret": aggregate_regret_summaries(regret_summaries),
            "paired_comparisons": {decode: _aggregate_comparison(comparisons[decode]) for decode in DECODE_MODES if comparisons[decode]},
        }
        if sft_regret_summaries:
            aggregate["sft_action_regret"] = aggregate_regret_summaries(sft_regret_summaries)

        write_json(output_dir / "aggregate.json", aggregate)
        print("Benchmark complete! Aggregate written to", output_dir / "aggregate.json")
        return aggregate
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run strict 5-fold x 3-seed DPO benchmark.")
    parser.add_argument("--mode", type=Path, default=Path("data/wordle-cv5.json"))
    parser.add_argument("--preferences", type=Path, default=Path("data/wordle-dpo-dev"))
    parser.add_argument("--sft-runs-dir", type=Path, default=Path("runs/scaling-cv5-1m"))
    parser.add_argument("--mechanics-dir", type=Path, default=Path("data/wordle-cv5-1m"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dpo-cv5-7.2m"))
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, nargs="*", default=None)
    parser.add_argument("--fold", type=int, nargs="*", default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    run_dpo_cv_benchmark(
        mode_path=args.mode,
        preferences_dir=args.preferences,
        sft_runs_dir=args.sft_runs_dir,
        mechanics_dir=args.mechanics_dir,
        output_dir=args.output_dir,
        words_path=args.words,
        device=args.device,
        target_seeds=args.seed,
        target_folds=args.fold,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
