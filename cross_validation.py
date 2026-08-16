from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from wordle import DEFAULT_WORDS, load_words

SCHEMA_VERSION = 1
FOLD_COUNT = 5
VALIDATION_SECRETS = 72
DEFAULT_SEED = 20260815
DEFAULT_FIXED_SPLITS = Path("data/wordle-100k/secret-splits.json")


@dataclass(frozen=True)
class SecretRun:
    run: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationMode:
    name: str
    runs: tuple[SecretRun, ...]
    model_seeds: tuple[int, ...]


def _stable_shuffle(values: Sequence[str], seed: str) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode("ascii")).digest(),
    )


def create_secret_folds(
    secrets: Sequence[str], *, seed: int = DEFAULT_SEED
) -> tuple[tuple[str, ...], ...]:
    """Partition every secret exactly once into deterministic 144/143 folds."""
    if len(secrets) < FOLD_COUNT:
        raise ValueError("cross-validation requires at least five secrets")
    if len(set(secrets)) != len(secrets):
        raise ValueError("secrets must be unique")
    shuffled = _stable_shuffle(secrets, f"folds:{seed}")
    base, remainder = divmod(len(shuffled), FOLD_COUNT)
    sizes = [base + (index < remainder) for index in range(FOLD_COUNT)]
    folds: list[tuple[str, ...]] = []
    cursor = 0
    for size in sizes:
        folds.append(tuple(shuffled[cursor : cursor + size]))
        cursor += size
    return tuple(folds)


def create_benchmark_mode(
    secrets: Sequence[str],
    *,
    split_seed: int = DEFAULT_SEED,
    model_seeds: Sequence[int] = (0,),
) -> EvaluationMode:
    """Build five held-out runs with 72 validation secrets inside each remainder."""
    folds = create_secret_folds(secrets, seed=split_seed)
    runs: list[SecretRun] = []
    for fold_index, test in enumerate(folds):
        test_set = set(test)
        remaining = [secret for secret in secrets if secret not in test_set]
        ordered = _stable_shuffle(remaining, f"validation:{split_seed}:{fold_index + 1}")
        validation = tuple(ordered[:VALIDATION_SECRETS])
        train = tuple(ordered[VALIDATION_SECRETS:])
        runs.append(
            SecretRun(
                run=fold_index + 1,
                train=train,
                validation=validation,
                test=test,
            )
        )
    mode = EvaluationMode("benchmark", tuple(runs), tuple(model_seeds))
    validate_evaluation_mode(mode, secrets)
    return mode


def create_development_mode(
    fixed_splits: Mapping[str, Sequence[str]], *, model_seed: int = 0
) -> EvaluationMode:
    """Use the fixed 575/72/72 split for fast one-seed iteration."""
    run = SecretRun(
        run=1,
        train=tuple(fixed_splits["train"]),
        validation=tuple(fixed_splits["validation"]),
        test=tuple(fixed_splits["test"]),
    )
    mode = EvaluationMode("development", (run,), (model_seed,))
    validate_evaluation_mode(mode, (*run.train, *run.validation, *run.test))
    return mode


def validate_evaluation_mode(
    mode: EvaluationMode, all_secrets: Sequence[str]
) -> None:
    universe = set(all_secrets)
    if not mode.model_seeds:
        raise ValueError("at least one model seed is required")
    held_out: list[str] = []
    for run in mode.runs:
        partitions = (set(run.train), set(run.validation), set(run.test))
        if any(left & right for index, left in enumerate(partitions) for right in partitions[index + 1 :]):
            raise ValueError(f"run {run.run} secret partitions overlap")
        if set.union(*partitions) != universe:
            raise ValueError(f"run {run.run} does not partition all secrets")
        if len(run.validation) != VALIDATION_SECRETS:
            raise ValueError(f"run {run.run} must have 72 validation secrets")
        held_out.extend(run.test)
    if mode.name == "benchmark":
        if len(mode.runs) != FOLD_COUNT:
            raise ValueError("benchmark mode requires five runs")
        if len(held_out) != len(universe) or set(held_out) != universe:
            raise ValueError("benchmark test folds must hold out every secret once")
    elif mode.name == "development" and len(mode.runs) != 1:
        raise ValueError("development mode requires one fixed run")


def mode_payload(mode: EvaluationMode) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "mode": mode.name,
        "model_seeds": list(mode.model_seeds),
        "runs": [asdict(run) for run in mode.runs],
    }


def load_mode(path: str | Path) -> EvaluationMode:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    runs = tuple(
        SecretRun(
            run=item["run"],
            train=tuple(item["train"]),
            validation=tuple(item["validation"]),
            test=tuple(item["test"]),
        )
        for item in payload["runs"]
    )
    mode = EvaluationMode(
        name=payload["mode"],
        runs=runs,
        model_seeds=tuple(payload["model_seeds"]),
    )
    universe = tuple(
        dict.fromkeys(
            secret
            for run in runs
            for secret in (*run.train, *run.validation, *run.test)
        )
    )
    validate_evaluation_mode(mode, universe)
    return mode


def write_mode(mode: EvaluationMode, path: str | Path) -> None:
    Path(path).write_text(
        json.dumps(mode_payload(mode), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def materialize_expert_fold(
    source_dir: str | Path,
    output_dir: str | Path,
    run: SecretRun,
    *,
    prefix: int | None = None,
) -> dict[str, object]:
    """Filter a universal expert pool without copying any held-out source secret."""
    train = set(run.train)
    validation = set(run.validation)
    test = set(run.test)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    omitted_test_examples = 0
    effective_counts: Counter[str] = Counter()
    source_path = Path(source_dir) / "examples.jsonl.gz"
    output_path = destination / "examples.jsonl.gz"
    with gzip.open(source_path, "rt", encoding="utf-8") as source, gzip.open(
        output_path, "wt", encoding="utf-8"
    ) as output:
        for line in source:
            example = json.loads(line)
            if prefix is not None and example["state_index"] >= prefix:
                break
            is_canonical_start = (
                example.get("example_type") == "expert"
                and not example.get("history")
            )
            secret = example["source_secret"]
            if is_canonical_start:
                split = "train"
                example["source_secret"] = None
                example["source_secret_role"] = "canonical-answer-independent-state"
            elif secret in test:
                omitted_test_examples += 1
                continue
            elif secret in train:
                split = "train"
            elif secret in validation:
                split = "validation"
            else:
                raise ValueError(f"unknown source secret: {secret}")
            example["split"] = split
            output.write(json.dumps(example, separators=(",", ":")) + "\n")
            counts[split] += 1
            effective_counts[split] += (
                int(example.get("sampling_weight", 1)) if split == "train" else 1
            )
    if not counts["train"] or not counts["validation"]:
        raise ValueError("materialized fold requires train and validation examples")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run": run.run,
        "source": str(source_path),
        "source_prefix": prefix,
        "split_counts": dict(sorted(counts.items())),
        "held_out_secrets": len(test),
        "effective_split_counts": dict(sorted(effective_counts.items())),
        "held_out_examples": 0,
        "omitted_test_examples": omitted_test_examples,
        "file": output_path.name,
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_results(path: str | Path) -> list[dict[str, object]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    results = payload["results"] if isinstance(payload, dict) else payload
    if not isinstance(results, list):
        raise ValueError("evaluation artifact must contain a results list")
    return results


def combine_fold_predictions(
    mode: EvaluationMode,
    fold_result_paths: Sequence[str | Path],
) -> dict[str, object]:
    """Combine one seed's five disjoint held-out prediction files."""
    if mode.name != "benchmark" or len(fold_result_paths) != len(mode.runs):
        raise ValueError("one result file is required for every benchmark fold")
    combined: list[dict[str, object]] = []
    seen: set[str] = set()
    for run, path in zip(mode.runs, fold_result_paths, strict=True):
        results = _load_results(path)
        expected = set(run.test)
        actual = {str(result["secret"]) for result in results}
        if actual != expected or len(results) != len(expected):
            raise ValueError(f"run {run.run} predictions do not match its test fold")
        if seen & actual:
            raise ValueError("a held-out secret was predicted more than once")
        seen.update(actual)
        combined.extend(results)
    wins = sum(bool(result["won"]) for result in combined)
    invalid = sum(int(result["invalid_guesses"]) for result in combined)
    won_guess_counts = [len(result["guesses"]) for result in combined if result["won"]]
    attempts = [len(result["guesses"]) for result in combined]
    return {
        "mode": "benchmark",
        "games": len(combined),
        "wins": wins,
        "win_rate": wins / len(combined),
        "average_guesses": statistics.fmean(won_guess_counts) if won_guess_counts else 0.0,
        "average_attempts": statistics.fmean(attempts),
        "invalid_guesses": invalid,
        "results": sorted(combined, key=lambda result: result["secret"]),
    }


def compare_paired_predictions(
    model_a: Mapping[str, object], model_b: Mapping[str, object]
) -> dict[str, object]:
    """Compare models secret-by-secret rather than comparing unpaired means."""
    a_results = {result["secret"]: result for result in model_a["results"]}
    b_results = {result["secret"]: result for result in model_b["results"]}
    if set(a_results) != set(b_results):
        raise ValueError("models must be evaluated on exactly the same secrets")
    transitions: Counter[str] = Counter()
    deltas: list[dict[str, object]] = []
    for secret in sorted(a_results):
        a = a_results[secret]
        b = b_results[secret]
        if a["won"] and b["won"]:
            outcome = "both_win"
        elif a["won"]:
            outcome = "a_wins_b_loses"
        elif b["won"]:
            outcome = "a_loses_b_wins"
        else:
            outcome = "both_lose"
        transitions[outcome] += 1
        deltas.append(
            {
                "secret": secret,
                "outcome": outcome,
                "a_guesses": len(a["guesses"]),
                "b_guesses": len(b["guesses"]),
                "guess_delta_b_minus_a": len(b["guesses"]) - len(a["guesses"]),
                "a_invalid_guesses": a["invalid_guesses"],
                "b_invalid_guesses": b["invalid_guesses"],
            }
        )
    return {
        "games": len(deltas),
        "outcomes": dict(sorted(transitions.items())),
        "mean_guess_delta_b_minus_a": statistics.fmean(
            result["guess_delta_b_minus_a"] for result in deltas
        ),
        "changed_games": [
            result
            for result in deltas
            if result["outcome"] not in {"both_win", "both_lose"}
            or result["guess_delta_b_minus_a"] != 0
            or result["a_invalid_guesses"] != result["b_invalid_guesses"]
        ],
        "per_secret": deltas,
    }


def aggregate_seed_summaries(
    summaries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Report mean and sample standard deviation across model initializations."""
    if not summaries:
        raise ValueError("at least one seed summary is required")
    metrics = ("wins", "win_rate", "average_guesses", "average_attempts", "invalid_guesses")
    aggregate: dict[str, object] = {"seeds": len(summaries)}
    for metric in metrics:
        values = [float(summary[metric]) for summary in summaries]
        aggregate[metric] = {
            "mean": statistics.fmean(values),
            "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return aggregate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Wordle development/CV splits.")
    parser.add_argument("--mode", choices=("development", "benchmark"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--fixed-splits", type=Path, default=DEFAULT_FIXED_SPLITS)
    parser.add_argument("--split-seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=(0,))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    words = load_words(args.words)
    if args.mode == "development":
        fixed = json.loads(args.fixed_splits.read_text(encoding="utf-8"))["splits"]
        mode = create_development_mode(fixed, model_seed=args.model_seeds[0])
    else:
        mode = create_benchmark_mode(
            words,
            split_seed=args.split_seed,
            model_seeds=args.model_seeds,
        )
    write_mode(mode, args.output)
    print(json.dumps(mode_payload(mode), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
