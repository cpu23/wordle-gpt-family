from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wordle import DEFAULT_WORDS, load_words, score_guess

DEFAULT_SOURCE = Path("data/wordle-v2-diverse-1m/examples.jsonl.gz")
DEFAULT_MODE = Path("data/wordle-development.json")
DEFAULT_OUTPUT_DIR = Path("data/grpo-continuations")
DEFAULT_SIZE = 100_000
DEFAULT_VALIDATION_SIZE = 512
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ContinuationState:
    secret: str
    history: tuple[tuple[str, str], ...]
    source_index: int


def load_corpus(path: Path) -> list[ContinuationState]:
    """Load continuation states from the builder's JSONL format."""
    states: list[ContinuationState] = []
    seen_indices: set[int] = set()
    with Path(path).open(encoding="utf-8") as corpus_file:
        for line_number, line in enumerate(corpus_file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON in {path} on line {line_number}: {error}"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(f"corpus row {line_number} must be an object")
            secret = row.get("secret")
            raw_history = row.get("history")
            source_index = row.get("source_index")
            if not isinstance(secret, str) or not secret:
                raise ValueError(f"corpus row {line_number} has an invalid secret")
            if not isinstance(raw_history, list) or not 1 <= len(raw_history) <= 5:
                raise ValueError(
                    f"corpus row {line_number} history depth must be between 1 and 5"
                )
            if (
                not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or source_index < 0
            ):
                raise ValueError(f"corpus row {line_number} has an invalid source_index")
            if source_index in seen_indices:
                raise ValueError(f"duplicate source_index {source_index} in corpus")
            seen_indices.add(source_index)
            history: list[tuple[str, str]] = []
            for turn_number, turn in enumerate(raw_history, start=1):
                if not isinstance(turn, list) or len(turn) != 2:
                    raise ValueError(
                        f"corpus row {line_number} turn {turn_number} must be a pair"
                    )
                guess, feedback = turn
                if not isinstance(guess, str) or not isinstance(feedback, str):
                    raise ValueError(
                        f"corpus row {line_number} turn {turn_number} must contain strings"
                    )
                if guess == secret or feedback == "GGGGG":
                    raise ValueError(
                        f"corpus row {line_number} contains a solved history"
                    )
                history.append((guess, feedback))
            states.append(
                ContinuationState(
                    secret=secret,
                    history=tuple(history),
                    source_index=source_index,
                )
            )
    return states


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source_file:
        for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _split_secrets(
    mode_path: Path, words: set[str]
) -> tuple[dict[str, list[str]], dict[str, Any], Any]:
    try:
        payload = json.loads(Path(mode_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid mode JSON in {mode_path}: {error}") from error
    if not isinstance(payload, Mapping):
        raise ValueError("mode file must contain a JSON object")
    runs = payload.get("runs")
    if not isinstance(runs, list) or not runs or not isinstance(runs[0], Mapping):
        raise ValueError("mode file must contain a nonempty runs list of objects")
    first_run = runs[0]
    splits: dict[str, list[str]] = {}
    for name in ("train", "validation", "test"):
        raw_secrets = first_run.get(name, [])
        if name != "test" and not isinstance(raw_secrets, list):
            raise ValueError(f"runs[0].{name} must be a list of secrets")
        if name == "test" and raw_secrets is None:
            raw_secrets = []
        if not isinstance(raw_secrets, list):
            raise ValueError(f"runs[0].{name} must be a list of secrets")
        if any(not isinstance(secret, str) for secret in raw_secrets):
            raise ValueError(f"runs[0].{name} contains a non-string secret")
        if len(set(raw_secrets)) != len(raw_secrets):
            raise ValueError(f"runs[0].{name} contains duplicate secrets")
        illegal = sorted(set(raw_secrets) - words)
        if illegal:
            raise ValueError(
                f"runs[0].{name} contains secrets outside the word list: "
                + ", ".join(illegal[:5])
            )
        splits[name] = raw_secrets

    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    overlaps = {
        "train/validation": train & validation,
        "train/test": train & test,
        "validation/test": validation & test,
    }
    conflicting = [name for name, overlap in overlaps.items() if overlap]
    if conflicting:
        details = "; ".join(
            f"{name}: {', '.join(sorted(overlaps[name])[:5])}"
            for name in conflicting
        )
        raise ValueError(f"mode splits must be disjoint ({details})")
    return splits, first_run, payload.get("mode")


def _validate_history(
    raw_history: list[Any],
    secret: str,
    allowed_guesses: set[str],
    source_index: int,
) -> tuple[tuple[tuple[str, str], ...] | None, str | None]:
    history: list[tuple[str, str]] = []
    solved = False
    for turn_number, turn in enumerate(raw_history, start=1):
        if not isinstance(turn, Mapping):
            raise ValueError(
                f"source state {source_index} turn {turn_number} must be an object"
            )
        guess = turn.get("guess")
        feedback = turn.get("feedback")
        if not isinstance(guess, str) or guess not in allowed_guesses:
            raise ValueError(
                f"source state {source_index} turn {turn_number} has an illegal guess"
            )
        if not isinstance(feedback, str):
            raise ValueError(
                f"source state {source_index} turn {turn_number} has invalid feedback"
            )
        actual_feedback = score_guess(secret, guess)
        if feedback != actual_feedback:
            raise ValueError(
                f"source state {source_index} turn {turn_number} feedback "
                f"{feedback!r} does not match {actual_feedback!r}"
            )
        history.append((guess, feedback))
        if guess == secret:
            solved = True
    if solved:
        return None, "terminal"
    return tuple(history), None


def _append_reservoir(
    reservoir: list[tuple[ContinuationState, str]],
    item: tuple[ContinuationState, str],
    eligible_count: int,
    capacity: int,
    rng: random.Random,
) -> None:
    if len(reservoir) < capacity:
        reservoir.append(item)
        return
    replacement_index = rng.randrange(eligible_count)
    if replacement_index < capacity:
        reservoir[replacement_index] = item


def _selected_counts(
    selected: Sequence[tuple[ContinuationState, str]],
) -> dict[str, Any]:
    depths = Counter(len(state.history) for state, _ in selected)
    secrets = Counter(state.secret for state, _ in selected)
    behaviors = Counter(behavior for _, behavior in selected)
    return {
        "count": len(selected),
        "depth_counts": {str(depth): depths[depth] for depth in sorted(depths)},
        "secret_counts": dict(sorted(secrets.items())),
        "behavior_counts": dict(sorted(behaviors.items())),
    }


def _write_corpus(
    path: Path,
    selected: Sequence[tuple[ContinuationState, str]],
    combined_digest: Any,
) -> str:
    digest = hashlib.sha256()
    with path.open("xb") as corpus_file:
        for state, _ in selected:
            row = {
                "secret": state.secret,
                "history": [[guess, feedback] for guess, feedback in state.history],
                "source_index": state.source_index,
            }
            line = (json.dumps(row, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
                "utf-8"
            )
            corpus_file.write(line)
            digest.update(line)
            combined_digest.update(line)
    return digest.hexdigest()


def build_grpo_corpus(
    output_dir: Path,
    *,
    source: Path = DEFAULT_SOURCE,
    mode: Path = DEFAULT_MODE,
    words: Path = DEFAULT_WORDS,
    size: int = DEFAULT_SIZE,
    validation_size: int = DEFAULT_VALIDATION_SIZE,
    seed: int = 20260923,
) -> dict[str, Any]:
    """Reservoir-sample reachable train and held-out validation states."""
    output_dir = Path(output_dir)
    source = Path(source)
    mode = Path(mode)
    words = Path(words)
    if size < 1 or validation_size < 1:
        raise ValueError("size and validation_size must be positive")
    output_paths = {
        "train": output_dir / "train.jsonl",
        "validation": output_dir / "validation.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    existing = [path.name for path in output_paths.values() if path.exists()]
    if existing:
        raise FileExistsError(
            f"refusing to overwrite existing corpus files in {output_dir}: "
            + ", ".join(existing)
        )

    allowed_words = set(load_words(words))
    split_secrets, first_run, mode_name = _split_secrets(mode, allowed_words)
    split_by_secret = {
        secret: split_name
        for split_name in ("train", "validation")
        for secret in split_secrets[split_name]
    }
    secret_sets = {
        "train": set(split_secrets["train"]),
        "validation": set(split_secrets["validation"]),
        "test": set(split_secrets["test"]),
    }
    capacities = {"train": size, "validation": validation_size}
    reservoirs: dict[str, list[tuple[ContinuationState, str]]] = {
        "train": [],
        "validation": [],
    }
    eligible_counts = Counter()
    source_rows_by_split = Counter()
    excluded_rows = Counter()
    randomizers = {
        "train": random.Random(seed),
        "validation": random.Random(seed + 1),
    }
    source_rows = 0

    with gzip.open(source, "rt", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, start=1):
            if not line.strip():
                continue
            source_rows += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid source JSON on line {line_number}: {error}"
                ) from error
            if not isinstance(row, Mapping):
                raise ValueError(f"source row {line_number} must be an object")
            secret = row.get("source_secret")
            split_name = split_by_secret.get(secret) if isinstance(secret, str) else None
            if split_name is None:
                if isinstance(secret, str) and secret in secret_sets["test"]:
                    source_rows_by_split["test"] += 1
                else:
                    source_rows_by_split["outside_panel"] += 1
                continue
            source_rows_by_split[split_name] += 1
            raw_history = row.get("history")
            if not isinstance(raw_history, list):
                raise ValueError(
                    f"source row {line_number} for {secret!r} has invalid history"
                )
            depth = len(raw_history)
            if depth == 0:
                excluded_rows["empty_history"] += 1
                continue
            if depth > 5:
                excluded_rows["depth_over_five"] += 1
                continue
            source_index = row.get("state_index")
            if (
                not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or source_index < 0
            ):
                raise ValueError(
                    f"source row {line_number} has an invalid state_index"
                )
            behavior = row.get("source_behavior")
            if not isinstance(behavior, str) or not behavior:
                raise ValueError(
                    f"source state {source_index} has an invalid source_behavior"
                )
            history, exclusion_reason = _validate_history(
                raw_history, secret, allowed_words, source_index
            )
            if exclusion_reason is not None:
                excluded_rows[exclusion_reason] += 1
                continue
            assert history is not None
            state = ContinuationState(
                secret=secret,
                history=history,
                source_index=source_index,
            )
            eligible_counts[split_name] += 1
            _append_reservoir(
                reservoirs[split_name],
                (state, behavior),
                eligible_counts[split_name],
                capacities[split_name],
                randomizers[split_name],
            )

    for split_name, capacity in capacities.items():
        eligible = eligible_counts[split_name]
        if eligible < capacity:
            raise ValueError(
                f"not enough eligible {split_name} rows: requested {capacity}, "
                f"found {eligible}"
            )
    for selected in reservoirs.values():
        selected.sort(key=lambda item: item[0].source_index)
    all_source_indices = [
        state.source_index
        for selected in reservoirs.values()
        for state, _ in selected
    ]
    if len(set(all_source_indices)) != len(all_source_indices):
        raise ValueError("sampled corpus contains duplicate source indices")

    source_sha256 = _sha256_file(source)
    mode_sha256 = _sha256_file(mode)
    words_sha256 = _sha256_file(words)
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_paths.values():
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing file: {path}")
    combined_digest = hashlib.sha256()
    train_sha256 = _write_corpus(
        output_paths["train"], reservoirs["train"], combined_digest
    )
    validation_sha256 = _write_corpus(
        output_paths["validation"], reservoirs["validation"], combined_digest
    )
    held_out_secrets = sorted(secret_sets["validation"] | secret_sets["test"])
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source": {
            "path": str(source),
            "sha256": source_sha256,
        },
        "mode": {
            "path": str(mode),
            "sha256": mode_sha256,
            "name": mode_name,
            "run": first_run.get("run", 1),
        },
        "words": {
            "path": str(words),
            "sha256": words_sha256,
            "count": len(allowed_words),
        },
        "seed": seed,
        "sampling": {
            "algorithm": "independent reservoir sampling",
            "sizes": {"train": size, "validation": validation_size},
            "source_rows": source_rows,
            "source_rows_by_split": dict(sorted(source_rows_by_split.items())),
            "eligible_rows": sum(eligible_counts.values()),
            "eligible_rows_by_split": {
                name: eligible_counts[name] for name in ("train", "validation")
            },
            "excluded_rows": dict(sorted(excluded_rows.items())),
        },
        "selected_counts": {
            name: _selected_counts(reservoirs[name])
            for name in ("train", "validation")
        },
        "held_out_exclusion": {
            "train_secret_count": len(secret_sets["train"]),
            "train_secrets": sorted(secret_sets["train"]),
            "validation_secret_count": len(secret_sets["validation"]),
            "test_secret_count": len(secret_sets["test"]),
            "held_out_secrets_excluded_from_training": held_out_secrets,
            "validation_secrets_included_only_in_validation": True,
            "test_secrets_included": False,
        },
        "files": {
            "train.jsonl": {"rows": size, "sha256": train_sha256},
            "validation.jsonl": {
                "rows": validation_size,
                "sha256": validation_sha256,
            },
        },
        "corpus_sha256": combined_digest.hexdigest(),
    }
    with output_paths["manifest"].open("x", encoding="utf-8") as manifest_file:
        json.dump(manifest, manifest_file, indent=2, sort_keys=True)
        manifest_file.write("\n")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build reservoir-sampled reachable Wordle continuation corpora."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument(
        "--validation-size", type=int, default=DEFAULT_VALIDATION_SIZE
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mode", type=Path, default=DEFAULT_MODE)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_grpo_corpus(
        args.output_dir,
        source=args.source,
        mode=args.mode,
        words=args.words,
        size=args.size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
