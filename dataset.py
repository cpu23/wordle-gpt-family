from __future__ import annotations

import argparse
import gzip
import io
import hashlib
import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import IO, Iterator, Sequence

from trajectories import STRATEGIES, generate_trajectory, validate_trajectory
from wordle import DEFAULT_WORDS, filter_answers, load_words

SCHEMA_VERSION = 1
DEFAULT_PREFIXES = (1_000, 10_000, 30_000, 100_000)
SPLITS = ("train", "validation", "test")
SPLIT_BUCKETS = (80, 10, 10)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def state_fingerprint(history: Sequence[dict[str, object]]) -> str:
    """Identify an observable state by its ordered guess/feedback history."""
    observable_history = [
        {"guess": turn["guess"], "feedback": turn["feedback"]} for turn in history
    ]
    return hashlib.sha256(_json_bytes(observable_history)).hexdigest()


def _trajectory_split(seed: int, trajectory_key: str) -> str:
    digest = hashlib.sha256(f"{seed}:{trajectory_key}".encode("ascii")).digest()
    bucket = int.from_bytes(digest[:8], "big") % sum(SPLIT_BUCKETS)
    if bucket < SPLIT_BUCKETS[0]:
        return "train"
    if bucket < SPLIT_BUCKETS[0] + SPLIT_BUCKETS[1]:
        return "validation"
    return "test"


def _open_gzip_text(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed_file = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    return io.TextIOWrapper(compressed_file, encoding="utf-8", newline="\n")


def _word_list_hash(words: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(words) + "\n").encode("ascii")).hexdigest()


def _state_records(
    trajectory: dict[str, object],
    words: Sequence[str],
    trajectory_id: int,
    split: str,
) -> list[dict[str, object]]:
    possible = tuple(words)
    history: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    strategy = trajectory["strategy"]
    answer = trajectory["answer"]
    for turn in trajectory["turns"]:
        fingerprint = state_fingerprint(history)
        records.append(
            {
                "trajectory_id": trajectory_id,
                "strategy": strategy,
                "split": split,
                "turn": turn["turn"],
                "state_fingerprint": fingerprint,
                "history": list(history),
                "possible_answers": list(possible),
                "answer": answer,
                "action": turn["guess"],
                "feedback": turn["feedback"],
                "action_policy": turn["policy"],
            }
        )
        history.append({"guess": turn["guess"], "feedback": turn["feedback"]})
        possible = filter_answers(possible, turn["guess"], turn["feedback"])
    return records


def _validate_configuration(
    total_states: int, prefixes: Sequence[int], strategies: Sequence[str]
) -> tuple[int, ...]:
    if total_states < 1:
        raise ValueError("total_states must be at least 1")
    if not strategies:
        raise ValueError("at least one strategy is required")
    if len(set(strategies)) != len(strategies):
        raise ValueError("strategies must not contain duplicates")
    unknown = set(strategies) - set(STRATEGIES)
    if unknown:
        raise ValueError(f"unknown strategies: {', '.join(sorted(unknown))}")
    normalized = tuple(sorted(set(prefixes)))
    if not normalized or normalized[-1] != total_states:
        raise ValueError("prefixes must include total_states")
    if normalized[0] < 1 or normalized[-1] > total_states:
        raise ValueError("prefix sizes must be between 1 and total_states")
    if any(size % len(strategies) for size in normalized):
        raise ValueError("every prefix size must be divisible by the strategy count")
    return normalized


def build_dataset(
    output_dir: str | Path,
    words: Sequence[str],
    *,
    total_states: int = 100_000,
    prefixes: Sequence[int] = DEFAULT_PREFIXES,
    strategies: Sequence[str] = STRATEGIES,
    seed: int = 20260813,
    max_turns: int = 6,
    random_rate: float = 0.35,
) -> dict[str, object]:
    """Build one balanced state pool with nested prefix datasets.

    States are emitted in strategy round-robin order, so every configured
    prefix is exactly balanced. Splits are deterministic at trajectory level,
    preventing states from one game from crossing split boundaries.
    """
    normalized_prefixes = _validate_configuration(total_states, prefixes, strategies)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    states_path = output / "states.jsonl.gz"
    trajectories_path = output / "trajectories.jsonl.gz"
    manifest_path = output / "manifest.json"

    master_rng = random.Random(seed)
    strategy_rngs = {
        strategy: random.Random(master_rng.getrandbits(64)) for strategy in strategies
    }
    queues: dict[str, deque[dict[str, object]]] = {
        strategy: deque() for strategy in strategies
    }
    clever_cache: dict[tuple[str, ...], str] = {}
    strategy_trajectory_counts: Counter[str] = Counter()
    strategy_state_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    fingerprints: Counter[str] = Counter()
    prefix_statistics: dict[int, dict[str, object]] = {}
    trajectory_id = 0

    with _open_gzip_text(states_path) as states_file, _open_gzip_text(
        trajectories_path
    ) as trajectories_file:
        for state_index in range(total_states):
            strategy = strategies[state_index % len(strategies)]
            queue = queues[strategy]
            if not queue:
                rng = strategy_rngs[strategy]
                answer = rng.choice(words)
                trajectory = generate_trajectory(
                    answer,
                    words,
                    strategy,
                    rng,
                    max_turns=max_turns,
                    random_rate=random_rate,
                    clever_cache=clever_cache,
                )
                validate_trajectory(trajectory, words)
                trajectory_key = f"{strategy}:{strategy_trajectory_counts[strategy]}"
                split = _trajectory_split(seed, trajectory_key)
                saved_trajectory = {
                    "trajectory_id": trajectory_id,
                    "split": split,
                    **trajectory,
                }
                trajectories_file.write(
                    json.dumps(saved_trajectory, separators=(",", ":")) + "\n"
                )
                queue.extend(
                    _state_records(trajectory, words, trajectory_id, split)
                )
                strategy_trajectory_counts[strategy] += 1
                trajectory_id += 1

            state = queue.popleft()
            state["state_index"] = state_index
            states_file.write(json.dumps(state, separators=(",", ":")) + "\n")
            strategy_state_counts[strategy] += 1
            split_counts[state["split"]] += 1
            fingerprints[state["state_fingerprint"]] += 1

            current_size = state_index + 1
            if current_size in normalized_prefixes:
                unique_states = len(fingerprints)
                prefix_statistics[current_size] = {
                    "total_states": current_size,
                    "unique_states": unique_states,
                    "duplicate_states": current_size - unique_states,
                    "duplicate_rate": (current_size - unique_states) / current_size,
                    "strategy_counts": dict(strategy_state_counts),
                    "split_counts": dict(split_counts),
                }

    duplicate_occurrences = {
        fingerprint: count for fingerprint, count in fingerprints.items() if count > 1
    }
    duplicate_summary = sorted(
        duplicate_occurrences.items(), key=lambda item: (-item[1], item[0])
    )
    duplicates_path = output / "duplicates.jsonl.gz"
    with _open_gzip_text(duplicates_path) as duplicates_file:
        for fingerprint, count in duplicate_summary:
            duplicates_file.write(
                json.dumps(
                    {"state_fingerprint": fingerprint, "occurrences": count},
                    separators=(",", ":"),
                )
                + "\n"
            )

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "word_count": len(words),
        "word_list_sha256": _word_list_hash(words),
        "state_identity": "sha256 of ordered prior guess/feedback history",
        "total_states": total_states,
        "strategies": list(strategies),
        "target_strategy_share": 1 / len(strategies),
        "max_turns": max_turns,
        "partly_random_rate": random_rate,
        "split_method": "trajectory-level sha256(seed:strategy:strategy_trajectory_index)",
        "split_targets": {"train": 0.8, "validation": 0.1, "test": 0.1},
        "files": {
            "states": states_path.name,
            "trajectories": trajectories_path.name,
            "duplicates": duplicates_path.name,
        },
        "nested_datasets": {
            str(size): {
                "file": states_path.name,
                "state_index_range": [0, size],
                **prefix_statistics[size],
            }
            for size in normalized_prefixes
        },
        "trajectory_counts": dict(strategy_trajectory_counts),
        "duplicate_fingerprint_count": len(duplicate_occurrences),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def iter_dataset_states(
    output_dir: str | Path, *, limit: int | None = None, split: str | None = None
) -> Iterator[dict[str, object]]:
    """Read a nested prefix and/or fixed split from the single state pool."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    if split is not None and split not in SPLITS:
        raise ValueError(f"split must be one of: {', '.join(SPLITS)}")
    path = Path(output_dir) / "states.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as states_file:
        for state_index, line in enumerate(states_file):
            if limit is not None and state_index >= limit:
                break
            state = json.loads(line)
            if split is None or state["split"] == split:
                yield state


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a balanced, nested Wordle state dataset."
    )
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-states", type=int, default=100_000)
    parser.add_argument("--prefixes", nargs="+", type=int, default=list(DEFAULT_PREFIXES))
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--random-rate", type=float, default=0.35)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        words = load_words(args.words)
        manifest = build_dataset(
            args.output_dir,
            words,
            total_states=args.total_states,
            prefixes=args.prefixes,
            seed=args.seed,
            max_turns=args.max_turns,
            random_rate=args.random_rate,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2
    print(
        f"Saved {manifest['total_states']} balanced states to "
        f"{args.output_dir / 'states.jsonl.gz'}"
    )
    for size, statistics in manifest["nested_datasets"].items():
        print(
            f"{size}: {statistics['unique_states']} unique, "
            f"{statistics['duplicate_states']} duplicates"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
