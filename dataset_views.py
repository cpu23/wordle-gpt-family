from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import random
from collections import Counter
from pathlib import Path
from typing import IO, Iterator, Sequence

from tokenizer import TOKENS, VOCABULARY_SIZE, decode, encode, serialize_trajectory
from wordle import DEFAULT_WORDS, load_words

SCHEMA_VERSION = 1
SPLIT_COUNTS = {"train": 575, "validation": 72, "test": 72}
RAW_FILENAMES = (
    "manifest.json",
    "states.jsonl.gz",
    "trajectories.jsonl.gz",
    "duplicates.jsonl.gz",
)


def _open_gzip_text(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed_file = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    return io.TextIOWrapper(compressed_file, encoding="utf-8", newline="\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def create_secret_splits(
    words: Sequence[str], seed: int
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Split all 719 secret words into deterministic 575/72/72 partitions."""
    if len(words) != sum(SPLIT_COUNTS.values()):
        raise ValueError(
            f"secret split expects {sum(SPLIT_COUNTS.values())} words, got {len(words)}"
        )
    shuffled = list(words)
    random.Random(seed).shuffle(shuffled)
    splits: dict[str, list[str]] = {}
    word_to_split: dict[str, str] = {}
    offset = 0
    for split, count in SPLIT_COUNTS.items():
        secrets = sorted(shuffled[offset : offset + count])
        splits[split] = secrets
        word_to_split.update((secret, split) for secret in secrets)
        offset += count
    return splits, word_to_split


def _iter_jsonl_gzip(path: Path) -> Iterator[dict[str, object]]:
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def _deduplicate_states(
    raw_states_path: Path,
    output_path: Path,
    word_to_split: dict[str, str],
) -> tuple[int, int, int]:
    aggregated: dict[str, dict[str, object]] = {}
    total_states = 0
    for state in _iter_jsonl_gzip(raw_states_path):
        total_states += 1
        fingerprint = state["state_fingerprint"]
        record = aggregated.get(fingerprint)
        if record is None:
            record = {
                "state_fingerprint": fingerprint,
                "state": {
                    "history": state["history"],
                    "possible_answers": state["possible_answers"],
                },
                "number_of_occurrences": 0,
                "first_state_index": state["state_index"],
                "strategy_counts": Counter(),
                "action_policy_counts": Counter(),
                "raw_split_counts": Counter(),
                "secret_split_counts": Counter(),
                "answer_counts": Counter(),
            }
            aggregated[fingerprint] = record
        elif record["state"]["history"] != state["history"]:
            raise ValueError("fingerprint collision between different histories")

        record["number_of_occurrences"] += 1
        record["strategy_counts"][state["strategy"]] += 1
        record["action_policy_counts"][state["action_policy"]] += 1
        record["raw_split_counts"][state["split"]] += 1
        record["secret_split_counts"][word_to_split[state["answer"]]] += 1
        record["answer_counts"][state["answer"]] += 1

    ordered = sorted(aggregated.values(), key=lambda item: item["first_state_index"])
    with _open_gzip_text(output_path) as output:
        for record in ordered:
            strategy_counts = dict(sorted(record.pop("strategy_counts").items()))
            action_policy_counts = dict(
                sorted(record.pop("action_policy_counts").items())
            )
            record["strategies"] = list(strategy_counts)
            record["action_policies"] = list(action_policy_counts)
            record["strategy_counts"] = strategy_counts
            record["action_policy_counts"] = action_policy_counts
            record["raw_split_counts"] = dict(
                sorted(record["raw_split_counts"].items())
            )
            record["secret_split_counts"] = dict(
                sorted(record["secret_split_counts"].items())
            )
            record["answer_counts"] = dict(sorted(record["answer_counts"].items()))
            output.write(json.dumps(record, separators=(",", ":")) + "\n")

    unique_states = len(aggregated)
    return total_states, unique_states, total_states - unique_states


def _empty_sequence_stats() -> dict[str, object]:
    return {
        "trajectory_count": 0,
        "total_tokens": 0,
        "shortest_sequence_length": None,
        "longest_sequence_length": 0,
        "average_sequence_length": 0.0,
    }


def _update_sequence_stats(stats: dict[str, object], length: int) -> None:
    stats["trajectory_count"] += 1
    stats["total_tokens"] += length
    shortest = stats["shortest_sequence_length"]
    if shortest is None or length < shortest:
        stats["shortest_sequence_length"] = length
    if length > stats["longest_sequence_length"]:
        stats["longest_sequence_length"] = length


def _finalize_sequence_stats(stats: dict[str, object]) -> None:
    count = stats["trajectory_count"]
    stats["average_sequence_length"] = stats["total_tokens"] / count if count else 0.0


def _tokenize_trajectories(
    trajectories_path: Path,
    output_path: Path,
    word_to_split: dict[str, str],
) -> tuple[dict[str, dict[str, object]], Counter[str]]:
    split_stats = {split: _empty_sequence_stats() for split in SPLIT_COUNTS}
    split_stats["all"] = _empty_sequence_stats()
    strategy_counts: Counter[str] = Counter()
    trajectory_split: dict[int, str] = {}

    with _open_gzip_text(output_path) as output:
        for trajectory in _iter_jsonl_gzip(trajectories_path):
            split = word_to_split[trajectory["answer"]]
            trajectory_id = trajectory["trajectory_id"]
            previous_split = trajectory_split.setdefault(trajectory_id, split)
            if previous_split != split:
                raise ValueError("trajectory appears in multiple secret splits")
            text = serialize_trajectory(trajectory["turns"])
            token_ids = encode(text)
            if decode(token_ids) != text:
                raise ValueError("tokenizer failed its round-trip invariant")
            record = {
                "trajectory_id": trajectory_id,
                "strategy": trajectory["strategy"],
                "secret_split": split,
                "answer": trajectory["answer"],
                "text": text,
                "token_ids": token_ids,
            }
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            strategy_counts[trajectory["strategy"]] += 1
            _update_sequence_stats(split_stats[split], len(token_ids))
            _update_sequence_stats(split_stats["all"], len(token_ids))

    for stats in split_stats.values():
        _finalize_sequence_stats(stats)
    return split_stats, strategy_counts


def build_comparison_views(
    dataset_dir: str | Path,
    words: Sequence[str],
    *,
    seed: int = 20260813,
) -> dict[str, object]:
    """Add deduplicated, secret-split, and tokenized views without changing raw data."""
    root = Path(dataset_dir)
    raw_checksums_before = {
        filename: _sha256_file(root / filename) for filename in RAW_FILENAMES
    }
    splits, word_to_split = create_secret_splits(words, seed)

    secret_splits_path = root / "secret-splits.json"
    deduplicated_path = root / "deduplicated-states.jsonl.gz"
    tokenized_path = root / "tokenized-trajectories.jsonl.gz"
    view_manifest_path = root / "view-manifest.json"

    secret_splits_document = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "method": "shuffle all secret words with random.Random(seed), then take 575/72/72",
        "counts": SPLIT_COUNTS,
        "splits": splits,
    }
    secret_splits_path.write_text(
        json.dumps(secret_splits_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    total_states, unique_states, duplicate_states = _deduplicate_states(
        root / "states.jsonl.gz", deduplicated_path, word_to_split
    )
    sequence_stats, trajectory_strategy_counts = _tokenize_trajectories(
        root / "trajectories.jsonl.gz", tokenized_path, word_to_split
    )

    raw_checksums_after = {
        filename: _sha256_file(root / filename) for filename in RAW_FILENAMES
    }
    if raw_checksums_after != raw_checksums_before:
        raise RuntimeError("raw dataset changed while comparison views were built")

    manifest: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "raw_artifact_sha256": raw_checksums_before,
        "raw_pool": {
            "file": "states.jsonl.gz",
            "total_states": total_states,
            "frequency_preserved": True,
        },
        "deduplicated_pool": {
            "file": deduplicated_path.name,
            "unique_states": unique_states,
            "duplicate_states_removed": duplicate_states,
            "record_fields": [
                "state",
                "number_of_occurrences",
                "strategies",
                "action_policies",
                "strategy_counts",
                "action_policy_counts",
                "secret_split_counts",
                "answer_counts",
            ],
        },
        "secret_split": {
            "file": secret_splits_path.name,
            "counts": SPLIT_COUNTS,
            "trajectory_atomic": True,
        },
        "tokenizer": {
            "tokens_by_id": list(TOKENS),
            "vocabulary_size": VOCABULARY_SIZE,
            "feedback_encoding": {"gray": "0", "yellow": "1", "green": "2"},
            "control_tokens": {"guess": "<G>", "feedback": "<F>", "end": "<E>"},
        },
        "tokenized_trajectories": {
            "file": tokenized_path.name,
            "sequence_statistics": sequence_stats,
            "strategy_counts": dict(sorted(trajectory_strategy_counts.items())),
        },
    }
    view_manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deduplicated and tokenized views of a Wordle dataset."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--seed", type=int, default=20260813)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        words = load_words(args.words)
        manifest = build_comparison_views(args.dataset_dir, words, seed=args.seed)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"error: {error}")
        return 2
    deduplicated = manifest["deduplicated_pool"]
    training = manifest["tokenized_trajectories"]["sequence_statistics"]["train"]
    print(
        f"Raw: {manifest['raw_pool']['total_states']} states; "
        f"deduplicated: {deduplicated['unique_states']} states"
    )
    print(
        f"Vocabulary: {manifest['tokenizer']['vocabulary_size']}; "
        f"training tokens: {training['total_tokens']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
