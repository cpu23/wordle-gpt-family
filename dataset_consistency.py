from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from tokenizer import END_TOKEN, FEEDBACK_TOKEN, serialize_trajectory
from tokenizer_v2 import MECHANICS_TOKEN, SECRET_TOKEN, VOCABULARY_SIZE, encode
from wordle import DEFAULT_WORDS, load_words

SCHEMA_VERSION = 1
DEFAULT_SOURCE_DIR = Path("data/wordle-100k")
DEFAULT_OUTPUT_DIR = Path("data/wordle-v2-consistency")
POSITIVE_LABEL = "2"
NEGATIVE_LABEL = "0"


def _open_gzip_text(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed_file = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    return io.TextIOWrapper(compressed_file, encoding="utf-8", newline="\n")


def _choose_negative(
    words: Sequence[str], possible_answers: frozenset[str], state_index: int
) -> str | None:
    for offset in range(len(words)):
        candidate = words[(state_index + offset) % len(words)]
        if candidate not in possible_answers:
            return candidate
    return None


def _example(
    state: dict[str, object],
    *,
    candidate: str,
    label: str,
) -> dict[str, object]:
    history_text = serialize_trajectory(state["history"])
    text = (
        MECHANICS_TOKEN
        + SECRET_TOKEN
        + candidate
        + history_text[: -len(END_TOKEN)]
        + FEEDBACK_TOKEN
        + label
        + END_TOKEN
    )
    token_ids = encode(text)
    label_target_start = len(token_ids) - 3
    return {
        "state_index": state["state_index"],
        "split": state["split"],
        "example_type": "consistency",
        "candidate": candidate,
        "consistent": label == POSITIVE_LABEL,
        "text": text,
        "token_ids": token_ids,
        "loss_ranges": [[label_target_start, label_target_start + 1]],
    }


def build_candidate_consistency_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    words: Sequence[str],
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Build balanced candidate-membership examples from observable Wordle states."""
    if limit is not None and limit < 0:
        raise ValueError("limit must be non-negative")
    source_path = Path(source_dir) / "states.jsonl.gz"
    destination = Path(output_dir)
    examples_path = destination / "examples.jsonl.gz"
    split_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    source_states = 0
    eligible_states = 0
    with gzip.open(source_path, "rt", encoding="utf-8") as source, _open_gzip_text(
        examples_path
    ) as output:
        for source_index, line in enumerate(source):
            if limit is not None and source_index >= limit:
                break
            state = json.loads(line)
            if state["state_index"] != source_index:
                raise ValueError("source state indices are not contiguous")
            source_states += 1
            possible_answers = tuple(state["possible_answers"])
            possible_set = frozenset(possible_answers)
            negative = _choose_negative(words, possible_set, source_index)
            if not possible_answers or negative is None:
                continue
            eligible_states += 1
            positive = possible_answers[source_index % len(possible_answers)]
            for candidate, label in (
                (positive, POSITIVE_LABEL),
                (negative, NEGATIVE_LABEL),
            ):
                record = _example(state, candidate=candidate, label=label)
                output.write(json.dumps(record, separators=(",", ":")) + "\n")
                split_counts[record["split"]] += 1
                label_counts[label] += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "objective": "Given observable history and a supplied candidate, predict whether the candidate remains consistent.",
        "format": "<M><S>candidate + observable history + <F> + binary feedback label + <E>",
        "labels": {
            NEGATIVE_LABEL: "candidate is inconsistent",
            POSITIVE_LABEL: "candidate is consistent",
        },
        "source": str(source_path),
        "source_states": source_states,
        "eligible_states": eligible_states,
        "examples": eligible_states * 2,
        "split_counts": dict(sorted(split_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "supervised_targets": eligible_states * 2,
        "vocabulary_size": VOCABULARY_SIZE,
        "file": examples_path.name,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build balanced Wordle candidate-consistency examples."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_candidate_consistency_dataset(
        args.source_dir,
        args.output_dir,
        load_words(args.words),
        limit=args.limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
