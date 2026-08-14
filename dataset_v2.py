from __future__ import annotations

import argparse
import gzip
import io
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import IO

from tokenizer import (
    END_TOKEN,
    FEEDBACK_TO_SYMBOL,
    FEEDBACK_TOKEN,
    GUESS_TOKEN,
    serialize_trajectory,
)
from tokenizer_v2 import (
    MECHANICS_TOKEN,
    POLICY_TOKEN,
    SECRET_TOKEN,
    VOCABULARY_SIZE,
    encode,
)
from wordle import DEFAULT_WORDS, choose_informative_guess, load_words, score_guess

SCHEMA_VERSION = 2
DEFAULT_SOURCE_DIR = Path("data/wordle-100k")
DEFAULT_OUTPUT_DIR = Path("data/wordle-v2")


def _open_gzip_text(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed_file = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    return io.TextIOWrapper(compressed_file, encoding="utf-8", newline="\n")


def _feedback_symbols(feedback: str) -> str:
    try:
        return "".join(FEEDBACK_TO_SYMBOL[mark] for mark in feedback)
    except KeyError as error:
        raise ValueError(f"invalid feedback mark: {error.args[0]!r}") from error


def _mechanics_example(state: dict[str, object]) -> dict[str, object]:
    answer = state["answer"]
    guess = state["action"]
    feedback = state["feedback"]
    if score_guess(answer, guess) != feedback:
        raise ValueError("source state feedback does not match its secret and guess")
    text = (
        f"{MECHANICS_TOKEN}{SECRET_TOKEN}{answer}{GUESS_TOKEN}{guess}"
        f"{FEEDBACK_TOKEN}{_feedback_symbols(feedback)}{END_TOKEN}"
    )
    token_ids = encode(text)
    return {
        "schema_version": SCHEMA_VERSION,
        "example_type": "mechanics",
        "source_state_index": state["state_index"],
        "split": state["split"],
        "strategy": state["strategy"],
        "action_policy": state["action_policy"],
        "text": text,
        "token_ids": token_ids,
        "loss_ranges": [[13, 18]],
        "supervised_target": "feedback",
    }


def _expert_example(
    state: dict[str, object],
    words: Sequence[str],
    expert_cache: dict[tuple[str, ...], str],
) -> dict[str, object]:
    possible_answers = tuple(state["possible_answers"])
    expert_guess = expert_cache.get(possible_answers)
    if expert_guess is None:
        expert_guess = choose_informative_guess(possible_answers, words)
        expert_cache[possible_answers] = expert_guess
    history = state["history"]
    history_text = serialize_trajectory(history)
    text = (
        POLICY_TOKEN
        + history_text[: -len(END_TOKEN)]
        + GUESS_TOKEN
        + expert_guess
        + END_TOKEN
    )
    token_ids = encode(text)
    guess_target_start = 1 + len(history) * 12
    return {
        "schema_version": SCHEMA_VERSION,
        "example_type": "expert",
        "source_state_index": state["state_index"],
        "split": state["split"],
        "strategy": state["strategy"],
        "source_action_policy": state["action_policy"],
        "expert_policy": "clever",
        "desired_guess": expert_guess,
        "text": text,
        "token_ids": token_ids,
        "loss_ranges": [[guess_target_start, guess_target_start + 5]],
        "supervised_target": "expert_guess_letters",
    }


def build_v2_dataset(
    source_dir: str | Path,
    output_dir: str | Path,
    words: Sequence[str],
    *,
    limit: int | None = None,
) -> dict[str, object]:
    """Build mechanics and clever-label examples with explicit loss masks."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be positive")

    source_path = Path(source_dir) / "states.jsonl.gz"
    output = Path(output_dir)
    examples_path = output / "examples.jsonl.gz"
    expert_cache: dict[tuple[str, ...], str] = {}
    split_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    strategy_counts: Counter[str] = Counter()
    action_policy_counts: Counter[str] = Counter()
    state_count = 0

    with gzip.open(source_path, "rt", encoding="utf-8") as source, _open_gzip_text(
        examples_path
    ) as destination:
        for source_index, line in enumerate(source):
            if limit is not None and source_index >= limit:
                break
            state = json.loads(line)
            if state["state_index"] != source_index:
                raise ValueError("source state indices are not contiguous")
            mechanics = _mechanics_example(state)
            expert = _expert_example(state, words, expert_cache)
            for example in (mechanics, expert):
                destination.write(json.dumps(example, separators=(",", ":")) + "\n")
                split_counts[example["split"]] += 1
                type_counts[example["example_type"]] += 1
            action_policy_counts[state["action_policy"]] += 1
            strategy_counts[state["strategy"]] += 1
            state_count += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": str(source_path),
        "source_states": state_count,
        "role_format_version": 2,
        "vocabulary_size": VOCABULARY_SIZE,
        "examples": state_count * 2,
        "file": examples_path.name,
        "example_type_counts": dict(sorted(type_counts.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "source_strategy_counts": dict(sorted(strategy_counts.items())),
        "source_action_policy_counts": dict(sorted(action_policy_counts.items())),
        "unique_expert_states": len(expert_cache),
        "objectives": {
            "mechanics": "Given secret and guess, predict only five feedback tokens.",
            "expert": "Given observable history, predict only the clever solver's next-guess letters.",
        },
        "loss_ranges": "Half-open next-token target indices included in cross-entropy.",
        "excluded_targets": (
            "History, supplied secrets and guesses, structural markers, end tokens, "
            "environment feedback without a supplied secret, and source-policy actions."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build the masked mechanics/expert Wordle dataset v2."
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--limit", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_v2_dataset(
        args.source_dir,
        args.output_dir,
        load_words(args.words),
        limit=args.limit,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
