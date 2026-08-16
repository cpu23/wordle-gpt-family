from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from dataset_v2 import _mechanics_example
from wordle import DEFAULT_WORDS, load_words, score_guess

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("data/wordle-v2-mechanics-cv")


def build_mechanics_cv_pool(
    output_dir: str | Path,
    words: Sequence[str],
    *,
    total_examples: int = 100_000,
    seed: int = 20260815,
) -> dict[str, object]:
    """Build unique secret/guess mechanics pairs that CV runs can filter by secret."""
    maximum = len(words) * len(words)
    if not 1 <= total_examples <= maximum:
        raise ValueError(f"total_examples must be between 1 and {maximum}")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples_path = output / "examples.jsonl.gz"
    rng = random.Random(seed)
    used_pairs: set[tuple[str, str]] = set()
    secret_counts: Counter[str] = Counter()
    with gzip.open(examples_path, "wt", encoding="utf-8") as destination:
        while len(used_pairs) < total_examples:
            state_index = len(used_pairs)
            secret = words[state_index % len(words)]
            guess = rng.choice(words)
            pair = (secret, guess)
            if pair in used_pairs:
                continue
            used_pairs.add(pair)
            state = {
                "state_index": state_index,
                "split": "universal",
                "strategy": "random-pair",
                "action_policy": "random-legal",
                "answer": secret,
                "action": guess,
                "feedback": score_guess(secret, guess),
            }
            example = _mechanics_example(state)
            example["source_secret"] = secret
            destination.write(json.dumps(example, separators=(",", ":")) + "\n")
            secret_counts[secret] += 1
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "file": examples_path.name,
        "total_examples": total_examples,
        "unique_secret_guess_pairs": len(used_pairs),
        "source_secrets": len(secret_counts),
        "minimum_examples_per_secret": min(secret_counts.values()),
        "maximum_examples_per_secret": max(secret_counts.values()),
        "seed": seed,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a CV-filterable mechanics pool.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--total-examples", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260815)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = build_mechanics_cv_pool(
        args.output_dir,
        load_words(args.words),
        total_examples=args.total_examples,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
