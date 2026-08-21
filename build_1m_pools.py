"""Build the 1M unique-state expert pools (development and CV variants).

Both pools continue the exact generation stream of the 500K pools
(``data/wordle-v2-diverse-500k`` and ``data/wordle-v2-diverse-500k-cv``):
same seed, same behaviors, same state order, same answer-set bias
(``repeat_probability=0.5`` from state 200,000 onward). The first
500,000 states of each pool are field-identical to the corresponding 500K
pool; this script verifies that after every build.

Every record additionally stores ``top_guesses``: the eight legal guesses
ranked by the minimum-expected-survivors solver together with their
expected-survivor scores. The field is metadata only -- it never influences
generation, the expert target, training, or evaluation.

The development pool contains only train/validation source secrets; the CV
pool contains all 719 secrets so every CV fold can filter held-out secrets.

Each build is followed by a full revalidation of every stored record
(unique fingerprint, reachable feedback, secret still candidate, clever
target including the stored top actions, serialization, token IDs, loss
mask, split metadata) written to ``validation.json``.

Reproduce with:

    uv run --with-requirements requirements.txt python build_1m_pools.py dev
    uv run --with-requirements requirements.txt python build_1m_pools.py cv
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from dataset_expert import (
    build_diverse_expert_dataset,
    validate_diverse_expert_dataset,
)
from wordle import load_words

WORDS = Path("words.txt")
SPLITS = Path("data/wordle-100k/secret-splits.json")
SEED = 20260815
TOTAL_STATES = 1_000_000
PREFIXES = (10_000, 50_000, 100_000, 200_000, 500_000, 1_000_000)
REPEAT_PROBABILITY = 0.5
BIAS_PREFIX = 200_000
TOP_K = 8

OUTPUTS = {
    "dev": (
        "data/wordle-v2-diverse-1m",
        ("train", "validation"),
        "data/wordle-v2-diverse-500k",
    ),
    "cv": (
        "data/wordle-v2-diverse-1m-cv",
        ("train", "validation", "test"),
        "data/wordle-v2-diverse-500k-cv",
    ),
}


def verify_prefix(new_dir: str, old_dir: str, count: int) -> int:
    """Check that the first ``count`` new records keep every old field.

    The 1M records may add the ``top_guesses`` metadata field; every field
    present in the 500K record must be byte-equal in the 1M record.
    """
    checked = 0
    with (
        gzip.open(Path(new_dir) / "examples.jsonl.gz", "rt", encoding="utf-8") as new,
        gzip.open(Path(old_dir) / "examples.jsonl.gz", "rt", encoding="utf-8") as old,
    ):
        for new_line, old_line in zip(new, old, strict=True):
            new_record = json.loads(new_line)
            old_record = json.loads(old_line)
            for key, value in old_record.items():
                if new_record.get(key) != value:
                    raise AssertionError(
                        f"state {checked} field {key!r} differs from the 500K pool"
                    )
            if len(new_record.get("top_guesses", [])) != TOP_K:
                raise AssertionError(
                    f"state {checked} does not store {TOP_K} top guesses"
                )
            checked += 1
            if checked >= count:
                break
    if checked < count:
        raise AssertionError("new pool is shorter than the verified prefix")
    return checked


def main() -> int:
    which = sys.argv[1:] or list(OUTPUTS)
    for name in which:
        if name not in OUTPUTS:
            raise SystemExit(f"unknown pool: {name}")
    payload = json.loads(SPLITS.read_text(encoding="utf-8"))
    for name in which:
        output, split_names, parent = OUTPUTS[name]
        split_secrets = {key: payload["splits"][key] for key in split_names}
        manifest = build_diverse_expert_dataset(
            output,
            words=load_words(WORDS),
            split_secrets=split_secrets,
            total_states=TOTAL_STATES,
            prefixes=PREFIXES,
            seed=SEED,
            repeat_probability=REPEAT_PROBABILITY,
            bias_prefix=BIAS_PREFIX,
            top_guesses=TOP_K,
        )
        validation = validate_diverse_expert_dataset(
            output,
            words=load_words(WORDS),
            split_secrets=split_secrets,
        )
        checked = verify_prefix(output, parent, 500_000)
        (Path(output) / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{name}: {manifest['unique_states']} states, "
            f"{manifest['answer_sets']['unique_answer_sets']} unique answer sets, "
            f"{checked} prefix states verified against {parent}, "
            f"splits={manifest['split_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
