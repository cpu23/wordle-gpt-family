"""Build the 500K unique-state expert pools (development and CV variants).

Both pools continue the exact generation stream of the existing 200K pools
(``data/wordle-v2-diverse`` and ``data/wordle-v2-diverse-cv``): same seed,
same behaviors, same state order. The first 200,000 states of each pool are
field-identical to the corresponding 200K pool (the 200K files use an older
JSON key order but identical record fields).

States from index 200,000 onward are generated with
``repeat_probability=0.5``: a state whose remaining-answer-set fingerprint was
already seen is stored only with probability 0.5 (one uniform RNG draw per
skipped candidate), biasing the tail toward novel candidate sets while the
200K prefix stream stays untouched.

The development pool contains only train/validation source secrets; the CV
pool contains all 719 secrets so every CV fold can filter held-out secrets.

Each build is followed by a full revalidation of every stored record
(unique fingerprint, reachable feedback, secret still candidate, clever
target, serialization, token IDs, loss mask, split metadata) written to
``validation.json``.

Reproduce with:

    uv run --with-requirements requirements.txt python build_500k_pools.py dev
    uv run --with-requirements requirements.txt python build_500k_pools.py cv
"""

from __future__ import annotations

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
TOTAL_STATES = 500_000
PREFIXES = (10_000, 50_000, 100_000, 200_000, 500_000)
REPEAT_PROBABILITY = 0.5
BIAS_PREFIX = 200_000

OUTPUTS = {
    "dev": ("data/wordle-v2-diverse-500k", ("train", "validation")),
    "cv": ("data/wordle-v2-diverse-500k-cv", ("train", "validation", "test")),
}


def main() -> int:
    which = sys.argv[1:] or list(OUTPUTS)
    for name in which:
        if name not in OUTPUTS:
            raise SystemExit(f"unknown pool: {name}")
    payload = json.loads(SPLITS.read_text(encoding="utf-8"))
    for name in which:
        output, split_names = OUTPUTS[name]
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
        )
        validation = validate_diverse_expert_dataset(
            output,
            words=load_words(WORDS),
            split_secrets=split_secrets,
        )
        (Path(output) / "validation.json").write_text(
            json.dumps(validation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"{name}: {manifest['unique_states']} states, "
            f"{manifest['answer_sets']['unique_answer_sets']} unique answer sets, "
            f"splits={manifest['split_counts']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
