"""Seed the 1M benchmark runs with prior 200K/500K checkpoints.

The 1M pools continue the 500K generation streams, so the 200K and 500K
expert data materialized from a 1M pool are training-identical to the
materialized data of the corresponding 500K pool: every field that the
training loader reads (``split``, ``example_type``, ``sampling_weight``,
``token_ids``, ``loss_ranges``) is equal; the only record difference is the
new metadata-only ``top_guesses`` field. This script verifies that
field-by-field for every fold, then copies the already-trained mechanics and
200K/500K checkpoints (plus their held-out raw/constrained predictions) so
the benchmark driver retrains only the new 1M variants.

Reproduce with:

    uv run --with-requirements requirements.txt python seed_1m_runs.py dev
    uv run --with-requirements requirements.txt python seed_1m_runs.py cv
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

TRAINING_FIELDS = (
    "split",
    "example_type",
    "sampling_weight",
    "token_ids",
    "loss_ranges",
)


def verify_materialized_equivalence(new_dir: Path, old_dir: Path) -> int:
    """Compare two materialized fold dirs on the training-relevant fields."""
    checked = 0
    with (
        gzip.open(new_dir / "examples.jsonl.gz", "rt", encoding="utf-8") as new,
        gzip.open(old_dir / "examples.jsonl.gz", "rt", encoding="utf-8") as old,
    ):
        for new_line, old_line in zip(new, old, strict=True):
            new_record = json.loads(new_line)
            old_record = json.loads(old_line)
            for field in TRAINING_FIELDS:
                if new_record[field] != old_record[field]:
                    raise AssertionError(
                        f"{new_dir} record {checked} field {field!r} differs "
                        f"from {old_dir}"
                    )
            checked += 1
    if checked == 0:
        raise AssertionError(f"no examples found in {new_dir}")
    return checked


def seed(which: str) -> int:
    if which == "dev":
        old_runs = Path("runs/dev-500k")
        new_runs = Path("runs/dev-1m")
        old_data = Path("data/wordle-dev-500k")
        new_data = Path("data/wordle-dev-1m")
        seeds = (0,)
    elif which == "cv":
        old_runs = Path("runs/cv5-500k")
        new_runs = Path("runs/cv5-1m")
        old_data = Path("data/wordle-cv5-500k")
        new_data = Path("data/wordle-cv5-1m")
        seeds = (0, 1, 2)
    else:
        raise SystemExit(f"unknown target: {which}")

    if not new_data.is_dir():
        raise SystemExit(f"materialize {new_data} first (benchmark_cv.py --prepare-only)")

    total = 0
    for seed in seeds:
        for run in range(1, 6 if which == "cv" else 2):
            for variant in ("mechanics", "expert-200k", "expert-500k"):
                new_dir = new_runs / f"seed-{seed}" / f"fold-{run}" / variant
                if new_dir.exists():
                    shutil.rmtree(new_dir)
                shutil.copytree(old_runs / f"seed-{seed}" / f"fold-{run}" / variant, new_dir)
                if variant.startswith("expert"):
                    checked = verify_materialized_equivalence(
                        new_data / f"fold-{run}" / variant,
                        old_data / f"fold-{run}" / variant,
                    )
                    total += checked
                    print(
                        f"seed-{seed}/fold-{run}/{variant}: {checked} examples "
                        "training-identical, checkpoint copied"
                    )
                else:
                    print(f"seed-{seed}/fold-{run}/{variant}: checkpoint copied")
    return 0


if __name__ == "__main__":
    raise SystemExit(seed(sys.argv[1] if len(sys.argv) > 1 else "cv"))
