"""Reassemble the 1M expert-pool archives from their committed split parts.

GitHub rejects files larger than 100 MB, so the two 1M pool archives are
committed as ``examples.jsonl.gz.part-0`` + ``examples.jsonl.gz.part-1`` in
their data directories. This script concatenates the parts, verifies the
result against the SHA-256 pinned in each pool's ``manifest.json``, and
overwrites the local ``examples.jsonl.gz`` (the working artifact the build
and benchmark tooling read).

The 500K pools are below the limit and are committed as single files.

    uv run --with-requirements requirements.txt python join_pools.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

POOLS = (
    Path("data/wordle-v2-diverse-1m"),
    Path("data/wordle-v2-diverse-1m-cv"),
)


def join_pool(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected = manifest["file_sha256"]
    target = directory / "examples.jsonl.gz"
    digest = hashlib.sha256()
    with target.open("wb") as out:
        for part_index in (0, 1):
            part = directory / f"examples.jsonl.gz.part-{part_index}"
            with part.open("rb") as source:
                while chunk := source.read(1 << 20):
                    out.write(chunk)
                    digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"{directory}: joined SHA-256 {actual} != manifest {expected}"
        )
    print(
        f"{directory}: reassembled {target.name} "
        f"({target.stat().st_size} bytes, SHA-256 ok)"
    )


if __name__ == "__main__":
    for pool in POOLS:
        if not (pool / "examples.jsonl.gz.part-0").exists():
            raise SystemExit(f"missing split parts for {pool}")
    for pool in POOLS:
        join_pool(pool)
    sys.exit(0)
