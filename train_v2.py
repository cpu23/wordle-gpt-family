from __future__ import annotations

import gzip
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from train import IGNORE_INDEX, create_shifted_pairs

EXAMPLE_TYPES = ("mechanics", "expert")
EXAMPLE_TYPE_TO_ID = {name: index for index, name in enumerate(EXAMPLE_TYPES)}


@dataclass(frozen=True)
class V2SplitData:
    inputs: Tensor
    targets: Tensor
    example_type_ids: Tensor

    @property
    def supervised_token_count(self) -> int:
        return int((self.targets != IGNORE_INDEX).sum().item())


def load_v2_split(data_dir: str | Path, split: str) -> V2SplitData:
    """Load one v2 split, masking every target outside declared loss ranges."""
    token_sequences: list[list[int]] = []
    loss_ranges: list[list[list[int]]] = []
    example_type_ids: list[int] = []
    path = Path(data_dir) / "examples.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            example = json.loads(line)
            if example["split"] != split:
                continue
            try:
                example_type_id = EXAMPLE_TYPE_TO_ID[example["example_type"]]
            except KeyError as error:
                raise ValueError(f"unknown v2 example type: {error.args[0]!r}") from error
            token_sequences.append(example["token_ids"])
            loss_ranges.append(example["loss_ranges"])
            example_type_ids.append(example_type_id)
    if not token_sequences:
        raise ValueError(f"v2 dataset has no {split} examples")

    inputs, unmasked_targets = create_shifted_pairs(token_sequences)
    targets = torch.full_like(unmasked_targets, IGNORE_INDEX)
    for row, ranges in enumerate(loss_ranges):
        for start, stop in ranges:
            if not 0 <= start < stop <= len(token_sequences[row]) - 1:
                raise ValueError("v2 loss range is outside its next-token targets")
            targets[row, start:stop] = unmasked_targets[row, start:stop]
    if (targets != IGNORE_INDEX).sum().item() == 0:
        raise ValueError("v2 split has no supervised targets")
    return V2SplitData(
        inputs=inputs,
        targets=targets,
        example_type_ids=torch.tensor(example_type_ids, dtype=torch.int8),
    )
