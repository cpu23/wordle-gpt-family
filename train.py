from __future__ import annotations

import argparse
import gzip
import json
import random
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from model import CONTEXT_LENGTH, VOCAB_SIZE, WordleGPT
from tokenizer import END_TOKEN, TOKEN_TO_ID

DEFAULT_DATA = Path("data/wordle-100k/tokenized-trajectories.jsonl.gz")
DEFAULT_BATCH_SIZE = 32
DEFAULT_STEPS = 1000
DEFAULT_LEARNING_RATE = 3e-4
IGNORE_INDEX = -100


def load_trajectories(
    path: str | Path,
    *,
    count: int = DEFAULT_BATCH_SIZE,
    split: str = "train",
    seed: int = 0,
) -> list[list[int]]:
    """Select a reproducible set of tokenized trajectories from one split."""
    if count < 1:
        raise ValueError("count must be positive")

    candidates: list[list[int]] = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            if record["secret_split"] == split:
                candidates.append(record["token_ids"])

    if len(candidates) < count:
        raise ValueError(
            f"requested {count} {split} trajectories, found {len(candidates)}"
        )
    return random.Random(seed).sample(candidates, count)


def create_shifted_pairs(
    trajectories: Sequence[Sequence[int]],
    *,
    context_length: int = CONTEXT_LENGTH,
) -> tuple[Tensor, Tensor]:
    """Pad trajectories and create next-token inputs and masked targets."""
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if any(len(trajectory) < 2 for trajectory in trajectories):
        raise ValueError("each trajectory must contain at least two tokens")
    if any(len(trajectory) - 1 > context_length for trajectory in trajectories):
        raise ValueError("trajectory exceeds the model context length")
    if any(
        token < 0 or token >= VOCAB_SIZE
        for trajectory in trajectories
        for token in trajectory
    ):
        raise ValueError("trajectory contains a token outside the vocabulary")

    sequence_length = max(len(trajectory) - 1 for trajectory in trajectories)
    inputs = torch.full(
        (len(trajectories), sequence_length),
        TOKEN_TO_ID[END_TOKEN],
        dtype=torch.long,
    )
    targets = torch.full(
        (len(trajectories), sequence_length),
        IGNORE_INDEX,
        dtype=torch.long,
    )
    for row, trajectory in enumerate(trajectories):
        prediction_count = len(trajectory) - 1
        inputs[row, :prediction_count] = torch.tensor(trajectory[:-1])
        targets[row, :prediction_count] = torch.tensor(trajectory[1:])
    return inputs, targets


def calculate_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Compute mean next-token cross-entropy, ignoring padded targets."""
    if logits.shape[:-1] != targets.shape:
        raise ValueError("logits and targets have incompatible shapes")
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=IGNORE_INDEX,
    )


def train(
    trajectories: Sequence[Sequence[int]],
    *,
    num_steps: int = DEFAULT_STEPS,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    device: str | torch.device | None = None,
    seed: int = 0,
    log_steps: Sequence[int] = (0, 100, 500),
) -> tuple[WordleGPT, dict[int, float]]:
    """Repeatedly train on a fixed trajectory batch and return monitored losses."""
    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    torch.manual_seed(seed)
    selected_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = WordleGPT().to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    inputs, targets = create_shifted_pairs(trajectories)
    inputs = inputs.to(selected_device)
    targets = targets.to(selected_device)
    monitored = set(log_steps)
    monitored.add(num_steps - 1)
    losses: dict[int, float] = {}

    model.train()
    for step in range(num_steps):
        logits = model(inputs)
        loss = calculate_loss(logits, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step in monitored:
            value = loss.detach().item()
            losses[step] = value
            print(f"step {step:<6} loss {value:.6f}")

    return model, losses


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overfit WordleGPT on a fixed set of tokenized trajectories."
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trajectories = load_trajectories(
        args.data,
        count=args.batch_size,
        seed=args.seed,
    )
    train(
        trajectories,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
