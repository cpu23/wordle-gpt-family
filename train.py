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
from tokenizer import END_TOKEN, ID_TO_TOKEN, TOKEN_TO_ID, decode

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
    vocab_size: int = VOCAB_SIZE,
) -> tuple[Tensor, Tensor]:
    """Pad trajectories and create next-token inputs and masked targets."""
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if any(len(trajectory) < 2 for trajectory in trajectories):
        raise ValueError("each trajectory must contain at least two tokens")
    if any(len(trajectory) - 1 > context_length for trajectory in trajectories):
        raise ValueError("trajectory exceeds the model context length")
    if any(
        token < 0 or token >= vocab_size
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


def generate_tokens(
    model: WordleGPT,
    prefix: Sequence[int],
    *,
    max_new_tokens: int,
    stop_token_id: int | None = None,
) -> list[int]:
    """Greedily append one model prediction at a time."""
    if not prefix:
        raise ValueError("prefix must contain at least one token")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative")
    if len(prefix) + max_new_tokens > model.context_length:
        raise ValueError("generation would exceed the model context length")

    device = next(model.parameters()).device
    generated = torch.empty(
        (1, len(prefix) + max_new_tokens),
        dtype=torch.long,
        device=device,
    )
    generated[0, : len(prefix)] = torch.tensor(prefix, device=device)
    generated_length = len(prefix)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                logits = model(generated[:, :generated_length])
                next_token = logits[0, -1].argmax()
                generated[0, generated_length] = next_token
                generated_length += 1
                if stop_token_id is not None and next_token.item() == stop_token_id:
                    break
    finally:
        model.train(was_training)
    return generated[0, :generated_length].tolist()


def generate_constrained_guess(
    model: WordleGPT,
    prefix: Sequence[int],
    allowed_words: frozenset[str],
    *,
    length: int = 5,
) -> list[int]:
    """Greedily decode ``length`` letters restricted to allowed-word prefixes.

    At every position the model scores the full vocabulary, then every token
    that is not a letter occurring at that position in some still-allowed
    word is masked to -inf before the argmax. The finished word is therefore
    guaranteed to be a member of ``allowed_words``; no other model
    probability is altered.
    """
    if not prefix:
        raise ValueError("prefix must contain at least one token")
    if length < 1:
        raise ValueError("length must be positive")
    if len(prefix) + length > model.context_length:
        raise ValueError("generation would exceed the model context length")
    remaining = list(allowed_words)
    if not remaining:
        raise ValueError("at least one allowed word is required")
    if any(len(word) != length for word in remaining):
        raise ValueError("allowed words must have exactly `length` letters")

    device = next(model.parameters()).device
    generated = torch.empty(
        (1, len(prefix) + length),
        dtype=torch.long,
        device=device,
    )
    generated[0, : len(prefix)] = torch.tensor(prefix, device=device)
    generated_length = len(prefix)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for position in range(length):
                logits = model(generated[:, :generated_length])[0, -1]
                mask = torch.full_like(logits, float("-inf"))
                allowed_letters = {word[position] for word in remaining}
                mask[[TOKEN_TO_ID[letter] for letter in allowed_letters]] = 0.0
                next_token = (logits + mask).argmax()
                generated[0, generated_length] = next_token
                generated_length += 1
                letter = ID_TO_TOKEN[int(next_token)]
                remaining = [word for word in remaining if word[position] == letter]
                if not remaining:
                    raise RuntimeError(
                        "constrained decoding eliminated every allowed word"
                    )
    finally:
        model.train(was_training)
    return generated[0, len(prefix) :].tolist()


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
    parser.add_argument(
        "--generate-example",
        action="store_true",
        help="greedily continue one memorized training trajectory",
    )
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--prefix-length", type=int, default=6)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    trajectories = load_trajectories(
        args.data,
        count=args.batch_size,
        seed=args.seed,
    )
    model, _ = train(
        trajectories,
        num_steps=args.steps,
        learning_rate=args.learning_rate,
        device=args.device,
        seed=args.seed,
    )
    if args.generate_example:
        if not 0 <= args.example_index < len(trajectories):
            raise ValueError("example-index is outside the selected training batch")
        target = trajectories[args.example_index]
        if not 1 <= args.prefix_length < len(target):
            raise ValueError("prefix-length must be within the selected trajectory")
        prefix = target[: args.prefix_length]
        predicted = generate_tokens(
            model,
            prefix,
            max_new_tokens=len(target) - len(prefix),
            stop_token_id=TOKEN_TO_ID[END_TOKEN],
        )
        predicted_continuation = predicted[len(prefix) :]
        expected_continuation = target[len(prefix) :]
        correct = sum(
            actual == expected
            for actual, expected in zip(
                predicted_continuation, expected_continuation, strict=False
            )
        )
        accuracy = correct / len(expected_continuation)
        print(f"prompt:    {decode(prefix)}")
        print(f"expected:  {decode(target)}")
        print(f"predicted: {decode(predicted)}")
        print(
            f"continuation tokens: {correct}/{len(expected_continuation)} "
            f"({accuracy:.1%})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
