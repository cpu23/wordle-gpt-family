from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from model import WordleGPT
from tokenizer import FEEDBACK_TO_SYMBOL, FEEDBACK_TOKEN, GUESS_TOKEN
from tokenizer_v2 import POLICY_TOKEN, VOCABULARY_SIZE, decode, encode
from train import generate_tokens
from wordle import DEFAULT_WORDS, load_words, score_guess

DEFAULT_SPLITS_PATH = Path("data/wordle-100k/secret-splits.json")
DEFAULT_MAX_GUESSES = 6
LETTER_TOKEN_LIMIT = 26


@dataclass(frozen=True)
class GameResult:
    secret: str
    guesses: tuple[str, ...]
    won: bool
    invalid_guesses: int


@dataclass(frozen=True)
class GameplaySummary:
    checkpoint: str
    games: int
    wins: int
    win_rate: float
    average_attempts: float
    average_guesses: float
    invalid_guesses: int
    results: tuple[GameResult, ...]


def _feedback_symbols(feedback: str) -> str:
    return "".join(FEEDBACK_TO_SYMBOL[mark] for mark in feedback)


def load_v2_model(checkpoint_path: str | Path, device: str) -> WordleGPT:
    """Restore one v2 checkpoint for deterministic greedy evaluation."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if checkpoint.get("vocabulary_size") != VOCABULARY_SIZE:
        raise ValueError("checkpoint does not use the current v2 vocabulary")
    model = WordleGPT(vocab_size=VOCABULARY_SIZE).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def play_secret(
    model: WordleGPT,
    secret: str,
    allowed_words: frozenset[str],
    *,
    max_guesses: int = DEFAULT_MAX_GUESSES,
) -> GameResult:
    """Play one game greedily; an invalid generated word ends the game."""
    prefix = encode(POLICY_TOKEN + GUESS_TOKEN)
    guesses: list[str] = []
    for _ in range(max_guesses):
        generated = generate_tokens(model, prefix, max_new_tokens=5)
        guess_ids = generated[-5:]
        if any(not 0 <= token_id < LETTER_TOKEN_LIMIT for token_id in guess_ids):
            guesses.append(decode(guess_ids))
            return GameResult(secret, tuple(guesses), False, 1)
        guess = decode(guess_ids)
        guesses.append(guess)
        if guess not in allowed_words:
            return GameResult(secret, tuple(guesses), False, 1)
        if guess == secret:
            return GameResult(secret, tuple(guesses), True, 0)
        feedback = _feedback_symbols(score_guess(secret, guess))
        prefix = generated + encode(FEEDBACK_TOKEN + feedback + GUESS_TOKEN)
    return GameResult(secret, tuple(guesses), False, 0)


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    secrets: Sequence[str],
    allowed_words: Sequence[str],
    *,
    device: str,
) -> GameplaySummary:
    """Play every supplied secret and aggregate game-level metrics."""
    model = load_v2_model(checkpoint_path, device)
    allowed = frozenset(allowed_words)
    results = tuple(play_secret(model, secret, allowed) for secret in secrets)
    wins = sum(result.won for result in results)
    return GameplaySummary(
        checkpoint=str(checkpoint_path),
        games=len(results),
        wins=wins,
        win_rate=wins / len(results),
        average_guesses=(
            sum(len(result.guesses) for result in results if result.won) / wins
            if wins
            else 0.0
        ),
        average_attempts=sum(len(result.guesses) for result in results) / len(results),
        invalid_guesses=sum(result.invalid_guesses for result in results),
        results=results,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Greedily play a v2 checkpoint on a held-out secret split."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="test")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=None)
    parser.add_argument("--details", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    summary = evaluate_checkpoint(
        args.checkpoint,
        splits[args.split],
        load_words(args.words),
        device=device,
    )
    payload = asdict(summary)
    if not args.details:
        payload.pop("results")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
