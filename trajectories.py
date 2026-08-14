from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import random
from pathlib import Path
from typing import IO, Iterator, Sequence

from wordle import (
    DEFAULT_WORDS,
    GREEN,
    choose_informative_guess,
    filter_answers,
    load_words,
    score_guess,
)

SCHEMA_VERSION = 1
STRATEGIES = ("clever", "simple", "random", "partly-random")


def _word_list_hash(words: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for word in words:
        digest.update(word.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _random_untried_guess(
    words: Sequence[str], used_guesses: set[str], rng: random.Random
) -> str:
    available = [word for word in words if word not in used_guesses]
    return rng.choice(available or list(words))


def generate_trajectory(
    answer: str,
    words: Sequence[str],
    strategy: str,
    rng: random.Random,
    *,
    max_turns: int = 6,
    random_rate: float = 0.35,
    clever_cache: dict[tuple[str, ...], str] | None = None,
) -> dict[str, object]:
    """Generate one valid game trajectory.

    Random trajectories sample any untried dictionary word. Partly-random
    trajectories use the clever policy except on random_rate of turns, where
    they sample any untried dictionary word. Simple trajectories sample only
    answers consistent with all feedback so far.
    """
    if answer not in words:
        raise ValueError("answer must be present in the word list")
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of: {', '.join(STRATEGIES)}")
    if max_turns < 1:
        raise ValueError("max_turns must be at least 1")
    if not 0.0 <= random_rate <= 1.0:
        raise ValueError("random_rate must be between 0 and 1")

    possible = tuple(words)
    used_guesses: set[str] = set()
    turns: list[dict[str, object]] = []
    cache = clever_cache if clever_cache is not None else {}

    for turn_number in range(1, max_turns + 1):
        possible_before = len(possible)
        if strategy == "simple":
            policy = "simple"
            guess = rng.choice(possible)
        elif strategy == "random":
            policy = "random"
            guess = _random_untried_guess(words, used_guesses, rng)
        else:
            use_random = strategy == "partly-random" and rng.random() < random_rate
            if use_random:
                policy = "random"
                guess = _random_untried_guess(words, used_guesses, rng)
            else:
                policy = "clever"
                guess = cache.get(possible)
                if guess is None:
                    guess = choose_informative_guess(possible, words)
                    cache[possible] = guess

        used_guesses.add(guess)
        feedback = score_guess(answer, guess)
        possible = filter_answers(possible, guess, feedback)
        turns.append(
            {
                "turn": turn_number,
                "policy": policy,
                "guess": guess,
                "feedback": feedback,
                "possible_before": possible_before,
                "possible_after": len(possible),
                "remaining_answers": list(possible),
            }
        )
        if feedback == GREEN * 5:
            break

    return {
        "schema_version": SCHEMA_VERSION,
        "strategy": strategy,
        "answer": answer,
        "solved": bool(turns and turns[-1]["feedback"] == GREEN * 5),
        "max_turns": max_turns,
        "turns": turns,
    }


def generate_trajectories(
    words: Sequence[str],
    count: int,
    strategies: Sequence[str] = STRATEGIES,
    *,
    seed: int = 0,
    max_turns: int = 6,
    random_rate: float = 0.35,
) -> Iterator[dict[str, object]]:
    """Yield reproducible trajectories, cycling evenly through strategies."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if not strategies:
        raise ValueError("at least one strategy is required")
    invalid = set(strategies) - set(STRATEGIES)
    if invalid:
        raise ValueError(f"unknown strategies: {', '.join(sorted(invalid))}")

    rng = random.Random(seed)
    clever_cache: dict[tuple[str, ...], str] = {}
    for trajectory_id in range(count):
        strategy = strategies[trajectory_id % len(strategies)]
        answer = rng.choice(words)
        trajectory = generate_trajectory(
            answer,
            words,
            strategy,
            rng,
            max_turns=max_turns,
            random_rate=random_rate,
            clever_cache=clever_cache,
        )
        trajectory["trajectory_id"] = trajectory_id
        yield trajectory


def _open_text(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", encoding="utf-8", newline="\n")
    return path.open("w", encoding="utf-8", newline="\n")


def save_trajectories(
    path: str | Path,
    trajectories: Iterator[dict[str, object]],
    words: Sequence[str],
    *,
    seed: int,
    strategies: Sequence[str],
    random_rate: float,
) -> tuple[int, int]:
    """Stream metadata and trajectories as JSON Lines; return games and states."""
    output_path = Path(path)
    metadata = {
        "record_type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "strategies": list(strategies),
        "random_rate": random_rate,
        "word_count": len(words),
        "word_list_sha256": _word_list_hash(words),
    }
    game_count = 0
    state_count = 0
    with _open_text(output_path) as output:
        output.write(json.dumps(metadata, separators=(",", ":")) + "\n")
        for trajectory in trajectories:
            record = {"record_type": "trajectory", **trajectory}
            output.write(json.dumps(record, separators=(",", ":")) + "\n")
            game_count += 1
            state_count += len(trajectory["turns"])
    return game_count, state_count


def validate_trajectory(trajectory: dict[str, object], words: Sequence[str]) -> None:
    """Raise ValueError if a saved trajectory is not a reachable Wordle game."""
    answer = trajectory.get("answer")
    if not isinstance(answer, str) or answer not in words:
        raise ValueError("trajectory answer is not in the word list")
    possible = tuple(words)
    solved = False
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        raise ValueError("trajectory turns must be a list")
    for expected_number, turn in enumerate(turns, start=1):
        if not isinstance(turn, dict):
            raise ValueError("trajectory turn must be an object")
        guess = turn.get("guess")
        feedback = turn.get("feedback")
        if guess not in words:
            raise ValueError("trajectory guess is not in the word list")
        expected_feedback = score_guess(answer, guess)
        if feedback != expected_feedback:
            raise ValueError("trajectory feedback does not match answer and guess")
        updated = filter_answers(possible, guess, expected_feedback)
        if turn.get("turn") != expected_number:
            raise ValueError("trajectory turn numbers are not consecutive")
        if turn.get("possible_before") != len(possible):
            raise ValueError("possible_before is incorrect")
        if turn.get("possible_after") != len(updated):
            raise ValueError("possible_after is incorrect")
        if turn.get("remaining_answers") != list(updated):
            raise ValueError("remaining_answers is incorrect")
        if solved:
            raise ValueError("trajectory contains turns after solving")
        solved = expected_feedback == GREEN * 5
        possible = updated
    if trajectory.get("solved") is not solved:
        raise ValueError("trajectory solved flag is incorrect")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate reproducible Wordle trajectories as JSON Lines."
    )
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--random-rate", type=float, default=0.35)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=list(STRATEGIES),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        words = load_words(args.words)
        trajectories = generate_trajectories(
            words,
            args.count,
            args.strategies,
            seed=args.seed,
            max_turns=args.max_turns,
            random_rate=args.random_rate,
        )
        games, states = save_trajectories(
            args.output,
            trajectories,
            words,
            seed=args.seed,
            strategies=args.strategies,
            random_rate=args.random_rate,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2
    print(f"Saved {games} trajectories containing {states} states to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
