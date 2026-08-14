from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

GRAY = "X"
YELLOW = "Y"
GREEN = "G"
VALID_FEEDBACK = frozenset((GRAY, YELLOW, GREEN))
DEFAULT_WORDS = Path(__file__).with_name("words.txt")


def _validate_word(word: str, label: str = "word") -> str:
    normalized = word.strip().lower()
    if len(normalized) != 5 or not normalized.isascii() or not normalized.isalpha():
        raise ValueError(f"{label} must contain exactly five ASCII letters: {word!r}")
    return normalized


def load_words(path: str | Path = DEFAULT_WORDS) -> tuple[str, ...]:
    """Load unique, lowercase five-letter words from a one-word-per-line file."""
    words: list[str] = []
    seen: set[str] = set()
    with Path(path).open(encoding="utf-8") as word_file:
        for line_number, line in enumerate(word_file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            word = _validate_word(stripped, f"word on line {line_number}")
            if word not in seen:
                seen.add(word)
                words.append(word)
    if not words:
        raise ValueError(f"word list is empty: {path}")
    return tuple(words)


def choose_answer(words: Sequence[str], rng: random.Random | None = None) -> str:
    if not words:
        raise ValueError("cannot choose an answer from an empty word list")
    return (rng or random).choice(words)


def score_guess(answer: str, guess: str) -> str:
    """Return G/Y/X feedback, consuming duplicate letters at most once."""
    answer = _validate_word(answer, "answer")
    guess = _validate_word(guess, "guess")
    feedback = [GRAY] * 5
    unmatched = [0] * 26

    for index, (answer_letter, guess_letter) in enumerate(zip(answer, guess)):
        if answer_letter == guess_letter:
            feedback[index] = GREEN
        else:
            unmatched[ord(answer_letter) - ord("a")] += 1

    for index, guess_letter in enumerate(guess):
        if feedback[index] == GREEN:
            continue
        letter_index = ord(guess_letter) - ord("a")
        if unmatched[letter_index]:
            feedback[index] = YELLOW
            unmatched[letter_index] -= 1

    return "".join(feedback)


def filter_answers(
    possible_answers: Iterable[str], guess: str, feedback: str
) -> tuple[str, ...]:
    """Keep exactly the answers that would have produced the observed feedback."""
    guess = _validate_word(guess, "guess")
    normalized_feedback = feedback.strip().upper()
    if len(normalized_feedback) != 5 or not set(normalized_feedback) <= VALID_FEEDBACK:
        raise ValueError("feedback must be five characters containing only G, Y, or X")
    return tuple(
        answer
        for answer in possible_answers
        if score_guess(answer, guess) == normalized_feedback
    )


def _feedback_code(answer: str, guess: str) -> int:
    """Allocation-light base-3 form of score_guess for solver bucketing."""
    marks = [0] * 5
    unmatched = [0] * 26
    for index in range(5):
        answer_index = ord(answer[index]) - ord("a")
        if answer[index] == guess[index]:
            marks[index] = 2
        else:
            unmatched[answer_index] += 1
    for index in range(5):
        if marks[index]:
            continue
        guess_index = ord(guess[index]) - ord("a")
        if unmatched[guess_index]:
            marks[index] = 1
            unmatched[guess_index] -= 1
    code = 0
    for mark in marks:
        code = code * 3 + mark
    return code


def choose_informative_guess(
    possible_answers: Sequence[str], allowed_guesses: Sequence[str]
) -> str:
    """Choose the guess with the smallest expected remaining answer set.

    For bucket sizes b_i and n possible answers, expected survivors are
    sum(b_i ** 2) / n. Minimizing the numerator maximizes expected elimination.
    """
    if not possible_answers:
        raise ValueError("cannot choose a guess with no possible answers")
    if not allowed_guesses:
        raise ValueError("cannot choose a guess with no allowed guesses")
    if len(possible_answers) == 1:
        return possible_answers[0]

    possible_set = set(possible_answers)
    candidates = list(possible_answers)
    candidates.extend(guess for guess in allowed_guesses if guess not in possible_set)
    best_guess = candidates[0]
    best_cost = len(possible_answers) ** 2 + 1
    ideal_cost = len(possible_answers)

    for guess in candidates:
        buckets: dict[int, int] = {}
        cost = 0
        for answer in possible_answers:
            code = _feedback_code(answer, guess)
            old_size = buckets.get(code, 0)
            buckets[code] = old_size + 1
            cost += 2 * old_size + 1
            if cost >= best_cost:
                break
        if cost < best_cost:
            best_cost = cost
            best_guess = guess
            if cost == ideal_cost:
                break

    return best_guess


@dataclass(frozen=True)
class Turn:
    guess: str
    feedback: str
    remaining: int


def solve(
    answer: str,
    words: Sequence[str],
    strategy: str = "clever",
    rng: random.Random | None = None,
) -> tuple[Turn, ...]:
    """Solve a known game while only using its returned feedback."""
    answer = _validate_word(answer, "answer")
    if answer not in words:
        raise ValueError("answer must be present in the word list")
    if strategy not in {"simple", "clever"}:
        raise ValueError("strategy must be 'simple' or 'clever'")

    random_source = rng or random.Random()
    possible = tuple(words)
    turns: list[Turn] = []
    while possible:
        if strategy == "simple":
            guess = random_source.choice(possible)
        else:
            guess = choose_informative_guess(possible, words)
        feedback = score_guess(answer, guess)
        possible = filter_answers(possible, guess, feedback)
        turns.append(Turn(guess, feedback, len(possible)))
        if feedback == GREEN * 5:
            return tuple(turns)
    raise RuntimeError("feedback eliminated the hidden answer")


def _feedback_names(feedback: str) -> str:
    names = {GREEN: "green", YELLOW: "yellow", GRAY: "gray"}
    return " ".join(names[mark] for mark in feedback)


def _play(words: Sequence[str], answer: str, attempts: int = 6) -> int:
    allowed = set(words)
    print("Feedback is shown as green, yellow, or gray for each letter.")
    attempt = 1
    while attempt <= attempts:
        try:
            guess = input(f"Guess {attempt}/{attempts}: ").strip().lower()
        except EOFError:
            print()
            return 1
        if guess not in allowed:
            print("Enter a word from the loaded five-letter word list.")
            continue
        feedback = score_guess(answer, guess)
        print(_feedback_names(feedback))
        if feedback == GREEN * 5:
            print(f"Solved in {attempt} guess{'es' if attempt != 1 else ''}.")
            return 0
        attempt += 1
    print(f"The answer was {answer}.")
    return 1


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play Wordle or watch a solver.")
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS, help="word-list path")
    subparsers = parser.add_subparsers(dest="command", required=True)

    play_parser = subparsers.add_parser("play", help="play interactively")
    play_parser.add_argument("--answer", help="fixed answer (otherwise chosen randomly)")
    play_parser.add_argument("--seed", type=int, help="random seed")

    solve_parser = subparsers.add_parser("solve", help="run an automatic solver")
    solve_parser.add_argument("--solver", choices=("simple", "clever"), default="clever")
    solve_parser.add_argument("--answer", help="fixed answer (otherwise chosen randomly)")
    solve_parser.add_argument("--seed", type=int, help="random seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        words = load_words(args.words)
        rng = random.Random(args.seed)
        answer = _validate_word(args.answer, "answer") if args.answer else choose_answer(words, rng)
        if answer not in words:
            raise ValueError("answer must be present in the word list")
    except (OSError, ValueError) as error:
        print(f"error: {error}")
        return 2

    if args.command == "play":
        return _play(words, answer)

    turns = solve(answer, words, args.solver, rng)
    for number, turn in enumerate(turns, start=1):
        print(
            f"{number}: {turn.guess} -> {_feedback_names(turn.feedback)} "
            f"({turn.remaining} possible)"
        )
    print(f"Solved {answer} in {len(turns)} guess{'es' if len(turns) != 1 else ''}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
