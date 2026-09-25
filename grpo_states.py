from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence

from dataset_expert import PARTLY_RANDOM_RATES, choose_entropy_guess, choose_poor_legal_guess
from tokenizer import END_TOKEN, GUESS_TOKEN, serialize_trajectory
from tokenizer_v2 import POLICY_TOKEN, encode
from wordle import GREEN, _validate_word, choose_informative_guess, score_guess

MAX_HISTORY_DEPTH = 5
GRPO_GROUP_SIZE = 8


@lru_cache(maxsize=200_000)
def _cached_feedback(answer: str, guess: str) -> str:
    """Reuse exact Wordle feedback across trajectories and reward calculations."""
    return score_guess(answer, guess)


def _filter_with_feedback(
    candidates: Sequence[str], guess: str, feedback: str
) -> tuple[str, ...]:
    return tuple(
        answer
        for answer in candidates
        if _cached_feedback(answer, guess) == feedback
    )


@dataclass(frozen=True)
class ReachableState:
    secret: str
    prompt: tuple[int, ...]
    candidates: tuple[str, ...]
    history: tuple[tuple[str, str], ...]


class ReachableStateSampler:
    """Sample fresh, reachable Wordle states from only the supplied secrets.

    Each sample chooses a requested history depth uniformly from 0 through 5
    (or through the largest depth that still leaves eight unused dictionary
    guesses for a smaller test dictionary), a secret uniformly from
    ``secrets``, and one source behavior uniformly from the eleven behaviors
    used here: random, candidate/simple, informative, entropy, poor, and the
    six dataset-expert partly-random rates.  The behavior is held fixed for a
    trajectory; partly-random trajectories make their random-vs-informative
    choice independently on each turn.  Candidate answers always start as
    the full ``words`` universe, while every action is an unused legal word
    from that same universe.

    A trajectory stops immediately when its actual secret is guessed.  If
    that happens before the requested depth, the returned state is the last
    unsolved state before the solving action; solved histories are never
    emitted.  Histories therefore contain at most five distinct guesses, and
    at least eight unused legal dictionary guesses remain available for a
    GRPO group (rather than restricting actions to current answer candidates).
    """

    def __init__(
        self, words: Sequence[str], secrets: Sequence[str], seed: int
    ) -> None:
        normalized_words = tuple(
            dict.fromkeys(_validate_word(word, "allowed word") for word in words)
        )
        if len(normalized_words) < GRPO_GROUP_SIZE:
            raise ValueError(
                f"at least {GRPO_GROUP_SIZE} distinct allowed words are required"
            )
        if not secrets:
            raise ValueError("at least one training secret is required")

        allowed = set(normalized_words)
        normalized_secrets = tuple(
            dict.fromkeys(_validate_word(secret, "training secret") for secret in secrets)
        )
        missing = tuple(secret for secret in normalized_secrets if secret not in allowed)
        if missing:
            raise ValueError("every training secret must be in the allowed word list")

        self.words = normalized_words
        self.secrets = normalized_secrets
        self.rng = random.Random(seed)
        self.max_history_depth = min(
            MAX_HISTORY_DEPTH, len(self.words) - GRPO_GROUP_SIZE
        )
        self._informative_cache: dict[tuple[str, ...], str] = {}
        self._entropy_cache: dict[tuple[str, ...], str] = {}
        self._behaviors = (
            "random",
            "candidate",
            "informative",
            "entropy",
            "poor",
            *(f"partly-random-{rate:.2f}" for rate in PARTLY_RANDOM_RATES),
        )

    def _choose_guess(
        self,
        behavior: str,
        candidates: tuple[str, ...],
        used: set[str],
    ) -> str:
        available = tuple(word for word in self.words if word not in used)
        if not available:
            raise RuntimeError("reachable state exhausted legal dictionary guesses")

        if behavior == "random":
            return self.rng.choice(available)
        if behavior == "candidate":
            candidate_guesses = tuple(word for word in candidates if word not in used)
            if not candidate_guesses:
                raise RuntimeError("reachable state has no untried candidate guess")
            return self.rng.choice(candidate_guesses)
        if behavior == "informative":
            guess = self._informative_cache.get(candidates)
            if guess is None or guess in used:
                guess = choose_informative_guess(candidates, available)
                if not used:
                    self._informative_cache[candidates] = guess
            return guess
        if behavior == "entropy":
            guess = self._entropy_cache.get(candidates)
            if guess is None or guess in used:
                guess = choose_entropy_guess(candidates, available)
                if not used:
                    self._entropy_cache[candidates] = guess
            return guess
        if behavior == "poor":
            return choose_poor_legal_guess(
                candidates, self.words, used, self.rng
            )
        if behavior.startswith("partly-random-"):
            rate = float(behavior.removeprefix("partly-random-"))
            if self.rng.random() < rate:
                return self.rng.choice(available)
            guess = self._informative_cache.get(candidates)
            if guess is None or guess in used:
                guess = choose_informative_guess(candidates, available)
                if not used:
                    self._informative_cache[candidates] = guess
            return guess
        raise RuntimeError(f"unknown source behavior: {behavior!r}")

    def sample(self) -> ReachableState:
        """Return one seeded-random pre-action state from a fresh trajectory."""
        secret = self.rng.choice(self.secrets)
        requested_depth = self.rng.randint(0, self.max_history_depth)
        behavior = self.rng.choice(self._behaviors)

        candidates = self.words
        history: list[tuple[str, str]] = []
        used: set[str] = set()
        for _ in range(requested_depth):
            guess = self._choose_guess(behavior, candidates, used)
            used.add(guess)
            feedback = _cached_feedback(secret, guess)
            if feedback == GREEN * 5:
                break
            history.append((guess, feedback))
            candidates = _filter_with_feedback(candidates, guess, feedback)
            if not candidates or secret not in candidates:
                raise RuntimeError("sampled feedback did not preserve its secret")

        history_tuple = tuple(history)
        history_text = serialize_trajectory(
            tuple({"guess": guess, "feedback": feedback} for guess, feedback in history_tuple)
        )
        prompt_text = (
            POLICY_TOKEN
            + history_text[: -len(END_TOKEN)]
            + GUESS_TOKEN
        )
        return ReachableState(
            secret=secret,
            prompt=tuple(encode(prompt_text)),
            candidates=tuple(candidates),
            history=history_tuple,
        )


def action_reward(
    state: ReachableState, guess: str, solve_bonus: float = 5.0
) -> tuple[float, int]:
    """Score a legal guess by actual candidate reduction and actual solving.

    Feedback is computed against ``state.secret`` and then used to filter the
    state's real candidate set.  The reward is ``log(before / after)`` plus
    ``solve_bonus`` only when that actual feedback is all green; an imagined
    outcome for another candidate is never substituted.
    """
    before = len(state.candidates)
    if before == 0:
        raise ValueError("cannot reward an action from an empty candidate set")
    if state.secret not in state.candidates:
        raise ValueError("state candidates do not contain the hidden secret")

    feedback = _cached_feedback(state.secret, guess)
    remaining = _filter_with_feedback(state.candidates, guess, feedback)
    if not remaining:
        raise ValueError("guess feedback eliminated every candidate")
    if state.secret not in remaining:
        raise ValueError("guess feedback eliminated the hidden secret")

    reward = math.log(before / len(remaining))
    if feedback == GREEN * 5:
        reward += solve_bonus
    return reward, len(remaining)
