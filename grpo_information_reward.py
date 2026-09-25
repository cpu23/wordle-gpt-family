"""Exact one-guess rewards over the entire observable candidate set."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

from wordle import _feedback_code, _validate_word


@dataclass(frozen=True)
class InformationScores:
    rewards: np.ndarray
    expected_candidates: np.ndarray
    information: np.ndarray
    solve_probability: np.ndarray


class ExpectedInformationReward:
    """Precompute true feedback; score partitions without consulting a source secret."""

    def __init__(self, words: Sequence[str]):
        self.words = tuple(_validate_word(word, 'word') for word in words)
        if not self.words or len(set(self.words)) != len(self.words):
            raise ValueError('reward dictionary must be nonempty and unique')
        self.indices = {word: index for index, word in enumerate(self.words)}
        self.feedback = np.fromiter(
            (_feedback_code(answer, guess) for guess in self.words for answer in self.words),
            dtype=np.uint8, count=len(self.words) ** 2,
        ).reshape(len(self.words), len(self.words))

    def score(self, candidates: Sequence[str], guesses: Sequence[str]) -> InformationScores:
        """E[remaining] = sum(bucket_size**2)/N; reward = log(N/E) + 2*P(solve)."""
        if not candidates or len(set(candidates)) != len(candidates):
            raise ValueError('candidate set must be nonempty and unique')
        if not guesses:
            raise ValueError('at least one proposed guess is required')
        candidate_ids = np.fromiter((self.indices[word] for word in candidates), dtype=np.int64)
        guess_ids = np.fromiter((self.indices[word] for word in guesses), dtype=np.int64)
        codes = self.feedback[np.ix_(guess_ids, candidate_ids)]
        # Separate every guess's 243 feedback buckets in one bincount operation.
        offsets = np.arange(len(guesses), dtype=np.int64)[:, None] * 243
        buckets = np.bincount((codes + offsets).ravel(), minlength=len(guesses) * 243)
        buckets = buckets.reshape(len(guesses), 243)
        expected = np.square(buckets).sum(axis=1) / len(candidates)
        information = np.log(len(candidates) / expected)
        solve = np.isin(guess_ids, candidate_ids).astype(np.float64) / len(candidates)
        return InformationScores(information + 2.0 * solve, expected, information, solve)
