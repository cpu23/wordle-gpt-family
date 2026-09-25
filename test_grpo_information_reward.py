import math
import unittest
from collections import Counter

import numpy as np

from grpo_information_reward import ExpectedInformationReward
from wordle import score_guess


class InformationRewardTests(unittest.TestCase):
    WORDS = ('cigar', 'rebut', 'sissy', 'humph', 'awake', 'blush', 'focal', 'evade', 'aback', 'abase')

    @classmethod
    def setUpClass(cls):
        cls.scorer = ExpectedInformationReward(cls.WORDS)

    def test_exact_all_secret_partition_expectation_including_duplicate_letters(self):
        candidates = self.WORDS[2:]
        result = self.scorer.score(candidates, self.WORDS)
        for index, guess in enumerate(self.WORDS):
            buckets = Counter(score_guess(answer, guess) for answer in candidates)
            remaining_for_each_secret = [buckets[score_guess(answer, guess)] for answer in candidates]
            expected = sum(remaining_for_each_secret) / len(candidates)
            bonus = 2 / len(candidates) if guess in candidates else 0
            self.assertAlmostEqual(result.expected_candidates[index], expected)
            self.assertAlmostEqual(result.rewards[index], math.log(len(candidates) / expected) + bonus)

    def test_log_of_expected_remaining_not_expected_log_reduction(self):
        result = self.scorer.score(self.WORDS, ('sissy',))
        buckets = Counter(score_guess(answer, 'sissy') for answer in self.WORDS)
        expected_log = sum(math.log(len(self.WORDS) / buckets[score_guess(answer, 'sissy')])
                           for answer in self.WORDS) / len(self.WORDS)
        self.assertNotAlmostEqual(result.information[0], expected_log)

    def test_singleton_bonus_and_candidate_order_invariance(self):
        scores = self.scorer.score(('awake',), ('awake', 'cigar'))
        np.testing.assert_array_equal(scores.expected_candidates, [1, 1])
        np.testing.assert_array_equal(scores.solve_probability, [1, 0])
        np.testing.assert_array_equal(scores.rewards, [2, 0])
        before = self.scorer.score(self.WORDS[:5], self.WORDS)
        after = self.scorer.score(tuple(reversed(self.WORDS[:5])), self.WORDS)
        np.testing.assert_array_equal(before.rewards, after.rewards)


if __name__ == '__main__':
    unittest.main()
