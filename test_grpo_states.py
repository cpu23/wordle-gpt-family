import math
import unittest

from grpo_states import ReachableState, ReachableStateSampler, action_reward
from tokenizer import END_TOKEN, GUESS_TOKEN, serialize_trajectory
from tokenizer_v2 import POLICY_TOKEN, encode
from wordle import GREEN, filter_answers, load_words, score_guess


class ReachableStateSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = load_words()

    def test_sampling_is_seeded_and_emits_only_reachable_unsolved_states(self):
        secrets = self.words[:40]
        first = ReachableStateSampler(self.words, secrets, seed=29)
        second = ReachableStateSampler(self.words, secrets, seed=29)
        states = [first.sample() for _ in range(64)]
        self.assertEqual(states, [second.sample() for _ in range(64)])

        for state in states:
            self.assertIn(state.secret, secrets)
            self.assertLessEqual(len(state.history), 5)
            self.assertEqual(len({guess for guess, _ in state.history}), len(state.history))
            possible = self.words
            for guess, feedback in state.history:
                self.assertIn(guess, self.words)
                self.assertNotEqual(feedback, GREEN * 5)
                self.assertEqual(feedback, score_guess(state.secret, guess))
                possible = filter_answers(possible, guess, feedback)
                self.assertIn(state.secret, possible)
            self.assertEqual(state.candidates, possible)
            self.assertGreaterEqual(
                sum(word not in {guess for guess, _ in state.history} for word in self.words),
                8,
            )
            history_text = serialize_trajectory(
                tuple(
                    {"guess": guess, "feedback": feedback}
                    for guess, feedback in state.history
                )
            )
            prompt = POLICY_TOKEN + history_text[: -len(END_TOKEN)] + GUESS_TOKEN
            self.assertEqual(state.prompt, tuple(encode(prompt)))


class ActionRewardTests(unittest.TestCase):
    def setUp(self):
        self.state = ReachableState(
            secret="cigar",
            prompt=(),
            candidates=("cigar", "rebut", "sissy"),
            history=(),
        )

    def test_reward_uses_actual_feedback_not_another_candidate_outcome(self):
        guess = "rebut"
        actual_feedback = score_guess(self.state.secret, guess)
        remaining = filter_answers(self.state.candidates, guess, actual_feedback)
        reward, count = action_reward(self.state, guess)

        self.assertNotEqual(actual_feedback, GREEN * 5)
        self.assertEqual(count, len(remaining))
        self.assertEqual(reward, math.log(len(self.state.candidates) / len(remaining)))

    def test_solve_bonus_requires_actual_secret_to_be_solved(self):
        reward, count = action_reward(self.state, self.state.secret)
        remaining = filter_answers(
            self.state.candidates,
            self.state.secret,
            score_guess(self.state.secret, self.state.secret),
        )

        self.assertEqual(count, len(remaining))
        self.assertEqual(count, 1)
        self.assertAlmostEqual(
            reward,
            math.log(len(self.state.candidates) / count) + 5.0,
        )


if __name__ == "__main__":
    unittest.main()
