import unittest

import torch

from grpo_rollouts import LegalWordDecoder, sample_games
from grpo_trajectory_loss import token_logps, trajectory_logits, trajectory_logps
from model import WordleGPT
from tokenizer import FEEDBACK_TO_SYMBOL, FEEDBACK_TOKEN, GUESS_TOKEN
from tokenizer_v2 import POLICY_TOKEN, TOKEN_TO_ID, VOCABULARY_SIZE, decode
from wordle import score_guess


WORDS = ("cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade")



class TargetSequenceModel(torch.nn.Module):
    """Deterministically emits a turn-indexed legal word without reading secrets."""

    def __init__(self, targets):
        super().__init__()
        self.targets = tuple(targets)
        self.guess_token = TOKEN_TO_ID[GUESS_TOKEN]
        self.feedback_token = TOKEN_TO_ID[FEEDBACK_TOKEN]

    def forward(self, tokens):
        rows = tokens.tolist()
        logits = torch.full(
            (*tokens.shape, VOCABULARY_SIZE), -1000.0, device=tokens.device
        )
        for row_index, token_row in enumerate(rows):
            for position in range(len(token_row)):
                prefix = token_row[: position + 1]
                guess_positions = [
                    index for index, token in enumerate(prefix) if token == self.guess_token
                ]
                if not guess_positions:
                    continue
                guess_start = guess_positions[-1]
                turn = min(prefix[:guess_start].count(self.feedback_token), len(self.targets) - 1)
                letter_position = position - guess_start
                if letter_position < 0 or letter_position >= 5:
                    continue
                letter_id = ord(self.targets[turn][letter_position]) - ord("a")
                logits[row_index, position, letter_id] = 1000.0
        return logits


class RolloutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoder = LegalWordDecoder(WORDS, "cpu")

    def test_groups_share_secret_and_start_and_prompts_rebuild_actual_history(self):
        targets = ("rebut", "sissy", "humph", "awake", "blush", "focal")
        secrets = ("rebut", "focal")
        batch = sample_games(TargetSequenceModel(targets), secrets, self.decoder)

        self.assertEqual(batch.secrets, secrets)
        self.assertEqual(batch.inputs.shape[0], len(secrets) * 8)
        self.assertEqual(batch.guesses[:8], (tuple(targets[:1]),) * 8)
        self.assertEqual(batch.guesses[8:], (tuple(targets),) * 8)
        for group_index, secret in enumerate(secrets):
            for member in range(8):
                row = group_index * 8 + member
                turns = len(batch.guesses[row])
                input_length = 2 + 12 * (turns - 1) + 5
                observed = decode(batch.inputs[row, :input_length].tolist())
                expected = POLICY_TOKEN + GUESS_TOKEN
                for guess_index, guess in enumerate(batch.guesses[row]):
                    expected += guess
                    if guess_index + 1 < turns:
                        feedback = score_guess(secret, guess)
                        symbols = "".join(FEEDBACK_TO_SYMBOL[mark] for mark in feedback)
                        expected += FEEDBACK_TOKEN + symbols + GUESS_TOKEN
                self.assertEqual(observed, expected)
                self.assertTrue(observed.startswith(POLICY_TOKEN + GUESS_TOKEN))
                self.assertNotIn("<S>", observed)

        # Guess positions point at the contexts that predict the recorded letters.
        for row in range(batch.inputs.shape[0]):
            for action_index in torch.where(batch.valid[row])[0].tolist():
                position = batch.positions[row, action_index]
                self.assertEqual(
                    batch.inputs[row, position + 1].item(),
                    batch.actions[row, action_index].item(),
                )
                self.assertTrue(batch.masks[row, action_index, batch.actions[row, action_index]])
        self.assertTrue(batch.masks[~batch.valid].all())

    def test_exact_reward_ladder_and_failure_termination(self):
        targets = WORDS[:6]
        secrets = (*targets, "evade")
        batch = sample_games(TargetSequenceModel(targets), secrets, self.decoder)

        for group_index, secret in enumerate(secrets):
            expected_turns = group_index + 1 if group_index < 6 else 6
            expected_win = group_index < 6
            expected_reward = 7 - expected_turns if expected_win else 0
            self.assertEqual(batch.guesses[group_index * 8], targets[:expected_turns])
            self.assertTrue((batch.attempts[group_index] == expected_turns).all())
            self.assertTrue((batch.won[group_index] == expected_win).all())
            self.assertTrue((batch.rewards[group_index] == expected_reward).all())
            if expected_win:
                self.assertEqual(batch.guesses[group_index * 8][-1], secret)
            else:
                self.assertEqual(len(batch.guesses[group_index * 8]), 6)
                self.assertNotIn(secret, batch.guesses[group_index * 8])
        self.assertEqual(batch.rewards[:, 0].tolist(), [6, 5, 4, 3, 2, 1, 0])

    def test_repeated_guesses_are_legal_and_not_removed_from_later_sampling(self):
        batch = sample_games(
            TargetSequenceModel(("rebut",) * 6), ("cigar",), self.decoder
        )
        self.assertEqual(batch.guesses, (("rebut",) * 6,) * 8)
        self.assertTrue((batch.attempts == 6).all())
        self.assertFalse(batch.won.any())
        self.assertFalse(batch.rewards.any())


    def test_one_remaining_guess_uses_total_attempts_for_reward(self):
        secret = "cigar"
        history_guesses = WORDS[1:6]
        history = tuple(
            (guess, score_guess(secret, guess)) for guess in history_guesses
        )

        solved = sample_games(
            TargetSequenceModel((secret,)), (secret,), self.decoder, histories=(history,)
        )
        self.assertEqual(solved.guesses, ((*history_guesses, secret),) * 8)
        self.assertTrue((solved.attempts == 6).all())
        self.assertTrue(solved.won.all())
        self.assertTrue((solved.rewards == 1).all())
        self.assertEqual(int(solved.valid.sum()), 8 * 5)
        self.assertFalse(solved.valid[:, :25].any())

        failed = sample_games(
            TargetSequenceModel(("focal",)), (secret,), self.decoder, histories=(history,)
        )
        self.assertEqual(failed.guesses, ((*history_guesses, "focal"),) * 8)
        self.assertTrue((failed.attempts == 6).all())
        self.assertFalse(failed.won.any())
        self.assertFalse(failed.rewards.any())
        self.assertEqual(int(failed.valid.sum()), 8 * 5)

    def test_mixed_history_depths_count_terminal_attempts_absolutely(self):
        secret = "cigar"
        shallow_guesses = WORDS[1:5]
        deep_guesses = WORDS[1:6]
        histories = tuple(
            tuple((guess, score_guess(secret, guess)) for guess in guesses)
            for guesses in (shallow_guesses, deep_guesses)
        )
        targets = ("rebut", "sissy", "humph", "awake", "blush", secret)
        batch = sample_games(
            TargetSequenceModel(targets), (secret, secret), self.decoder, histories=histories
        )

        self.assertEqual(batch.guesses[:8], ((*shallow_guesses, "blush", secret),) * 8)
        self.assertEqual(batch.guesses[8:], ((*deep_guesses, secret),) * 8)
        self.assertTrue((batch.attempts == 6).all())
        self.assertTrue(batch.won.all())
        self.assertTrue((batch.rewards == 1).all())
        self.assertEqual(int(batch.valid[:8].sum()), 8 * 10)
        self.assertEqual(int(batch.valid[8:].sum()), 8 * 5)

        for row in (0, 8):
            guesses = batch.guesses[row]
            input_length = 2 + 12 * (len(guesses) - 1) + 5
            prompt = decode(batch.inputs[row, :input_length].tolist())
            expected = POLICY_TOKEN + GUESS_TOKEN
            for index, guess in enumerate(guesses):
                expected += guess
                if index + 1 < len(guesses):
                    feedback = score_guess(secret, guess)
                    symbols = "".join(FEEDBACK_TO_SYMBOL[mark] for mark in feedback)
                    expected += FEEDBACK_TOKEN + symbols + GUESS_TOKEN
            self.assertEqual(prompt, expected)
            self.assertNotIn("<S>", prompt)

    def test_history_likelihood_covers_only_generated_continuation(self):
        torch.manual_seed(23)
        model = WordleGPT(
            vocab_size=VOCABULARY_SIZE, embedding_size=16,
            num_layers=1, num_heads=2, mlp_size=32,
        ).eval()
        secret = "cigar"
        history_guesses = WORDS[1:3]
        history = tuple(
            (guess, score_guess(secret, guess)) for guess in history_guesses
        )
        batch = sample_games(
            model, (secret,), self.decoder, histories=(history,)
        )

        logits = trajectory_logits(model, batch)
        teacher_forced = token_logps(logits, batch)
        torch.testing.assert_close(
            batch.old_token_logps, teacher_forced, atol=1e-5, rtol=1e-5
        )
        self.assertFalse(batch.old_token_logps[~batch.valid].any())
        self.assertFalse(batch.old_token_logps.requires_grad)
        self.assertFalse(batch.valid[:, : len(history) * 5].any())
        for row, guesses in enumerate(batch.guesses):
            generated_count = (len(guesses) - len(history)) * 5
            self.assertEqual(int(batch.valid[row].sum()), generated_count)

    def test_constrained_old_likelihood_matches_teacher_forced_prefixes(self):
        torch.manual_seed(19)
        model = WordleGPT(
            vocab_size=VOCABULARY_SIZE, embedding_size=16,
            num_layers=1, num_heads=2, mlp_size=32,
        ).eval()
        batch = sample_games(model, ("cigar", "evade"), self.decoder)
        logits = trajectory_logits(model, batch)
        teacher_forced = token_logps(logits, batch)
        torch.testing.assert_close(
            batch.old_token_logps, teacher_forced, atol=1e-5, rtol=1e-5
        )
        torch.testing.assert_close(
            batch.old_token_logps.sum(-1).view_as(batch.rewards),
            trajectory_logps(logits, batch),
            atol=1e-5,
            rtol=1e-5,
        )
        self.assertFalse(batch.old_token_logps[~batch.valid].any())
        self.assertFalse(batch.old_token_logps.requires_grad)

        legal_letter_ids = batch.masks[..., :26].sum(-1)
        self.assertTrue((legal_letter_ids[batch.valid] >= 1).all())
        self.assertEqual(batch.positions.shape, (16, 30))
        self.assertEqual(batch.actions.shape, (16, 30))
        self.assertEqual(batch.masks.shape, (16, 30, VOCABULARY_SIZE))
        self.assertEqual(batch.valid.shape, (16, 30))
        self.assertEqual(batch.old_token_logps.shape, (16, 30))
        self.assertLessEqual(batch.inputs.shape[1], 67)


if __name__ == "__main__":
    unittest.main()
