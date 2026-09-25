import itertools
import random
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from grpo_information_actions import (
    build_guess_batch,
    sample_information_actions,
    sample_policy_guesses,
)
from grpo_rollouts import LegalWordDecoder
from grpo_trajectory_loss import token_logps, trajectory_logits
from tokenizer import FEEDBACK_TO_SYMBOL
from tokenizer_v2 import VOCABULARY_SIZE, decode, encode
from wordle import score_guess


WORDS = tuple("aa" + "".join(letters) for letters in itertools.product("abcd", repeat=3))


def state_at_depth(depth):
    text = "<P><G>"
    feedback = score_guess("aaadd", "aaaaa")
    symbols = "".join(FEEDBACK_TO_SYMBOL[mark] for mark in feedback)
    text += ("aaaaa<F>" + symbols + "<G>") * depth
    # Legal guesses must not be restricted to the surviving secret candidates.
    return SimpleNamespace(prompt=tuple(encode(text)), candidates=("aaadd",))


class PrefixPolicy(torch.nn.Module):
    """Small causal policy whose distribution depends on all preceding tokens."""

    def __init__(self, concentrated=False):
        super().__init__()
        values = torch.arange(VOCABULARY_SIZE, dtype=torch.float32)
        self.scores = torch.nn.Parameter(-1000 * values if concentrated else values * 0.37)
        self.concentrated = concentrated
        self.batch_sizes = []

    def forward(self, inputs):
        self.batch_sizes.append(len(inputs))
        if self.concentrated:
            return self.scores.expand(*inputs.shape, -1)
        context = inputs.cumsum(dim=1).float() * 0.13
        return (context[:, :, None] + self.scores).sin()


class InformationActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decoder = LegalWordDecoder(WORDS)

    def test_concentrated_policy_refills_terminate_with_real_policy_origins(self):
        model = PrefixPolicy(concentrated=True)
        batch = sample_information_actions(
            model, (state_at_depth(0),), self.decoder, random.Random(31),
            forward_batch_size=32,
        )
        guesses = batch.guesses[0]
        self.assertEqual(len(guesses), 64)
        self.assertEqual(set(guesses), set(WORDS))
        self.assertEqual(batch.proposal_sources[0], ("policy",) * 48 + ("random",) * 16)
        self.assertEqual(batch.proposal_stats["initial_duplicates"], 47)
        self.assertEqual(batch.proposal_stats["refill_rounds"], 47)
        self.assertEqual(batch.proposal_stats["refill_policy_draws"], sum(range(1, 48)))
        self.assertEqual(batch.proposal_stats["policy_proposals"], 48)
        self.assertEqual(batch.proposal_stats["random_proposals"], 16)
        self.assertLessEqual(max(model.batch_sizes), 32)
        # With separated logits every exclusion draw is the lexicographically
        # smallest remaining legal word, not a uniform or hardcoded fallback.
        random_words = set(guesses[48:])
        expected_policy = tuple(word for word in WORDS if word not in random_words)
        self.assertEqual(guesses[:48], expected_policy)
        self.assertEqual(guesses[0], "aaaaa")
        self.assertNotIn("aaaaa", random_words)
        # Uniform proposals retain support even where model sampling underflows.
        self.assertTrue((batch.old_token_logps[48:].sum(dim=1) < -999).all())
        self.assertTrue(torch.isfinite(batch.old_token_logps).all())
        # Exclusions never alter update masks or old likelihoods. All four next
        # letters remain legal at every branching prefix, including late refills.
        self.assertTrue(batch.masks[:, 2:, :4].all())
        raw = model(batch.inputs)
        selected = raw.gather(1, batch.positions[:, :, None].expand(-1, -1, VOCABULARY_SIZE))
        expected_logps = F.log_softmax(selected[:, :, :4], dim=-1).gather(
            -1, batch.actions[:, :, None]
        ).squeeze(-1)
        expected_logps[:, :2] = 0  # The first two dictionary letters are forced.
        torch.testing.assert_close(batch.old_token_logps, expected_logps)
        self.assertFalse(batch.old_token_logps.requires_grad)

    def test_mixed_depths_only_score_five_letters_under_original_prefix_masks(self):
        model = PrefixPolicy()
        states = tuple(state_at_depth(depth) for depth in (0, 2, 5))
        batch = sample_information_actions(
            model, states, self.decoder, random.Random(19), forward_batch_size=17,
        )
        self.assertLessEqual(max(model.batch_sizes), 17)
        self.assertEqual(batch.actions.shape, (192, 5))
        self.assertEqual(batch.positions.shape, (192, 5))
        self.assertTrue(batch.valid.all())
        self.assertFalse(batch.masks[:, :, 26:].any())
        for group, state in enumerate(states):
            self.assertEqual(len(set(batch.guesses[group])), 64)
            for member, guess in enumerate(batch.guesses[group]):
                row = group * 64 + member
                expected_positions = list(range(len(state.prompt) - 1, len(state.prompt) + 4))
                self.assertEqual(batch.positions[row].tolist(), expected_positions)
                self.assertEqual(decode(batch.actions[row].tolist()), guess)
                actual_input = batch.inputs[row, :len(state.prompt) + 5].tolist()
                self.assertEqual(actual_input, list(state.prompt) + encode(guess))
                for offset in range(5):
                    allowed_letters = {word[offset] for word in WORDS if word.startswith(guess[:offset])}
                    actual_letters = {
                        chr(index + ord("a"))
                        for index in torch.where(batch.masks[row, offset, :26])[0].tolist()
                    }
                    self.assertEqual(actual_letters, allowed_letters)
        torch.testing.assert_close(
            batch.old_token_logps, token_logps(trajectory_logits(model, batch), batch),
        )
        # Mixed-depth right padding must not change any selected causal logit.
        for group in range(len(states)):
            row = group * 64
            prefix = batch.inputs[row:row + 1, :len(states[group].prompt) + 5]
            logits = model(prefix)[0, batch.positions[row]]
            expected = F.log_softmax(logits.masked_fill(~batch.masks[row], -torch.inf), -1)
            expected = expected.gather(-1, batch.actions[row, :, None]).squeeze(-1)
            torch.testing.assert_close(batch.old_token_logps[row], expected)
        middle = batch.slice_groups(1, 2)
        self.assertEqual(middle.guesses, batch.guesses[1:2])
        self.assertEqual(middle.rewards.shape, (1, 64))
        self.assertEqual(middle.proposal_stats["group_count"], 1)
        middle.rewards.fill_(3)
        self.assertTrue((batch.rewards[1] == 3).all())
        middle.inputs[0, 0] = 0
        self.assertEqual(batch.inputs[64, 0].item(), 0)

    def test_evaluation_draws_keep_repeats_and_builder_supports_variable_width(self):
        model = PrefixPolicy(concentrated=True)
        states = (state_at_depth(0), state_at_depth(5))
        for samples, greedy in ((8, False), (1, True)):
            with self.subTest(samples=samples):
                guesses = sample_policy_guesses(
                    model, states, self.decoder, samples=samples,
                    greedy=greedy, forward_batch_size=3,
                )
                self.assertEqual(guesses, (("aaaaa",) * samples,) * 2)
                batch = build_guess_batch(
                    model, states, self.decoder, guesses, forward_batch_size=3,
                )
                self.assertEqual(batch.rewards.shape, (2, samples))
                self.assertEqual(batch.valid.shape, (2 * samples, 5))
                sliced = batch.slice_groups(1, 2)
                self.assertEqual(sliced.inputs.shape[0], samples)
                self.assertEqual(sliced.guesses, (guesses[1],))
                torch.testing.assert_close(
                    sliced.old_token_logps, token_logps(trajectory_logits(model, sliced), sliced),
                )
                sliced.rewards.fill_(9)
                self.assertTrue((batch.rewards[1] == 9).all())
        with self.assertRaises(ValueError):
            build_guess_batch(model, states, self.decoder, (("aaaaa",), ("aaaaa", "aaaab")))
        with self.assertRaises(ValueError):
            build_guess_batch(model, states, self.decoder, (("zzzzz",), ("aaaaa",)))

    def test_seeded_mixture_replays_and_small_dictionary_rejects(self):
        model = PrefixPolicy()
        states = (state_at_depth(1),)
        first = sample_information_actions(model, states, self.decoder, random.Random(7))
        torch.manual_seed(991)
        second = sample_information_actions(model, states, self.decoder, random.Random(7))
        self.assertEqual(first.guesses, second.guesses)
        torch.testing.assert_close(first.old_token_logps, second.old_token_logps)
        with self.assertRaises(ValueError):
            sample_information_actions(
                model, states, LegalWordDecoder(WORDS[:63]), random.Random(7),
            )


if __name__ == "__main__":
    unittest.main()
