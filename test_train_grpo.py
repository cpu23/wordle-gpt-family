import math
import unittest
from types import SimpleNamespace

import torch

from grpo_states import ReachableState
from model import WordleGPT
from tokenizer_v2 import VOCABULARY_SIZE, encode
from train_grpo import (
    checkpoint_rank, group_diagnostics, grpo_loss, policy_diagnostics,
    relative_advantages, sample_actions, selected_logits,
)


class GrpoTests(unittest.TestCase):
    def test_identical_rewards_have_exactly_zero_advantage(self):
        rewards = torch.full((3, 8), math.log(720 / 7))
        self.assertTrue(torch.equal(relative_advantages(rewards), torch.zeros_like(rewards)))
        mixed = relative_advantages(torch.tensor([[0., 1., 2., 3.]]))
        self.assertAlmostEqual(float(mixed.mean()), 0., places=6)
        self.assertAlmostEqual(float(mixed.std(correction=0)), 1., places=6)

    def test_distinct_constrained_draws_match_update_likelihood_and_have_finite_gradients(self):
        torch.manual_seed(12)
        torch.set_num_threads(1)
        model = WordleGPT(vocab_size=VOCABULARY_SIZE, embedding_size=16,
                          num_layers=1, num_heads=2, mlp_size=32).eval()
        words = ('apple', 'apply', 'ample', 'angle', 'ankle', 'addle', 'agile', 'aisle')
        state = ReachableState(secret='apple', prompt=tuple(encode('<P><G>')),
                               candidates=words, history=())
        batch = sample_actions(model, [state], words)
        self.assertEqual(set(batch.guesses[0]), set(words))
        self.assertEqual(len(batch.guesses[0]), 8)
        logits = selected_logits(model, batch)
        loss, stats = grpo_loss(logits, logits.detach(), batch)
        self.assertAlmostEqual(stats['mean_importance_ratio'], 1., places=5)
        self.assertAlmostEqual(stats['raw_prefix_kl_from_sft'], 0., places=6)
        loss.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.parameters()), 0.)

    def test_kl_penalizes_drift_even_with_no_reward_signal(self):
        policy = torch.tensor([[[1., -1., 0.]]], requires_grad=True)
        reference = torch.zeros_like(policy)
        batch = SimpleNamespace(masks=torch.ones_like(policy, dtype=torch.bool),
                                actions=torch.tensor([[0]]), old_logps=torch.tensor([[0.]]),
                                rewards=torch.tensor([[1.]]))
        loss, stats = grpo_loss(policy, reference, batch, kl_beta=0.1)
        self.assertGreater(stats['raw_prefix_kl_from_sft'], 0.)
        loss.backward()
        self.assertGreater(float(policy.grad[0, 0, 0]), 0.)

    def test_checkpoint_ranking_uses_regret_before_kl_on_gameplay_ties(self):
        def report(wins=59, attempts=3.9, regret=0.04, kl=0.0):
            return {
                "gameplay": {"constrained": {"wins": wins, "average_attempts": attempts}},
                "action_regret": {"constrained": {"summary": {"mean_action_regret": regret}}},
                "diagnostics": {"fixed_sft_prefix_kl_from_sft": kl},
            }
        baseline = checkpoint_rank(report())
        self.assertGreater(checkpoint_rank(report(regret=0.03, kl=0.2)), baseline)
        self.assertGreater(checkpoint_rank(report(attempts=3.8, regret=0.5, kl=1.0)), baseline)
        self.assertGreater(checkpoint_rank(report(wins=60, attempts=5.0, regret=1., kl=2.)), baseline)
        self.assertGreater(checkpoint_rank(report(kl=0.01)), checkpoint_rank(report(kl=0.02)))
        self.assertLess(checkpoint_rank(report(regret=0.05)), baseline)

    def test_group_metrics_separate_solve_bonus_and_unforced_diversity(self):
        guesses = ['apple', 'apply', 'ample', 'angle', 'ankle', 'addle', 'agile', 'aisle']
        batch = SimpleNamespace(
            guesses=[guesses, guesses],
            rewards=torch.tensor([[5.] + [0.] * 7, [2.] * 8]),
            reductions=torch.zeros(2, 8),
        )
        metrics = group_diagnostics(
            batch, [SimpleNamespace(secret='apple'), SimpleNamespace(secret='zebra')]
        )
        self.assertAlmostEqual(metrics['mean_group_reward'], 21 / 16)
        self.assertAlmostEqual(metrics['mean_log_candidate_reduction'], 1.)
        self.assertAlmostEqual(metrics['immediate_solve_fraction'], 1 / 16)
        self.assertAlmostEqual(metrics['identical_reward_group_rate'], 0.5)
        self.assertAlmostEqual(metrics['mean_group_reward_std'], math.sqrt(175 / 64) / 2)
        self.assertEqual(metrics['unique_action_count'], 8)
        self.assertAlmostEqual(metrics['unique_action_fraction'], 0.5)
        self.assertEqual(metrics['first_draw_unique_action_count'], 1)
        self.assertAlmostEqual(metrics['first_draw_empirical_action_entropy_nats'], 0.)

    def test_entropy_and_kl_distinguish_raw_from_constrained_policy(self):
        logits = torch.zeros(2, 5, 3)
        masks = torch.tensor([True, True, False]).expand_as(logits)
        batch = SimpleNamespace(masks=masks, rewards=torch.zeros(1, 2))
        uniform = policy_diagnostics(logits, logits, batch)
        self.assertAlmostEqual(uniform['raw_token_entropy_nats'], math.log(3), places=6)
        self.assertAlmostEqual(uniform['first_draw_action_entropy_nats'], 5 * math.log(2), places=6)
        shifted = logits.clone()
        shifted[:, :, 2] = 5
        diagnostics = policy_diagnostics(shifted, logits, batch)
        self.assertGreater(diagnostics['raw_prefix_kl_from_sft'], 0.5)
        self.assertAlmostEqual(diagnostics['constrained_prefix_kl_from_sft'], 0.)
        self.assertAlmostEqual(diagnostics['conditional_action_entropy_nats'], 5 * math.log(2), places=6)


if __name__ == '__main__':
    unittest.main()
