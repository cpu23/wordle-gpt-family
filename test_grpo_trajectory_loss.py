import math
import json
import unittest
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from grpo_trajectory_loss import (
    rollout_metrics,
    token_logps,
    trajectory_gradient_metrics,
    trajectory_logps,
    trajectory_loss,
    trajectory_policy_metrics,
)


GROUP_SIZE = 8


def make_batch(groups=1, actions_width=3, vocabulary_size=4, valid=None, rewards=None):
    rows = groups * GROUP_SIZE
    if valid is None:
        valid = torch.ones((rows, actions_width), dtype=torch.bool)
    masks = torch.ones((rows, actions_width, vocabulary_size), dtype=torch.bool)
    if rewards is None:
        rewards = torch.zeros((groups, GROUP_SIZE))
    return SimpleNamespace(
        inputs=torch.zeros((rows, 1), dtype=torch.long),
        positions=torch.zeros((rows, actions_width), dtype=torch.long),
        actions=torch.zeros((rows, actions_width), dtype=torch.long),
        masks=masks,
        valid=valid,
        old_token_logps=torch.zeros((rows, actions_width)),
        rewards=rewards,
        won=torch.zeros((groups, GROUP_SIZE), dtype=torch.bool),
        attempts=torch.zeros((groups, GROUP_SIZE), dtype=torch.long),
    )


class TrajectoryObjectiveTests(unittest.TestCase):
    def test_token_logps_and_diagnostic_sum_include_every_generated_turn(self):
        valid = torch.zeros((GROUP_SIZE, 15), dtype=torch.bool)
        valid[:, :10] = True  # two complete five-letter guesses
        batch = make_batch(actions_width=15, vocabulary_size=3, valid=valid)
        logits = torch.zeros((GROUP_SIZE, 15, 3))
        logits[:, :5, 0] = 2.0
        logits[:, 5:10, 0] = 1.0
        logits[:, 10:, 0] = 50.0  # excluded padding/control positions

        result = trajectory_logps(logits, batch)
        selected = token_logps(logits, batch)
        self.assertEqual(selected.shape, valid.shape)
        torch.testing.assert_close(selected[:, :10], F.log_softmax(logits[:, :10], -1)[..., 0])
        self.assertTrue(torch.equal(selected[:, 10:], torch.zeros_like(selected[:, 10:])))
        first_guess_only = 5 * F.log_softmax(logits[0, 0], -1)[0]
        expected = first_guess_only + 5 * F.log_softmax(logits[0, 5], -1)[0]
        self.assertTrue(torch.allclose(result, expected.expand_as(result)))
        self.assertFalse(torch.allclose(result[0, 0], first_guess_only))
        self.assertFalse(torch.allclose(result[0, 0], expected / 2))

    def test_padding_and_feedback_positions_are_excluded_from_policy_and_kl(self):
        valid = torch.zeros((GROUP_SIZE, 3), dtype=torch.bool)
        valid[:, 0] = True
        batch = make_batch(actions_width=3, vocabulary_size=4, valid=valid,
                           rewards=torch.tensor([[1.0] + [0.0] * 7]))
        batch.masks[:, 0, 2:] = False
        policy = torch.zeros((GROUP_SIZE, 3, 4), requires_grad=True)
        reference = torch.zeros_like(policy)
        reference[:, 0, 0] = 1.0
        batch.old_token_logps = token_logps(policy.detach(), batch)

        baseline = trajectory_policy_metrics(policy, reference, batch)
        perturbed = policy.detach().clone()
        perturbed[:, 1:, :] = float("nan")
        batch.actions[:, 1:] = 999  # invalid history/control slots are not sampled letters
        batch.old_token_logps[:, 1:] = float("nan")
        expected_logps = token_logps(policy, batch)
        self.assertTrue(torch.equal(token_logps(perturbed, batch), expected_logps))
        changed = trajectory_policy_metrics(perturbed, reference, batch)
        self.assertEqual(baseline, changed)

        contaminated = perturbed.requires_grad_()
        loss, _ = trajectory_loss(contaminated, reference, batch)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.equal(contaminated.grad[:, 1:], torch.zeros_like(contaminated.grad[:, 1:])))
        loss, _ = trajectory_loss(policy, reference, batch)
        loss.backward()
        self.assertTrue(torch.equal(policy.grad[:, 1:, :], torch.zeros_like(policy.grad[:, 1:, :])))

    def test_late_turn_tokens_receive_policy_gradient(self):
        batch = make_batch(actions_width=10, vocabulary_size=3,
                           rewards=torch.tensor([[1.0] + [0.0] * 7]))
        policy = torch.zeros((GROUP_SIZE, 10, 3), requires_grad=True)
        batch.old_token_logps.fill_(-math.log(3))
        loss, _ = trajectory_loss(policy, policy.detach(), batch)
        loss.backward()
        self.assertGreater(float(policy.grad[0, 9].abs().sum()), 0.0)

    def test_advantages_are_normalized_within_each_secret_group(self):
        batch = make_batch(groups=2, actions_width=1, vocabulary_size=3,
                           rewards=torch.stack((torch.full((GROUP_SIZE,), 4.0),
                                                torch.tensor([0., 1., 2., 3., 4., 5., 6., 0.]))))
        policy = torch.zeros((2 * GROUP_SIZE, 1, 3), requires_grad=True)
        batch.old_token_logps.fill_(-math.log(3))
        loss, _ = trajectory_loss(policy, policy.detach(), batch, kl_beta=0.0)
        loss.backward()
        self.assertTrue(torch.equal(policy.grad[:GROUP_SIZE], torch.zeros_like(policy.grad[:GROUP_SIZE])))
        self.assertGreater(float(policy.grad[GROUP_SIZE:].abs().sum()), 0.0)

    def test_identical_rewards_have_no_policy_signal_but_kl_still_penalizes_drift(self):
        batch = make_batch(actions_width=1, vocabulary_size=3,
                           rewards=torch.ones((1, GROUP_SIZE)))
        policy = torch.tensor([1.0, -1.0, 0.0]).expand(GROUP_SIZE, 1, 3).clone().requires_grad_()
        reference = torch.zeros_like(policy)
        loss, stats = trajectory_loss(policy, reference, batch, kl_beta=0.1)
        self.assertGreater(stats["raw_prefix_kl_from_sft"], 0.0)
        loss.backward()
        self.assertGreater(float(policy.grad.abs().sum()), 0.0)

    def test_compensating_token_ratios_clip_independently_not_as_trajectory_product(self):
        batch = make_batch(actions_width=2, vocabulary_size=5,
                           rewards=torch.tensor([[1.0] + [0.0] * 7]))
        policy = torch.zeros((GROUP_SIZE, 2, 5), requires_grad=True)
        ratios = torch.tensor([2.0, 0.5])
        batch.old_token_logps = token_logps(policy.detach(), batch) - ratios.log()
        loss, stats = trajectory_loss(policy, policy.detach(), batch, kl_beta=0.0)
        positive_advantage = math.sqrt(7)
        negative_advantage = -1 / math.sqrt(7)
        expected = -(0.85 * positive_advantage + 7 * 1.4 * negative_advantage) / GROUP_SIZE
        self.assertAlmostEqual(float(loss.detach()), expected, places=6)
        self.assertGreater(float(loss.detach()), 0.1)  # product-of-ratios clipping would give zero
        self.assertAlmostEqual(stats["mean_importance_ratio"], 1.25)
        self.assertAlmostEqual(stats["token_ratio_min"], 0.5)
        self.assertAlmostEqual(stats["token_ratio_max"], 2.0)
        self.assertAlmostEqual(stats["token_clip_fraction"], 1.0)
        self.assertAlmostEqual(stats["mean_abs_advantage"], (positive_advantage - 7 * negative_advantage) / 8, places=6)
        loss.backward()
        self.assertEqual(float(policy.grad[0, 0].abs().sum()), 0.0)
        self.assertLess(float(policy.grad[0, 1, 0]), 0.0)
        self.assertGreater(float(policy.grad[1, 0, 0]), 0.0)
        self.assertEqual(float(policy.grad[1, 1].abs().sum()), 0.0)

    def test_duplicating_one_rollouts_tokens_preserves_reward_kl_and_parameter_gradients(self):
        rows = torch.arange(GROUP_SIZE, dtype=torch.float32)
        parameters = torch.stack((0.2 * rows, 0.1 - 0.1 * rows, -0.3 * rows), dim=-1).requires_grad_()
        policy = parameters[:, None, :].expand(-1, 15, -1)
        reference = torch.zeros_like(policy)
        results = []
        for first_length in (5, 15):
            valid = torch.zeros((GROUP_SIZE, 15), dtype=torch.bool)
            valid[:, :5] = True
            valid[0, :first_length] = True
            batch = make_batch(actions_width=15, vocabulary_size=3, valid=valid,
                               rewards=torch.tensor([[1.0] + [0.0] * 7]))
            batch.masks[:, :, 2] = False
            batch.old_token_logps = token_logps(policy.detach(), batch)
            batch.old_token_logps[0, :first_length] -= math.log(1.1)
            reward_loss, _ = trajectory_loss(policy, reference, batch, kl_beta=0.0)
            combined_loss, stats = trajectory_loss(policy, reference, batch, kl_beta=0.3)
            reward_gradient, = torch.autograd.grad(reward_loss, parameters, retain_graph=True)
            combined_gradient, = torch.autograd.grad(combined_loss, parameters, retain_graph=True)
            results.append((reward_loss, combined_loss, reward_gradient, combined_gradient, stats))
        for left, right in zip(results[0][:4], results[1][:4]):
            torch.testing.assert_close(left, right)
        for key in ("raw_prefix_kl_from_sft", "constrained_prefix_kl_from_sft", "mean_importance_ratio"):
            self.assertAlmostEqual(results[0][4][key], results[1][4][key], places=6)
        self.assertGreater(results[0][4]["raw_prefix_kl_from_sft"], 0.0)
        self.assertGreater(results[0][4]["constrained_prefix_kl_from_sft"], 0.0)

    def test_gradient_metrics_measure_global_parameter_contributions_without_mutating_grad(self):
        model = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.2, -0.1], [-0.3, 0.1], [0.1, 0.3]]))
        model.register_parameter("unused", torch.nn.Parameter(torch.tensor([2.0])))
        generated = torch.tensor([1, 2, 3, 4, 5, 6, 1, 2])
        remaining = torch.tensor([1, 2, 3, 4, 5, 6, 6, 5])
        valid = torch.arange(30)[None, :] < 5 * generated[:, None]
        rewards = torch.arange(GROUP_SIZE, dtype=torch.float32).reshape(1, -1)
        batch = make_batch(actions_width=30, vocabulary_size=3, valid=valid, rewards=rewards)
        batch.attempts = (6 - remaining + generated).reshape(1, -1)
        features = torch.stack((torch.arange(GROUP_SIZE) * 0.2 + 0.3, torch.ones(GROUP_SIZE)), -1)
        row_logits = model(features)
        policy = row_logits[:, None, :].expand(-1, 30, -1)
        reference = torch.zeros_like(policy)
        batch.old_token_logps = token_logps(policy.detach(), batch)
        model.weight.grad = torch.full_like(model.weight, 0.17)
        saved_grad = model.weight.grad.clone()
        beta = 0.3
        metrics = trajectory_gradient_metrics(model, policy, reference, batch, beta)
        torch.testing.assert_close(model.weight.grad, saved_grad, rtol=0, atol=0)
        self.assertIsNone(model.unused.grad)
        json.dumps(metrics, allow_nan=False)

        # An independent one-position expression is exact here because each
        # rollout repeats its logits. Its derivatives are parameter-space, not
        # logit-space, and every bucket retains the global batch denominator.
        logp = F.log_softmax(row_logits, -1)
        advantage = ((rewards - rewards.mean()) / rewards.std(correction=0)).flatten()
        reward_rows = -(logp[:, 0] - logp[:, 0].detach()).exp() * advantage
        kl_rows = (logp.exp() * (logp + math.log(3))).sum(-1)
        combined_rows = reward_rows + beta * kl_rows
        for key, lengths in (("by_generated_guesses", generated), ("by_remaining_guesses", remaining)):
            for length in range(1, 7):
                selected = lengths == length
                bucket = metrics[key][str(length)]
                self.assertEqual(bucket["count"], int(selected.sum()))
                self.assertAlmostEqual(bucket["mean_abs_advantage"], float(advantage[selected].abs().mean()))
                self.assertAlmostEqual(bucket["mean_token_normalization_weight"], float((5 * generated[selected]).float().reciprocal().mean()))
                for name, terms in (("reward_gradient_l2", reward_rows), ("combined_gradient_l2", combined_rows)):
                    expected, = torch.autograd.grad(terms[selected].sum() / GROUP_SIZE, model.weight, retain_graph=True)
                    self.assertAlmostEqual(bucket[name], float(expected.norm()), places=6)
        for name, terms in (("reward_gradient_l2", reward_rows), ("combined_gradient_l2", combined_rows)):
            expected, = torch.autograd.grad(terms.mean(), model.weight, retain_graph=True)
            self.assertAlmostEqual(metrics[name], float(expected.norm()), places=6)

        model.zero_grad(set_to_none=True)
        loss, _ = trajectory_loss(policy, reference, batch, beta)
        loss.backward()  # diagnostic backward calls must have retained this graph
        self.assertAlmostEqual(float(model.weight.grad.norm()), metrics["combined_gradient_l2"], places=6)

    def test_gradient_metrics_empty_buckets_and_zero_advantage_groups(self):
        model = torch.nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        batch = make_batch(actions_width=5, vocabulary_size=2, rewards=torch.ones((1, GROUP_SIZE)))
        batch.attempts.fill_(6)  # one generated guess after five supplied guesses
        policy = model(torch.ones(GROUP_SIZE, 5, 1))
        reference = torch.zeros_like(policy)
        batch.old_token_logps = token_logps(policy.detach(), batch)
        metrics = trajectory_gradient_metrics(model, policy, reference, batch, 0.2)
        for key in ("by_generated_guesses", "by_remaining_guesses"):
            self.assertEqual(metrics[key]["1"]["count"], GROUP_SIZE)
            self.assertEqual(metrics[key]["1"]["mean_abs_advantage"], 0.0)
            self.assertEqual(metrics[key]["1"]["reward_gradient_l2"], 0.0)
            self.assertGreater(metrics[key]["1"]["combined_gradient_l2"], 0.0)
            for length in range(2, 7):
                self.assertEqual(metrics[key][str(length)]["count"], 0)
                self.assertIsNone(metrics[key][str(length)]["mean_abs_advantage"])
                self.assertEqual(metrics[key][str(length)]["reward_gradient_l2"], 0.0)
                self.assertEqual(metrics[key][str(length)]["combined_gradient_l2"], 0.0)
        json.dumps(metrics, allow_nan=False)

    def test_rollout_metrics_use_winning_attempts_and_are_json_compatible(self):
        batch = make_batch(groups=2, actions_width=1, rewards=torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
             [5.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
        ))
        batch.won[1, :2] = True
        batch.attempts.fill_(6)
        batch.attempts[1, 0] = 2
        batch.attempts[1, 1] = 4
        metrics = rollout_metrics(batch)
        self.assertEqual(metrics["winning_rollouts"], 2)
        self.assertEqual(metrics["sampled_rollouts"], 16)
        self.assertEqual(metrics["mean_guesses_for_winning_rollouts"], 3.0)
        self.assertEqual(metrics["identical_reward_group_rate"], 0.5)
        json.dumps(metrics)

    def test_no_winners_has_null_mean_winning_attempts(self):
        metrics = rollout_metrics(make_batch(actions_width=1))
        self.assertIsNone(metrics["mean_guesses_for_winning_rollouts"])


if __name__ == "__main__":
    unittest.main()
