import copy
import math
import unittest

import torch
from torch.nn import functional as F

from model import WordleGPT
from soft_policy import candidate_sequence_logps, distillation_loss


def direct_five_letter_logps(model, prompts, lengths, candidates):
    """Independent reference: five successive unmodified model forwards."""
    rows = []
    for index, length in enumerate(lengths.tolist()):
        context = prompts[index : index + 1, :length].expand(candidates.shape[1], -1)
        terms = []
        for letter in range(5):
            logps = F.log_softmax(model(context)[:, -1], dim=-1)
            terms.append(logps.gather(-1, candidates[index, :, letter, None]).squeeze(-1))
            context = torch.cat((context, candidates[index, :, letter, None]), dim=-1)
        rows.append(torch.stack(terms, dim=-1).sum(dim=-1))
    return torch.stack(rows)


def tiny_model():
    return WordleGPT(
        vocab_size=35,
        context_length=12,
        embedding_size=12,
        num_layers=2,
        num_heads=3,
        mlp_size=24,
    )


class CandidateSequenceTests(unittest.TestCase):
    def test_prefix_reuse_matches_five_forwards_and_every_parameter_gradient(self):
        # Different attention reduction orders are not bitwise identical. Float64
        # is deliberately strict; float32 tolerances also cover accumulated grads.
        for dtype, rtol, atol in (
            (torch.float64, 1e-8, 1e-9),
            (torch.float32, 2e-5, 2e-5),
        ):
            with self.subTest(dtype=dtype):
                torch.manual_seed(19)
                cached = tiny_model().to(dtype=dtype).train()
                reference = copy.deepcopy(cached)
                prompts = torch.tensor([[32, 4, 28, 34], [32, 34, 999, 999], [32, 9, 27, 34]])
                lengths = torch.tensor([4, 2, 4])  # regrouping must restore original order
                candidates = torch.randint(0, 26, (3, 2, 5))
                observed = candidate_sequence_logps(cached, prompts, lengths, candidates)
                expected = direct_five_letter_logps(reference, prompts, lengths, candidates)
                self.assertEqual(observed.shape, (3, 2))
                torch.testing.assert_close(observed, expected, rtol=rtol, atol=atol)
                weights = torch.tensor([[0.1, 0.2], [0.15, 0.05], [0.3, 0.2]], dtype=dtype)
                (observed * weights).sum().backward()
                (expected * weights).sum().backward()
                for (name, actual), (_, wanted) in zip(
                    cached.named_parameters(), reference.named_parameters()
                ):
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(actual.grad)
                        self.assertIsNotNone(wanted.grad)
                        torch.testing.assert_close(actual.grad, wanted.grad, rtol=rtol, atol=atol)
                self.assertGreater(float(cached.token_embedding.weight.grad.abs().sum()), 0)

    def test_only_five_letters_are_scored_with_all_35_tokens_in_denominator(self):
        model = tiny_model().double()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.copy_(torch.linspace(-1, 1, 35, dtype=torch.float64))
            model.output.bias[34] = 3  # Nonletter mass must not be removed.
        candidates = torch.tensor([[[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]]]).expand(2, -1, -1)
        prompts = torch.tensor([[32, 34, 999, 999], [32, 7, 28, 34]])
        observed = candidate_sequence_logps(model, prompts, torch.tensor([2, 4]), candidates)
        raw_logps = F.log_softmax(model.output.bias, dim=-1)
        expected = raw_logps[candidates].sum(dim=-1)
        torch.testing.assert_close(observed, expected, rtol=1e-8, atol=1e-9)
        observed.sum().backward()
        counts = torch.bincount(candidates.flatten(), minlength=35).double()
        torch.testing.assert_close(
            model.output.bias.grad,
            counts - candidates.numel() * raw_logps.detach().exp(),
            rtol=1e-8,
            atol=1e-9,
        )

    def test_context_boundary_uses_four_input_letters_not_five(self):
        torch.manual_seed(21)
        model = tiny_model().double()
        prompts = torch.randint(0, 35, (1, 8))
        candidates = torch.randint(0, 26, (1, 2, 5))
        actual = candidate_sequence_logps(model, prompts, torch.tensor([8]), candidates)
        expected = direct_five_letter_logps(model, prompts, torch.tensor([8]), candidates)
        torch.testing.assert_close(actual, expected, rtol=1e-8, atol=1e-9)
        with self.assertRaisesRegex(ValueError, "exceeds context length"):
            candidate_sequence_logps(model, torch.zeros((1, 9), dtype=torch.long), torch.tensor([9]), candidates)

    def test_nonzero_attention_dropout_is_not_silently_changed(self):
        model = tiny_model()
        model.blocks[0].attention.dropout = 0.1
        with self.assertRaisesRegex(ValueError, "dropout=0"):
            candidate_sequence_logps(
                model, torch.tensor([[32, 34]]), torch.tensor([2]), torch.zeros((1, 2, 5), dtype=torch.long)
            )


class DistillationLossTests(unittest.TestCase):
    def test_full_cross_entropy_metrics_and_detached_teacher(self):
        student = torch.tensor([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]], dtype=torch.float64)
        sequence_logps = student.log().requires_grad_()
        teacher = torch.tensor([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
        ranks = torch.tensor([[1, 4, 10], [2, 3, 8]])
        loss, metrics = distillation_loss(sequence_logps, teacher, ranks)
        expected_ce = -(teacher.detach() * student.log()).sum(dim=-1).mean()
        expected_teacher_entropy = -sum(q * math.log(q) for q in (0.2, 0.3, 0.5)) / 2
        torch.testing.assert_close(loss, expected_ce)
        self.assertAlmostEqual(float(metrics["teacher_entropy"]), expected_teacher_entropy)
        self.assertAlmostEqual(float(metrics["teacher_student_kl"]), float(expected_ce) - expected_teacher_entropy)
        self.assertAlmostEqual(float(metrics["student_entropy"]), -sum(p * math.log(p) for p in (0.5, 0.3, 0.2)))
        self.assertAlmostEqual(float(metrics["rank1_probability"]), 0.25)
        self.assertAlmostEqual(float(metrics["top3_probability"]), 0.5)
        self.assertAlmostEqual(float(metrics["top8_probability"]), 0.9)
        for metric in metrics.values():
            self.assertEqual(metric.ndim, 0)
            self.assertFalse(metric.requires_grad)
        loss.backward()
        torch.testing.assert_close(sequence_logps.grad, (student - teacher.detach()) / 2)
        self.assertIsNone(teacher.grad)

    def test_candidate_normalization_is_invariant_to_sequence_probability_scale(self):
        sequence_logps = torch.tensor([[-24.0, -26.0, -25.0], [-16.0, -17.0, -19.0]], dtype=torch.float64)
        teacher = torch.tensor([[0.5, 0.25, 0.25], [0.2, 0.3, 0.5]], dtype=torch.float64)
        ranks = torch.tensor([[10, 1, 5], [3, 2, 8]])
        baseline, metrics = distillation_loss(sequence_logps, teacher, ranks)
        shifted, shifted_metrics = distillation_loss(sequence_logps + torch.tensor([[20.0], [-100.0]]), teacher, ranks)
        torch.testing.assert_close(baseline, shifted)
        for key in metrics:
            torch.testing.assert_close(metrics[key], shifted_metrics[key])
        policy = sequence_logps.softmax(dim=-1)
        torch.testing.assert_close(policy.sum(dim=-1), torch.ones(2, dtype=torch.float64))
        torch.testing.assert_close(baseline, -(teacher * policy.log()).sum(dim=-1).mean())


if __name__ == "__main__":
    unittest.main()
