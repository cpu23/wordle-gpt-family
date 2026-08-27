import math
import tempfile
import unittest
from pathlib import Path

import torch

from model import WordleGPT
from tokenizer_v2 import VOCABULARY_SIZE, encode
from train_dpo import completion_logps, dpo_loss


class DPOTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = WordleGPT(vocab_size=VOCABULARY_SIZE, context_length=16, embedding_size=16, num_layers=1, num_heads=2, mlp_size=32)
        prompt = encode("<P><G>")
        self.prompts = torch.zeros((2, 16), dtype=torch.long)
        self.prompts[:, : len(prompt)] = torch.tensor(prompt)
        self.lengths = torch.tensor([len(prompt), len(prompt)])
        self.chosen = torch.tensor([encode("irate"), encode("crate")])
        self.rejected = torch.tensor([encode("fuzzy"), encode("zebra")])

    def test_completion_logps_sum_exactly_five_letter_probabilities(self):
        actual = completion_logps(self.model, self.prompts, self.lengths, self.chosen)
        inputs = self.prompts.clone()
        positions = self.lengths[:, None] + torch.arange(5)
        inputs.scatter_(1, positions, self.chosen)
        logits = self.model(inputs)
        manual = torch.stack([
            sum(torch.log_softmax(logits[row, self.lengths[row] + offset - 1], dim=-1)[self.chosen[row, offset]] for offset in range(5))
            for row in range(2)
        ])
        torch.testing.assert_close(actual, manual)

    def test_identical_policy_and_reference_have_log_two_dpo_loss(self):
        chosen = completion_logps(self.model, self.prompts, self.lengths, self.chosen)
        rejected = completion_logps(self.model, self.prompts, self.lengths, self.rejected)
        loss = dpo_loss(chosen, rejected, chosen, rejected, beta=0.1)
        torch.testing.assert_close(loss, torch.full_like(loss, math.log(2)))

    def test_improving_policy_margin_reduces_loss(self):
        reference_chosen = torch.tensor([-10.0, -10.0])
        reference_rejected = torch.tensor([-11.0, -11.0])
        baseline = dpo_loss(reference_chosen, reference_rejected, reference_chosen, reference_rejected, beta=0.2)
        improved = dpo_loss(reference_chosen + 2, reference_rejected, reference_chosen, reference_rejected, beta=0.2)
        self.assertTrue(torch.all(improved < baseline))

    def test_nonpositive_beta_is_rejected(self):
        values = torch.zeros(2)
        with self.assertRaisesRegex(ValueError, "beta"):
            dpo_loss(values, values, values, values, beta=0)


if __name__ == "__main__":
    unittest.main()
