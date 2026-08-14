import unittest

import torch

from model import (
    CONTEXT_LENGTH,
    EMBEDDING_SIZE,
    MLP_SIZE,
    NUM_HEADS,
    NUM_LAYERS,
    VOCAB_SIZE,
    WordleGPT,
)
from train import IGNORE_INDEX, calculate_loss, create_shifted_pairs


class WordleGPTTests(unittest.TestCase):
    def test_requested_architecture_and_output_shape(self):
        model = WordleGPT()
        self.assertEqual(model.token_embedding.num_embeddings, VOCAB_SIZE)
        self.assertEqual(model.token_embedding.embedding_dim, EMBEDDING_SIZE)
        self.assertEqual(model.position_embedding.num_embeddings, CONTEXT_LENGTH)
        self.assertEqual(len(model.blocks), NUM_LAYERS)
        self.assertEqual(model.blocks[0].attention.num_heads, NUM_HEADS)
        self.assertEqual(model.blocks[0].mlp[0].out_features, MLP_SIZE)

        logits = model(torch.randint(VOCAB_SIZE, (2, 12)))
        self.assertEqual(logits.shape, (2, 12, VOCAB_SIZE))

    def test_attention_is_causal(self):
        torch.manual_seed(7)
        model = WordleGPT().eval()
        original = torch.randint(VOCAB_SIZE, (1, 8))
        changed = original.clone()
        changed[:, 5:] = (changed[:, 5:] + 1) % VOCAB_SIZE

        with torch.no_grad():
            original_logits = model(original)
            changed_logits = model(changed)
        torch.testing.assert_close(original_logits[:, :5], changed_logits[:, :5])

    def test_context_limit_is_enforced(self):
        model = WordleGPT()
        with self.assertRaisesRegex(ValueError, "exceeds context length"):
            model(torch.zeros((1, CONTEXT_LENGTH + 1), dtype=torch.long))


class TrainingTests(unittest.TestCase):
    def test_shifted_pairs_pad_inputs_and_mask_targets(self):
        inputs, targets = create_shifted_pairs([[29, 1, 2, 31], [29, 3, 31]])
        self.assertEqual(inputs.tolist(), [[29, 1, 2], [29, 3, 31]])
        self.assertEqual(targets.tolist(), [[1, 2, 31], [3, 31, IGNORE_INDEX]])

    def test_cross_entropy_ignores_padding(self):
        logits = torch.zeros((1, 2, VOCAB_SIZE))
        targets = torch.tensor([[4, IGNORE_INDEX]])
        loss = calculate_loss(logits, targets)
        self.assertAlmostEqual(loss.item(), torch.tensor(32.0).log().item())


if __name__ == "__main__":
    unittest.main()
