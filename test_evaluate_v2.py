import unittest

import torch
from torch import nn

from evaluate_v2 import play_secret
from tokenizer_v2 import VOCABULARY_SIZE, encode


class ScriptedModel(nn.Module):
    def __init__(self, word: str) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.context_length = 96
        self.token_ids = encode(word)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = torch.full(
            (*tokens.shape, VOCABULARY_SIZE),
            -100.0,
            device=tokens.device,
        )
        next_index = tokens.shape[1] - 2
        logits[:, -1, self.token_ids[next_index]] = 100.0 + self.anchor
        return logits


class GameplayTests(unittest.TestCase):
    def test_valid_generated_secret_wins_in_one_guess(self):
        result = play_secret(ScriptedModel("crane"), "crane", frozenset({"crane"}))
        self.assertTrue(result.won)
        self.assertEqual(result.guesses, ("crane",))
        self.assertEqual(result.invalid_guesses, 0)

    def test_out_of_dictionary_word_is_invalid_and_ends_game(self):
        result = play_secret(ScriptedModel("zzzzz"), "crane", frozenset({"crane"}))
        self.assertFalse(result.won)
        self.assertEqual(result.guesses, ("zzzzz",))
        self.assertEqual(result.invalid_guesses, 1)


if __name__ == "__main__":
    unittest.main()
