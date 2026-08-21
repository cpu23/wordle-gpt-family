import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from evaluate_v2 import _build_parser, load_v2_model, play_secret
from model import WordleGPT
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

    def test_constrained_decoding_forces_an_allowed_word(self):
        result = play_secret(
            ScriptedModel("zzzzz"),
            "crane",
            frozenset({"crane", "grape"}),
            constrained=True,
        )
        self.assertIn(result.guesses[0], {"crane", "grape"})
        self.assertEqual(result.invalid_guesses, 0)

    def test_constrained_decoding_wins_on_allowed_scripted_word(self):
        result = play_secret(
            ScriptedModel("crane"),
            "crane",
            frozenset({"crane"}),
            constrained=True,
        )
        self.assertTrue(result.won)
        self.assertEqual(result.guesses, ("crane",))

    def test_evaluate_model_rejects_unknown_decode_mode(self):
        from evaluate_v2 import evaluate_model

        with self.assertRaises(ValueError):
            evaluate_model(
                ScriptedModel("crane"),
                ["crane"],
                ["crane"],
                decode="beam",
            )

    def test_checkpoint_loader_restores_stored_architecture(self):
        model = WordleGPT(
            vocab_size=VOCABULARY_SIZE,
            embedding_size=16,
            num_layers=1,
            num_heads=2,
            mlp_size=32,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "custom.pt"
            torch.save(
                {
                    "vocabulary_size": VOCABULARY_SIZE,
                    "model_config": model.config,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            restored = load_v2_model(checkpoint_path, "cpu")
        self.assertEqual(restored.config, model.config)

    def test_cli_defaults_to_validation_secrets(self):
        self.assertEqual(_build_parser().parse_args(["model.pt"]).split, "validation")

    def test_cli_accepts_decode_mode(self):
        self.assertEqual(
            _build_parser().parse_args(["model.pt", "--decode", "constrained"]).decode,
            "constrained",
        )
        self.assertEqual(_build_parser().parse_args(["model.pt"]).decode, "raw")


if __name__ == "__main__":
    unittest.main()
