import gzip
import io
import json
import math
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from model import VOCAB_SIZE, WordleGPT
from tokenizer import decode
from train_nested import (
    TokenizedState,
    create_split_data,
    evaluate_losses,
    load_nested_examples,
    nested_sizes,
    train_nested_prefix,
)


class NestedDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temporary_directory.name)
        manifest = {"nested_datasets": {"3": {}, "2": {}}}
        (self.data_dir / "manifest.json").write_text(json.dumps(manifest))
        states = [
            {
                "split": "train",
                "history": [],
                "action": "crane",
                "feedback": "XXGYX",
                "strategy": "clever",
            },
            {
                "split": "validation",
                "history": [{"guess": "slate", "feedback": "XXGYX"}],
                "action": "drink",
                "feedback": "GGGGG",
                "strategy": "partly-random",
            },
            {
                "split": "test",
                "history": [],
                "action": "colon",
                "feedback": "GGGGG",
                "strategy": "random",
            },
        ]
        with gzip.open(
            self.data_dir / "states.jsonl.gz", "wt", encoding="utf-8"
        ) as output:
            for state in states:
                output.write(json.dumps(state) + "\n")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_sizes_are_numeric_and_nested(self):
        self.assertEqual(nested_sizes(self.data_dir), (2, 3))

    def test_loader_uses_prefix_and_keeps_splits_separate(self):
        examples = load_nested_examples(self.data_dir, 2)
        self.assertEqual(len(examples["train"]), 1)
        self.assertEqual(len(examples["validation"]), 1)
        self.assertEqual(
            decode(examples["train"][0].tokens), "<G>crane<F>00210<E>"
        )
        self.assertEqual(
            decode(examples["validation"][0].tokens),
            "<G>slate<F>00210<G>drink<F>22222<E>",
        )
        self.assertEqual(examples["validation"][0].current_guess_target_start, 12)


class NestedMetricsTests(unittest.TestCase):
    def setUp(self):
        self.data = create_split_data(
            [
                TokenizedState(
                    tokens=[29, 2, 17, 0, 13, 4, 30, 26, 26, 28, 27, 26, 31],
                    strategy="clever",
                    current_guess_target_start=0,
                )
            ]
        )

    def test_evaluation_categorizes_every_target_token(self):
        model = WordleGPT()
        with torch.no_grad():
            model.output.weight.zero_()
            model.output.bias.zero_()

        losses = evaluate_losses(model, self.data, batch_size=1)

        expected = math.log(VOCAB_SIZE)
        for name in (
            "overall",
            "guess_letter",
            "feedback",
            "structure",
            "end",
            "trajectory_clever",
            "next_guess_clever",
        ):
            self.assertAlmostEqual(losses[name], expected, places=5)

    def test_training_records_steps_tokens_epochs_and_learning_rate(self):
        with redirect_stdout(io.StringIO()):
            _, records = train_nested_prefix(
                self.data,
                self.data,
                num_steps=1,
                batch_size=1,
                eval_batch_size=1,
                device="cpu",
                checkpoints=(0, 1),
            )

        self.assertEqual(records[0].tokens_seen, 0)
        self.assertEqual(records[0].epochs, 0.0)
        self.assertEqual(records[1].tokens_seen, 12)
        self.assertEqual(records[1].epochs, 1.0)
        self.assertEqual(records[1].learning_rate, 3e-4)


if __name__ == "__main__":
    unittest.main()
