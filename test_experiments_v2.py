import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from experiments_v2 import load_initialization_checkpoint, train_stage
from model import WordleGPT
from tokenizer_v2 import POLICY_TOKEN, VOCABULARY_SIZE, encode
from train import IGNORE_INDEX, create_shifted_pairs
from train_v2 import EXAMPLE_TYPE_TO_ID, V2SplitData


class V2ExperimentTests(unittest.TestCase):
    def test_early_stopping_keeps_best_validation_checkpoint(self):
        token_ids = encode(f"{POLICY_TOKEN}<G>crane<E>")
        inputs, unmasked_targets = create_shifted_pairs(
            [token_ids],
            vocab_size=VOCABULARY_SIZE,
        )
        targets = torch.full_like(unmasked_targets, IGNORE_INDEX)
        targets[:, 1:6] = unmasked_targets[:, 1:6]
        data = V2SplitData(
            inputs=inputs,
            targets=targets,
            example_type_ids=torch.tensor([EXAMPLE_TYPE_TO_ID["expert"]]),
        )

        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            output_dir = Path(directory)
            best_path, records = train_stage(
                data,
                data,
                output_dir,
                objective="expert",
                batch_size=1,
                eval_batch_size=1,
                learning_rate=0.0,
                patience=2,
                max_epochs=10,
                device="cpu",
            )
            best = torch.load(best_path, weights_only=True)
            metrics = [
                json.loads(line)
                for line in (output_dir / "metrics.jsonl").read_text().splitlines()
            ]

        self.assertEqual(len(records), 3)
        self.assertEqual(records[-1].epoch, 2)
        self.assertEqual(best["epoch"], 0)
        self.assertIn("model_state_dict", best)
        self.assertEqual(best["model_config"]["vocab_size"], VOCABULARY_SIZE)
        self.assertEqual(sum(record["improved"] for record in metrics), 1)

    def test_v1_checkpoint_expands_only_vocabulary_parameters(self):
        torch.manual_seed(7)
        v1_model = WordleGPT(vocab_size=32)
        torch.manual_seed(11)
        v2_model = WordleGPT(vocab_size=VOCABULARY_SIZE)
        new_token_rows = v2_model.token_embedding.weight[32:].detach().clone()
        new_output_rows = v2_model.output.weight[32:].detach().clone()
        new_output_bias = v2_model.output.bias[32:].detach().clone()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "v1.pt"
            torch.save({"model_state_dict": v1_model.state_dict()}, checkpoint_path)
            load_initialization_checkpoint(
                v2_model,
                checkpoint_path,
                device=torch.device("cpu"),
            )

        self.assertTrue(
            torch.equal(v2_model.token_embedding.weight[:32], v1_model.token_embedding.weight)
        )
        self.assertTrue(torch.equal(v2_model.output.weight[:32], v1_model.output.weight))
        self.assertTrue(torch.equal(v2_model.output.bias[:32], v1_model.output.bias))
        self.assertTrue(torch.equal(v2_model.token_embedding.weight[32:], new_token_rows))
        self.assertTrue(torch.equal(v2_model.output.weight[32:], new_output_rows))
        self.assertTrue(torch.equal(v2_model.output.bias[32:], new_output_bias))


if __name__ == "__main__":
    unittest.main()
