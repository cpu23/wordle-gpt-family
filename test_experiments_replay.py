import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from experiments_replay import (
    OBJECTIVES,
    _sample_expert_epoch,
    even_replay_schedule,
    replay_batch_counts,
    train_with_replay,
)
from model import WordleGPT
from tokenizer_v2 import VOCABULARY_SIZE, encode
from train import IGNORE_INDEX, create_shifted_pairs
from train_v2 import EXAMPLE_TYPE_TO_ID, V2SplitData


def masked_data(text: str, start: int, stop: int, objective: str) -> V2SplitData:
    token_ids = encode(text)
    inputs, unmasked = create_shifted_pairs(
        [token_ids],
        vocab_size=VOCABULARY_SIZE,
    )
    targets = torch.full_like(unmasked, IGNORE_INDEX)
    targets[:, start:stop] = unmasked[:, start:stop]
    return V2SplitData(
        inputs=inputs,
        targets=targets,
        example_type_ids=torch.tensor([EXAMPLE_TYPE_TO_ID[objective]]),
    )


class ReplayScheduleTests(unittest.TestCase):
    def test_batch_counts_preserve_a_full_expert_epoch(self):
        counts = replay_batch_counts(
            2505,
            {"expert": 0.9, "mechanics": 0.1, "consistency": 0.0},
        )
        self.assertEqual(counts, {"expert": 2505, "mechanics": 278, "consistency": 0})

    def test_token_equivalent_epochs_carry_batch_overshoot_forward(self):
        one = masked_data("<P><G>crane<E>", 1, 6, "expert")
        data = V2SplitData(
            inputs=one.inputs.repeat(3, 1),
            targets=one.targets.repeat(3, 1),
            example_type_ids=one.example_type_ids.repeat(3),
        )
        generator = torch.Generator().manual_seed(0)
        first, tokens_seen = _sample_expert_epoch(
            data,
            target_tokens=15,
            tokens_seen=0,
            batch_size=2,
            generator=generator,
        )
        second, tokens_seen = _sample_expert_epoch(
            data,
            target_tokens=30,
            tokens_seen=tokens_seen,
            batch_size=2,
            generator=generator,
        )
        self.assertEqual((len(first), len(second), tokens_seen), (2, 1, 30))

    def test_even_schedule_has_exact_counts_and_spreads_replay(self):
        counts = {"expert": 18, "mechanics": 2, "consistency": 0}
        schedule = even_replay_schedule(counts)
        self.assertEqual(len(schedule), 20)
        self.assertEqual(
            {objective: schedule.count(objective) for objective in OBJECTIVES},
            counts,
        )
        replay_positions = [
            index for index, objective in enumerate(schedule) if objective == "mechanics"
        ]
        self.assertGreaterEqual(replay_positions[1] - replay_positions[0], 8)


class ReplayTrainingTests(unittest.TestCase):
    def test_records_both_validation_losses_and_gameplay_each_epoch(self):
        expert = masked_data("<P><G>crane<E>", 1, 6, "expert")
        mechanics = masked_data(
            "<M><S>crane<G>slate<F>00000<E>",
            13,
            18,
            "mechanics",
        )
        ratios = {"expert": 1.0, "mechanics": 0.0, "consistency": 0.0}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            root = Path(directory)
            checkpoint_path = root / "initial.pt"
            model = WordleGPT(vocab_size=VOCABULARY_SIZE)
            torch.save(
                {
                    "vocabulary_size": VOCABULARY_SIZE,
                    "model_state_dict": model.state_dict(),
                },
                checkpoint_path,
            )
            best_path, records = train_with_replay(
                {"expert": expert},
                {"expert": expert, "mechanics": mechanics},
                root / "run",
                initialization_checkpoint=checkpoint_path,
                ratios=ratios,
                secrets=["crane"],
                allowed_words=["crane"],
                batch_size=1,
                eval_batch_size=1,
                learning_rate=0.0,
                patience=1,
                max_epochs=2,
                device="cpu",
            )
            metrics = [
                json.loads(line)
                for line in (root / "run" / "metrics.jsonl").read_text().splitlines()
            ]
            best = torch.load(best_path, weights_only=True)

        self.assertEqual(len(records), 2)
        self.assertGreater(records[0].mechanics_validation_loss, 0.0)
        self.assertEqual(records[0].games, 1)
        self.assertIn("expert_validation_loss", metrics[1])
        self.assertIsNone(records[0].gradient_norm)
        self.assertGreater(records[1].gradient_norm, 0.0)
        self.assertEqual(set(records[1].gradient_norms), {"expert"})
        self.assertIn("gradient_norm", metrics[1])
        self.assertIn("gradient_norms", metrics[1])
        self.assertIn("mechanics_validation_loss", metrics[1])
        self.assertEqual(metrics[1]["batch_counts"]["expert"], 1)
        self.assertEqual(best["epoch"], 0)
        self.assertIn("train_losses", best)
        self.assertIn("gradient_norm", best)


if __name__ == "__main__":
    unittest.main()
