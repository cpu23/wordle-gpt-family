import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from train_epochs import train_to_epochs
from train_nested import TokenizedState, create_split_data


class EpochTrainingTests(unittest.TestCase):
    def test_saves_metrics_and_resumes_to_total_epoch_target(self):
        data = create_split_data(
            [
                TokenizedState(
                    tokens=[29, 2, 17, 0, 13, 4, 30, 26, 26, 28, 27, 26, 31],
                    strategy="clever",
                    current_guess_target_start=0,
                )
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with redirect_stdout(io.StringIO()):
                first_records = train_to_epochs(
                    data,
                    data,
                    output_dir,
                    target_epochs=1.0,
                    save_epochs=(0.0, 1.0),
                    batch_size=1,
                    eval_batch_size=1,
                    device="cpu",
                )
            latest = json.loads((output_dir / "latest.json").read_text())
            checkpoint_path = Path(latest["checkpoint"])
            checkpoint = torch.load(checkpoint_path, weights_only=True)
            with redirect_stdout(io.StringIO()):
                resumed_records = train_to_epochs(
                    data,
                    data,
                    output_dir,
                    target_epochs=2.0,
                    save_epochs=(2.0,),
                    batch_size=1,
                    eval_batch_size=1,
                    device="cpu",
                    resume=checkpoint_path,
                )
            metric_lines = (output_dir / "metrics.jsonl").read_text().splitlines()

        self.assertEqual([record.step for record in first_records], [0, 1])
        self.assertEqual(checkpoint["tokens_seen"], 12)
        self.assertEqual(resumed_records[-1].step, 2)
        self.assertEqual(resumed_records[-1].tokens_seen, 24)
        self.assertEqual(resumed_records[-1].epochs, 2.0)
        self.assertEqual(len(metric_lines), 3)


if __name__ == "__main__":
    unittest.main()
