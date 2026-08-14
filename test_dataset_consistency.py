import gzip
import json
import tempfile
import unittest
from pathlib import Path

from dataset_consistency import build_candidate_consistency_dataset
from tokenizer_v2 import decode
from train import IGNORE_INDEX
from train_v2 import load_v2_split


class CandidateConsistencyDatasetTests(unittest.TestCase):
    def test_builds_balanced_binary_targets_and_masks_prompts(self):
        words = ("crane", "slate")
        states = [
            {
                "state_index": 0,
                "split": "validation",
                "history": [],
                "possible_answers": list(words),
            },
            {
                "state_index": 1,
                "split": "validation",
                "history": [{"guess": "slate", "feedback": "XXXXX"}],
                "possible_answers": ["crane"],
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            output_dir = root / "output"
            source_dir.mkdir()
            with gzip.open(source_dir / "states.jsonl.gz", "wt", encoding="utf-8") as output:
                for state in states:
                    output.write(json.dumps(state) + "\n")

            manifest = build_candidate_consistency_dataset(source_dir, output_dir, words)
            with gzip.open(output_dir / "examples.jsonl.gz", "rt", encoding="utf-8") as source:
                examples = [json.loads(line) for line in source]
            data = load_v2_split(output_dir, "validation", example_type="consistency")

        self.assertEqual(manifest["eligible_states"], 1)
        self.assertEqual(manifest["label_counts"], {"0": 1, "2": 1})
        self.assertEqual([example["consistent"] for example in examples], [True, False])
        self.assertTrue(examples[0]["text"].startswith("<M><S>crane<G>slate<F>00000"))
        self.assertEqual(decode(examples[0]["token_ids"]), examples[0]["text"])
        self.assertEqual(data.supervised_token_count, 2)
        self.assertEqual((data.targets != IGNORE_INDEX).sum(dim=1).tolist(), [1, 1])


if __name__ == "__main__":
    unittest.main()
