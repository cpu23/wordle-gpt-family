import gzip
import json
import tempfile
import unittest
from pathlib import Path

from dataset_expert import (
    build_diverse_expert_dataset,
    choose_entropy_guess,
    choose_poor_legal_guess,
    validate_diverse_expert_dataset,
)
from wordle import choose_informative_guess, filter_answers, load_words
class DiverseExpertDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = load_words()

    def test_entropy_and_poor_policies_return_legal_guesses(self):
        possible = tuple(self.words[:12])
        entropy = choose_entropy_guess(possible, self.words[:40])
        import random

        poor = choose_poor_legal_guess(
            possible,
            self.words[:40],
            {entropy},
            random.Random(3),
            candidate_count=10,
        )
        self.assertIn(entropy, self.words[:40])
        self.assertIn(poor, self.words[:40])
        self.assertNotEqual(poor, entropy)

    def test_builds_nested_unique_states_with_clever_targets(self):
        source_splits = {
            "train": self.words[:30],
            "validation": self.words[30:40],
        }
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_diverse_expert_dataset(
                directory,
                self.words,
                source_splits,
                total_states=40,
                prefixes=(20, 40),
                seed=7,
            )
            with gzip.open(
                Path(directory) / "examples.jsonl.gz", "rt", encoding="utf-8"
            ) as source:
                examples = [json.loads(line) for line in source]
            validation = validate_diverse_expert_dataset(
                directory,
                self.words,
                source_splits,
            )

        self.assertEqual(len(examples), 40)
        self.assertEqual(len({example["state_fingerprint"] for example in examples}), 40)
        self.assertEqual(manifest["nested_datasets"]["20"]["unique_states"], 20)
        self.assertEqual(manifest["unique_states"], 40)
        self.assertEqual(validation["unique_histories"], 40)
        self.assertTrue(
            {example["source_secret"] for example in examples}
            <= set(source_splits["train"] + source_splits["validation"])
        )
        for example in examples:
            possible = tuple(self.words)
            for turn in example["history"]:
                possible = tuple(
                    filter_answers(possible, turn["guess"], turn["feedback"])
                )
            self.assertEqual(
                example["desired_guess"],
                choose_informative_guess(possible, self.words),
            )


if __name__ == "__main__":
    unittest.main()
