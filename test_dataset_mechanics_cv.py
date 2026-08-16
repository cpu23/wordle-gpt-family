import gzip
import json
import tempfile
import unittest
from pathlib import Path

from dataset_mechanics_cv import build_mechanics_cv_pool
from tokenizer_v2 import decode, encode
from wordle import load_words, score_guess


class MechanicsCVPoolTests(unittest.TestCase):
    def test_builds_unique_filterable_secret_guess_pairs(self):
        words = load_words()
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_mechanics_cv_pool(
                directory,
                words[:20],
                total_examples=100,
                seed=3,
            )
            with gzip.open(
                Path(directory) / "examples.jsonl.gz", "rt", encoding="utf-8"
            ) as source:
                examples = [json.loads(line) for line in source]
        self.assertEqual(manifest["source_secrets"], 20)
        pairs = {
            (item["source_secret"], decode(item["token_ids"][8:13]))
            for item in examples
        }
        self.assertEqual(len(pairs), 100)
        for item in examples:
            secret = item["source_secret"]
            guess = decode(item["token_ids"][8:13])
            self.assertEqual(item["token_ids"], encode(item["text"]))
            self.assertIn(secret, words[:20])
            self.assertEqual(len(score_guess(secret, guess)), 5)


if __name__ == "__main__":
    unittest.main()
