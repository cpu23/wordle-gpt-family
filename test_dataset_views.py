import gzip
import hashlib
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dataset import build_dataset
from dataset_views import (
    RAW_FILENAMES,
    SPLIT_COUNTS,
    build_comparison_views,
    create_secret_splits,
)
from tokenizer import TOKENS, VOCABULARY_SIZE, decode, encode, serialize_trajectory
from wordle import load_words


class TokenizerTests(unittest.TestCase):
    def test_requested_example_round_trips_with_exact_ids(self):
        text = "<G>crane<F>01200<E>"
        token_ids = encode(text)
        self.assertEqual(
            token_ids,
            [29, 2, 17, 0, 13, 4, 30, 26, 27, 28, 26, 26, 31],
        )
        self.assertEqual(decode(token_ids), text)

    def test_trajectory_serialization(self):
        turns = [
            {"guess": "slate", "feedback": "XXGYX"},
            {"guess": "crony", "feedback": "XYXXX"},
            {"guess": "drink", "feedback": "GGGGG"},
        ]
        text = serialize_trajectory(turns)
        self.assertEqual(
            text, "<G>slate<F>00210<G>crony<F>01000<G>drink<F>22222<E>"
        )
        self.assertEqual(decode(encode(text)), text)
        self.assertEqual(len(encode(text)), 37)

    def test_vocabulary_is_dense_and_has_expected_size(self):
        self.assertEqual(VOCABULARY_SIZE, 32)
        self.assertEqual(len(TOKENS), len(set(TOKENS)))
        self.assertEqual(sorted(encode("abcdefghijklmnopqrstuvwxyz012<G><F><E>")), list(range(32)))


class ComparisonViewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = load_words()

    def test_secret_split_is_exact_disjoint_and_complete(self):
        splits, assignment = create_secret_splits(self.words, 20260813)
        self.assertEqual({name: len(words) for name, words in splits.items()}, SPLIT_COUNTS)
        self.assertEqual(len(assignment), 719)
        self.assertEqual(set().union(*(set(words) for words in splits.values())), set(self.words))
        self.assertFalse(set(splits["train"]) & set(splits["validation"]))
        self.assertFalse(set(splits["train"]) & set(splits["test"]))
        self.assertFalse(set(splits["validation"]) & set(splits["test"]))

    def test_views_preserve_raw_files_and_aggregate_duplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            build_dataset(
                root,
                self.words,
                total_states=80,
                prefixes=(80,),
                seed=31,
            )
            before = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in RAW_FILENAMES
            }
            manifest = build_comparison_views(root, self.words, seed=31)
            after = {
                name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                for name in RAW_FILENAMES
            }
            self.assertEqual(before, after)
            self.assertEqual(manifest["raw_pool"]["total_states"], 80)
            self.assertEqual(
                manifest["deduplicated_pool"]["unique_states"]
                + manifest["deduplicated_pool"]["duplicate_states_removed"],
                80,
            )

            with gzip.open(
                root / "deduplicated-states.jsonl.gz", "rt", encoding="utf-8"
            ) as source:
                deduplicated = [json.loads(line) for line in source]
            self.assertEqual(
                sum(record["number_of_occurrences"] for record in deduplicated), 80
            )
            self.assertTrue(all(record["strategies"] for record in deduplicated))
            self.assertTrue(all(record["action_policies"] for record in deduplicated))

            with gzip.open(
                root / "tokenized-trajectories.jsonl.gz", "rt", encoding="utf-8"
            ) as source:
                tokenized = [json.loads(line) for line in source]
            split_by_answer = {}
            for record in tokenized:
                previous = split_by_answer.setdefault(
                    record["answer"], record["secret_split"]
                )
                self.assertEqual(previous, record["secret_split"])
                self.assertEqual(decode(record["token_ids"]), record["text"])
            counts = Counter(record["secret_split"] for record in tokenized)
            reported = manifest["tokenized_trajectories"]["sequence_statistics"]
            self.assertEqual(
                sum(counts.values()), reported["all"]["trajectory_count"]
            )


if __name__ == "__main__":
    unittest.main()
