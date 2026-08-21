import gzip
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from dataset_expert import (
    answer_set_fingerprint,
    build_diverse_expert_dataset,
    choose_entropy_guess,
    choose_poor_legal_guess,
    validate_diverse_expert_dataset,
)
from wordle import (
    choose_informative_guess,
    filter_answers,
    load_words,
    top_informative_guesses,
)


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
            dataset_hash = hashlib.sha256(
                (Path(directory) / "examples.jsonl.gz").read_bytes()
            ).hexdigest()
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
        self.assertEqual(manifest["file_sha256"], dataset_hash)
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

    def test_answer_set_fingerprint_is_canonical(self):
        self.assertEqual(
            answer_set_fingerprint(["able", "bake", "cane"]),
            answer_set_fingerprint(["cane", "able", "bake"]),
        )
        self.assertNotEqual(
            answer_set_fingerprint(["able", "bake"]),
            answer_set_fingerprint(["able", "cane"]),
        )

    def _build_small(self, directory, **kwargs):
        source_splits = {
            "train": self.words[:50],
            "validation": self.words[50:60],
        }
        manifest = build_diverse_expert_dataset(
            directory,
            self.words[:80],
            source_splits,
            total_states=60,
            prefixes=(30, 60),
            seed=7,
            **kwargs,
        )
        with gzip.open(Path(directory) / "examples.jsonl.gz", "rt", encoding="utf-8") as source:
            lines = source.read().splitlines()
        return manifest, lines

    def test_repeat_probability_one_reproduces_unbiased_stream(self):
        with tempfile.TemporaryDirectory() as unbiased:
            _, unbiased_lines = self._build_small(unbiased)
        with tempfile.TemporaryDirectory() as biased:
            manifest, biased_lines = self._build_small(
                biased,
                repeat_probability=1.0,
                bias_prefix=30,
            )
        self.assertEqual(biased_lines, unbiased_lines)
        self.assertEqual(
            manifest["answer_set_bias"],
            {"repeat_acceptance_probability": 1.0, "bias_prefix": 30},
        )

    def test_repeat_probability_preserves_prefix_and_biases_tail(self):
        with tempfile.TemporaryDirectory() as unbiased:
            unbiased_manifest, unbiased_lines = self._build_small(unbiased)
        with tempfile.TemporaryDirectory() as half:
            half_manifest, half_lines = self._build_small(
                half,
                repeat_probability=0.5,
                bias_prefix=30,
            )
        with tempfile.TemporaryDirectory() as zero:
            zero_manifest, zero_lines = self._build_small(
                zero,
                repeat_probability=0.0,
                bias_prefix=30,
            )
        self.assertEqual(half_lines[:30], unbiased_lines[:30])
        self.assertNotEqual(half_lines, unbiased_lines)
        self.assertEqual(zero_lines[:30], unbiased_lines[:30])

        def tail_answer_sets(lines: list[str]) -> list[str]:
            answer_sets = []
            for line in lines[30:]:
                example = json.loads(line)
                possible = tuple(self.words[:80])
                for turn in example["history"]:
                    possible = tuple(
                        filter_answers(possible, turn["guess"], turn["feedback"])
                    )
                answer_sets.append(answer_set_fingerprint(possible))
            return answer_sets

        zero_tail = tail_answer_sets(zero_lines)
        self.assertEqual(len(zero_tail), 30)
        self.assertEqual(len(set(zero_tail)), 30)
        self.assertEqual(
            zero_manifest["answer_sets"]["unique_answer_sets_at_prefix"]["30"],
            zero_manifest["nested_datasets"]["30"]["unique_answer_sets"],
        )
        self.assertEqual(
            half_manifest["answer_sets"]["unique_answer_sets_at_prefix"]["30"],
            zero_manifest["answer_sets"]["unique_answer_sets_at_prefix"]["30"],
        )
        self.assertGreater(
            half_manifest["answer_sets"]["unique_answer_sets"],
            zero_manifest["answer_sets"]["unique_answer_sets_at_prefix"]["30"],
        )

    def test_validator_rejects_stale_answer_set_manifest(self):
        source_splits = {
            "train": self.words[:50],
            "validation": self.words[50:60],
        }
        with tempfile.TemporaryDirectory() as directory:
            build_diverse_expert_dataset(
                directory,
                self.words[:80],
                source_splits,
                total_states=40,
                prefixes=(40,),
                seed=7,
            )
            manifest_path = Path(directory) / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["answer_sets"]["unique_answer_sets"] += 1
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                validate_diverse_expert_dataset(
                    directory,
                    self.words[:80],
                    source_splits,
                )

    def test_top_guesses_field_matches_solver_ranking(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, lines = self._build_small(directory, top_guesses=4)
            self.assertEqual(manifest["top_guesses"]["k"], 4)
            for line in lines[:10]:
                example = json.loads(line)
                possible = tuple(self.words[:80])
                for turn in example["history"]:
                    possible = tuple(
                        filter_answers(possible, turn["guess"], turn["feedback"])
                    )
                top = top_informative_guesses(possible, self.words[:80], 4)
                self.assertEqual(example["desired_guess"], top[0][0])
                self.assertEqual(
                    example["top_guesses"],
                    [
                        {"guess": guess, "expected_survivors": score}
                        for guess, score in top
                    ],
                )
            self.assertEqual(
                validate_diverse_expert_dataset(
                    directory,
                    self.words[:80],
                    {"train": self.words[:50], "validation": self.words[50:60]},
                )["top_guesses"]["k"],
                4,
            )

    def test_top_guesses_do_not_change_generation_stream(self):
        with tempfile.TemporaryDirectory() as plain:
            plain_manifest, plain_lines = self._build_small(plain)
        with tempfile.TemporaryDirectory() as topped:
            topped_manifest, topped_lines = self._build_small(
                topped, top_guesses=4
            )
        self.assertEqual(plain_manifest["top_guesses"], None)
        self.assertEqual(len(plain_lines), len(topped_lines))
        for plain_line, topped_line in zip(plain_lines, topped_lines):
            plain_record = json.loads(plain_line)
            topped_record = json.loads(topped_line)
            topped_record.pop("top_guesses")
            self.assertEqual(plain_record, topped_record)
        for line in plain_lines:
            self.assertNotIn("top_guesses", json.loads(line))


if __name__ == "__main__":
    unittest.main()
