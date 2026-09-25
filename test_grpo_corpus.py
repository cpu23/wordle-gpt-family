import gzip
import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from grpo_corpus import ContinuationState, build_grpo_corpus, load_corpus
from wordle import load_words, score_guess


class GrpoCorpusTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = load_words()

    def _inputs(self, directory):
        directory = Path(directory)
        train_secrets = list(self.words[:2])
        validation_secrets = [self.words[2]]
        test_secrets = [self.words[3]]
        panel_secrets = set(train_secrets + validation_secrets + test_secrets)
        guesses = [word for word in self.words if word not in panel_secrets][10:30]
        mode_path = directory / "mode.json"
        mode_path.write_text(
            json.dumps(
                {
                    "mode": "development",
                    "runs": [
                        {
                            "run": 1,
                            "train": train_secrets,
                            "validation": validation_secrets,
                            "test": test_secrets,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        words_path = directory / "words.txt"
        words_path.write_text("\n".join(self.words) + "\n", encoding="utf-8")
        source_path = directory / "examples.jsonl.gz"
        rows = []
        source_index = 0
        for split_secrets, count in (
            (train_secrets, 8),
            (validation_secrets, 5),
            (test_secrets, 4),
        ):
            for secret in split_secrets:
                for row_index in range(count):
                    depth = 1 + row_index % 5
                    history = []
                    for turn in range(depth):
                        guess = guesses[(row_index + turn) % len(guesses)]
                        history.append(
                            {
                                "guess": guess,
                                "feedback": score_guess(secret, guess),
                            }
                        )
                    rows.append(
                        {
                            "state_index": source_index,
                            "source_secret": secret,
                            "source_behavior": ("random", "simple", "entropy")[
                                row_index % 3
                            ],
                            "history": history,
                            # These SFT-only columns must not affect corpus selection.
                            "sampling_weight": "not a number",
                            "desired_guess": None,
                        }
                    )
                    source_index += 1
        train_secret = train_secrets[0]
        rows.extend(
            [
                {
                    "state_index": source_index,
                    "source_secret": train_secret,
                    "source_behavior": "random",
                    "history": [],
                },
                {
                    "state_index": source_index + 1,
                    "source_secret": train_secret,
                    "source_behavior": "random",
                    "history": [
                        {
                            "guess": guesses[index % len(guesses)],
                            "feedback": score_guess(
                                train_secret, guesses[index % len(guesses)]
                            ),
                        }
                        for index in range(6)
                    ],
                },
                {
                    "state_index": source_index + 2,
                    "source_secret": train_secret,
                    "source_behavior": "random",
                    "history": [
                        {"guess": train_secret, "feedback": "GGGGG"}
                    ],
                },
                {
                    "state_index": source_index + 3,
                    "source_secret": self.words[4],
                    "source_behavior": "random",
                    "history": [
                        {
                            "guess": guesses[0],
                            "feedback": score_guess(self.words[4], guesses[0]),
                        }
                    ],
                },
            ]
        )
        with gzip.open(source_path, "wt", encoding="utf-8") as source_file:
            for row in rows:
                source_file.write(json.dumps(row) + "\n")
        return source_path, mode_path, words_path, train_secrets, validation_secrets, test_secrets

    def test_builds_reproducible_split_safe_reachable_samples(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, mode, words, train_secrets, validation_secrets, test_secrets = (
                self._inputs(root)
            )
            common = {
                "source": source,
                "mode": mode,
                "words": words,
                "size": 7,
                "validation_size": 3,
                "seed": 91,
            }
            first_manifest = build_grpo_corpus(root / "first", **common)
            second_manifest = build_grpo_corpus(root / "second", **common)
            train = load_corpus(root / "first" / "train.jsonl")
            validation = load_corpus(root / "first" / "validation.jsonl")

            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                (root / "first" / "train.jsonl").read_bytes(),
                (root / "second" / "train.jsonl").read_bytes(),
            )
            self.assertEqual(
                (root / "first" / "validation.jsonl").read_bytes(),
                (root / "second" / "validation.jsonl").read_bytes(),
            )
            self.assertEqual(len(train), 7)
            self.assertEqual(len(validation), 3)
            self.assertTrue({state.secret for state in train} <= set(train_secrets))
            self.assertTrue(
                {state.secret for state in validation} <= set(validation_secrets)
            )
            self.assertFalse(
                {state.secret for state in train} & set(validation_secrets + test_secrets)
            )
            self.assertFalse(
                {state.secret for state in validation} & set(train_secrets + test_secrets)
            )
            all_states = train + validation
            self.assertEqual(
                len({state.source_index for state in all_states}), len(all_states)
            )
            for state in all_states:
                self.assertGreaterEqual(len(state.history), 1)
                self.assertLessEqual(len(state.history), 5)
                self.assertNotIn((state.secret, "GGGGG"), state.history)
                for guess, feedback in state.history:
                    self.assertEqual(score_guess(state.secret, guess), feedback)
            self.assertEqual(
                first_manifest["sampling"]["eligible_rows_by_split"],
                {"train": 16, "validation": 5},
            )
            self.assertEqual(
                first_manifest["sampling"]["excluded_rows"],
                {"depth_over_five": 1, "empty_history": 1, "terminal": 1},
            )
            self.assertEqual(
                first_manifest["held_out_exclusion"]["test_secrets_included"],
                False,
            )
            self.assertEqual(
                first_manifest["selected_counts"]["train"]["count"], 7
            )
            self.assertEqual(
                first_manifest["selected_counts"]["validation"]["count"], 3
            )
            for row_path in (
                root / "first" / "train.jsonl",
                root / "first" / "validation.jsonl",
            ):
                for line in row_path.read_text(encoding="utf-8").splitlines():
                    self.assertEqual(
                        set(json.loads(line)), {"secret", "history", "source_index"}
                    )

    def test_insufficient_eligible_rows_fails_without_writing_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source, mode, words, *_ = self._inputs(root)
            with gzip.open(source, "wt", encoding="utf-8") as source_file:
                secret = self.words[0]
                source_file.write(
                    json.dumps(
                        {
                            "state_index": 0,
                            "source_secret": secret,
                            "source_behavior": "random",
                            "history": [
                                {
                                    "guess": self.words[20],
                                    "feedback": score_guess(secret, self.words[20]),
                                }
                            ],
                        }
                    )
                    + "\n"
                )
            output_dir = root / "insufficient"
            with self.assertRaisesRegex(
                ValueError, "not enough eligible train rows: requested 2, found 1"
            ):
                build_grpo_corpus(
                    output_dir,
                    source=source,
                    mode=mode,
                    words=words,
                    size=2,
                    validation_size=1,
                    seed=5,
                )
            self.assertFalse(output_dir.exists())

    def test_state_is_frozen(self):
        state = ContinuationState("about", (("adieu", "XXXXX"),), 4)
        with self.assertRaises(FrozenInstanceError):
            state.secret = "other"


if __name__ == "__main__":
    unittest.main()
