import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from dataset import build_dataset, iter_dataset_states, state_fingerprint
from trajectories import STRATEGIES, validate_trajectory


class DatasetTests(unittest.TestCase):
    WORDS = (
        "arise",
        "slate",
        "crane",
        "trace",
        "crate",
        "grace",
        "grade",
        "glade",
        "blade",
        "blame",
        "flame",
        "frame",
        "apple",
        "ample",
        "angle",
    )

    def test_balanced_nested_prefixes_and_fixed_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = build_dataset(
                directory,
                self.WORDS,
                total_states=80,
                prefixes=(20, 40, 80),
                seed=19,
            )
            all_states = list(iter_dataset_states(directory))
            self.assertEqual(len(all_states), 80)
            for size in (20, 40, 80):
                prefix = list(iter_dataset_states(directory, limit=size))
                expected_per_strategy = size // len(STRATEGIES)
                self.assertEqual(
                    Counter(state["strategy"] for state in prefix),
                    Counter({strategy: expected_per_strategy for strategy in STRATEGIES}),
                )
                stats = manifest["nested_datasets"][str(size)]
                unique = len({state["state_fingerprint"] for state in prefix})
                self.assertEqual(stats["total_states"], size)
                self.assertEqual(stats["unique_states"], unique)
                self.assertEqual(stats["duplicate_states"], size - unique)
            self.assertEqual(all_states[:20], list(iter_dataset_states(directory, limit=20)))
            for split in ("train", "validation", "test"):
                expected = [state for state in all_states if state["split"] == split]
                self.assertEqual(expected, list(iter_dataset_states(directory, split=split)))

    def test_state_identity_uses_observable_history_not_answer(self):
        history = [{"guess": "arise", "feedback": "XYGXX"}]
        self.assertEqual(state_fingerprint(history), state_fingerprint(list(history)))
        self.assertNotEqual(state_fingerprint(history), state_fingerprint([]))

    def test_full_trajectories_are_saved_and_do_not_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            build_dataset(
                directory,
                self.WORDS,
                total_states=40,
                prefixes=(40,),
                seed=27,
            )
            with gzip.open(
                Path(directory, "trajectories.jsonl.gz"), "rt", encoding="utf-8"
            ) as trajectory_file:
                trajectories = [json.loads(line) for line in trajectory_file]
            states = list(iter_dataset_states(directory))
            split_by_trajectory = {}
            for state in states:
                trajectory_id = state["trajectory_id"]
                previous = split_by_trajectory.setdefault(trajectory_id, state["split"])
                self.assertEqual(previous, state["split"])
                self.assertEqual(
                    state["state_fingerprint"], state_fingerprint(state["history"])
                )
                self.assertIn(state["answer"], state["possible_answers"])
            for trajectory in trajectories:
                validate_trajectory(trajectory, self.WORDS)

    def test_same_seed_produces_identical_records(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory, "first")
            second = Path(directory, "second")
            first_manifest = build_dataset(
                first, self.WORDS, total_states=40, prefixes=(40,), seed=8
            )
            second_manifest = build_dataset(
                second, self.WORDS, total_states=40, prefixes=(40,), seed=8
            )
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(
                list(iter_dataset_states(first)), list(iter_dataset_states(second))
            )


if __name__ == "__main__":
    unittest.main()
