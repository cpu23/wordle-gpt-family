import gzip
import json
import random
import tempfile
import unittest
from pathlib import Path

from trajectories import (
    STRATEGIES,
    generate_trajectories,
    generate_trajectory,
    save_trajectories,
    validate_trajectory,
)


class TrajectoryTests(unittest.TestCase):
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

    def test_all_strategies_generate_valid_reachable_states(self):
        for strategy in STRATEGIES:
            with self.subTest(strategy=strategy):
                trajectory = generate_trajectory(
                    "apple",
                    self.WORDS,
                    strategy,
                    random.Random(14),
                    random_rate=0.5,
                )
                validate_trajectory(trajectory, self.WORDS)
                self.assertEqual(trajectory["strategy"], strategy)
                self.assertLessEqual(len(trajectory["turns"]), 6)
                self.assertTrue(trajectory["turns"])

    def test_simple_guesses_are_consistent_candidates(self):
        trajectory = generate_trajectory(
            "apple", self.WORDS, "simple", random.Random(8)
        )
        for turn in trajectory["turns"]:
            self.assertIn(turn["guess"], turn["remaining_answers"] if turn["feedback"] == "GGGGG" else self.WORDS)
            self.assertEqual(turn["policy"], "simple")

    def test_random_uses_valid_dictionary_guesses_without_repeats(self):
        trajectory = generate_trajectory(
            "apple", self.WORDS, "random", random.Random(2), max_turns=6
        )
        guesses = [turn["guess"] for turn in trajectory["turns"]]
        self.assertEqual(len(guesses), len(set(guesses)))
        self.assertTrue(set(guesses) <= set(self.WORDS))
        self.assertTrue(all(turn["policy"] == "random" for turn in trajectory["turns"]))

    def test_partly_random_can_force_either_policy(self):
        random_trajectory = generate_trajectory(
            "apple",
            self.WORDS,
            "partly-random",
            random.Random(3),
            random_rate=1.0,
        )
        clever_trajectory = generate_trajectory(
            "apple",
            self.WORDS,
            "partly-random",
            random.Random(3),
            random_rate=0.0,
        )
        self.assertTrue(all(turn["policy"] == "random" for turn in random_trajectory["turns"]))
        self.assertTrue(all(turn["policy"] == "clever" for turn in clever_trajectory["turns"]))

    def test_generation_is_reproducible_and_balances_strategies(self):
        first = list(generate_trajectories(self.WORDS, 9, seed=51))
        second = list(generate_trajectories(self.WORDS, 9, seed=51))
        self.assertEqual(first, second)
        self.assertEqual(
            [trajectory["strategy"] for trajectory in first],
            list(STRATEGIES) * 2 + [STRATEGIES[0]],
        )
        self.assertEqual([trajectory["trajectory_id"] for trajectory in first], list(range(9)))

    def test_save_streams_jsonl_metadata_and_trajectories(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory, "games.jsonl.gz")
            trajectories = generate_trajectories(self.WORDS, 4, seed=22)
            games, states = save_trajectories(
                output,
                trajectories,
                self.WORDS,
                seed=22,
                strategies=STRATEGIES,
                random_rate=0.35,
            )
            with gzip.open(output, "rt", encoding="utf-8") as saved_file:
                records = [json.loads(line) for line in saved_file]
            self.assertEqual(games, 4)
            self.assertEqual(states, sum(len(record["turns"]) for record in records[1:]))
            self.assertEqual(records[0]["record_type"], "metadata")
            self.assertEqual(records[0]["word_count"], len(self.WORDS))
            self.assertEqual(len(records[0]["word_list_sha256"]), 64)
            self.assertTrue(all(record["record_type"] == "trajectory" for record in records[1:]))
            for record in records[1:]:
                validate_trajectory(record, self.WORDS)

    def test_validator_rejects_corrupt_feedback(self):
        trajectory = generate_trajectory(
            "apple", self.WORDS, "simple", random.Random(1)
        )
        trajectory["turns"][0]["feedback"] = "XXXXX"
        with self.assertRaisesRegex(ValueError, "feedback"):
            validate_trajectory(trajectory, self.WORDS)


if __name__ == "__main__":
    unittest.main()
