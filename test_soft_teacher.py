import gzip
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from soft_teacher import (
    TeacherDataset,
    build_teacher_dataset,
    distribution_statistics,
    exhaustive_scores,
    feedback_matrix,
    select_candidates,
    teacher_probabilities,
)
from wordle import expected_survivors, top_informative_guesses


class SoftTeacherTests(unittest.TestCase):
    def test_duplicate_letter_costs_and_tie_order_match_exhaustive_solver(self):
        words = ("allee", "apple", "apply", "pleat", "eerie", "level")
        remaining = (0, 1, 4)
        numerators, ranks = exhaustive_scores(feedback_matrix(words), remaining)
        answers = tuple(words[index] for index in remaining)
        np.testing.assert_allclose(numerators / len(remaining),
                                   [expected_survivors(answers, guess) for guess in words])
        expected = top_informative_guesses(answers, words, len(words))
        self.assertEqual([words[index] for index in np.argsort(ranks)],
                         [guess for guess, _ in expected])
        numerators, ranks = exhaustive_scores(feedback_matrix(words), (4,))
        self.assertTrue(np.all(numerators == 1))
        self.assertEqual(int(ranks[4]), 1)

    def test_mandatory_candidates_deterministic_quotas_and_refill(self):
        ranks = np.arange(1, 720, dtype=np.uint16)
        kwargs = dict(ranks=ranks, remaining=(400, 650), stored_top=(0, 1, 2, 3, 4, 5, 80, 700),
                      desired_guess=650, seed=19, observable_key=b"observable prompt")
        first = select_candidates(**kwargs)
        second = select_candidates(**kwargs)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(set(map(int, first))), 128)
        self.assertTrue(set(range(32)).issubset(first))
        self.assertTrue({80, 650, 700}.issubset(first))
        self.assertTrue(np.all((first[32:64] >= 32) & (first[32:64] < 256)))
        self.assertFalse(np.array_equal(first, select_candidates(**{**kwargs, "seed": 20})))
        # A small legal vocabulary refills deterministically without repetitions.
        small = select_candidates(np.arange(1, 129), (127,), (0, 1), 127,
                                  seed=19, observable_key=b"observable prompt")
        self.assertEqual(set(map(int, small)), set(range(128)))

    def test_temperature_reweighting_ties_and_singleton_override(self):
        costs = torch.tensor([[1., 1., 2., 4.], [1., 1., 1., 1.]], requires_grad=True)
        counts = torch.tensor([3, 1])
        membership = torch.tensor([[True, False, True, False], [False, False, True, False]])
        for temperature in (0.25, 0.5, 1.0):
            probabilities = teacher_probabilities(costs, counts, membership, temperature)
            reference = costs[0].detach().pow(-1 / temperature)
            reference /= reference.sum()
            torch.testing.assert_close(probabilities[0], reference)
            self.assertEqual(probabilities[0, 0], probabilities[0, 1])
            torch.testing.assert_close(probabilities[1], torch.tensor([0., 0., 1., 0.]))
            self.assertFalse(probabilities.requires_grad)
        for temperature in (0, -1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                teacher_probabilities(costs, counts, membership, temperature)

    def test_full_builder_source_independence_reload_and_split_statistics(self):
        # Two possible source secrets produce the exact same observable opening.
        words = tuple("aaa" + chr(97 + index // 26) + chr(97 + index % 26) for index in range(130))
        numerators, ranks = exhaustive_scores(feedback_matrix(words), tuple(range(len(words))))
        top_ids = np.argsort(ranks)[:8]
        rows = [{"state_index": index + 10, "source_secret": words[index], "history": [],
                 "possible_answer_count": len(words), "desired_guess": words[-1],
                 "top_guesses": [{"guess": words[word_id]} for word_id in top_ids],
                 "sampling_weight": 10_000}
                for index in range(2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            words_path = root / "words.txt"
            words_path.write_text("\n".join(words) + "\n")
            source = root / "examples.jsonl.gz"
            with gzip.open(source, "wt") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            manifest = build_teacher_dataset(root / "teacher", source=source, words=words_path,
                                             seed=19, workers=1, expected_count=2)
            dataset = TeacherDataset(root / "teacher")
            self.assertEqual(len(dataset), 2)
            self.assertEqual(manifest["source"]["rows"], 2)
            self.assertFalse(manifest["source"]["subset"])
            self.assertEqual(dataset.words, words)
            self.assertIsInstance(dataset.costs, np.memmap)
            for name in ("prompts", "lengths", "candidate_ids", "costs", "ranks", "remaining_counts", "is_remaining"):
                np.testing.assert_array_equal(getattr(dataset, name)[0], getattr(dataset, name)[1])
            np.testing.assert_array_equal(dataset.source_ids, [0, 1])
            np.testing.assert_array_equal(dataset.state_ids, [10, 11])
            np.testing.assert_array_equal(dataset.weights, [10_000, 10_000])
            selected = dataset.candidate_ids[0]
            self.assertIn(129, selected)
            self.assertTrue(set(top_ids).issubset(selected))
            np.testing.assert_allclose(dataset.costs[0], numerators[selected] / len(words))
            np.testing.assert_array_equal(dataset.ranks[0], ranks[selected])
            mode = root / "mode.json"
            mode.write_text(json.dumps({"mode": "development", "model_seeds": [0], "runs": [{
                "run": 1, "train": [words[0], *words[73:]],
                "validation": list(words[1:73]), "test": []}]}))
            report = distribution_statistics(dataset, mode=mode, chunk_size=1)
            for temperature in ("0.25", "0.5", "1.0"):
                summaries = report["temperatures"][temperature]
                self.assertEqual(summaries["all"]["all"]["count"], 2)
                self.assertEqual(summaries["all"]["singleton"]["count"], 0)
                self.assertEqual(summaries["source_train"]["all"], summaries["source_validation"]["all"])
                targets = teacher_probabilities(torch.tensor(dataset.costs[:]), torch.tensor(dataset.remaining_counts[:]),
                                               torch.tensor(dataset.is_remaining[:]), float(temperature))
                entropy = -(targets * targets.log()).sum(dim=1).mean().item()
                self.assertAlmostEqual(summaries["all"]["all"]["entropy"]["mean"], entropy)
            # Full-count requirements cannot silently accept a subset.
            with self.assertRaisesRegex(ValueError, "expected the full 3"):
                build_teacher_dataset(root / "incomplete", source=source, words=words_path,
                                      seed=19, workers=1, expected_count=3)


if __name__ == "__main__":
    unittest.main()
