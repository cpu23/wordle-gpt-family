import gzip
import json
import tempfile
import unittest
from pathlib import Path

from cross_validation import (
    EvaluationMode,
    SecretRun,
    aggregate_seed_summaries,
    combine_fold_predictions,
    compare_paired_predictions,
    create_benchmark_mode,
    materialize_expert_fold,
)
from wordle import load_words


class CrossValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.words = load_words()

    def test_benchmark_partitions_all_719_secrets_once(self):
        mode = create_benchmark_mode(self.words, split_seed=17, model_seeds=(0, 1, 2))
        self.assertEqual([len(run.test) for run in mode.runs], [144, 144, 144, 144, 143])
        self.assertEqual([len(run.train) for run in mode.runs], [503, 503, 503, 503, 504])
        self.assertTrue(all(len(run.validation) == 72 for run in mode.runs))
        held_out = [secret for run in mode.runs for secret in run.test]
        self.assertEqual(len(held_out), 719)
        self.assertEqual(set(held_out), set(self.words))

    def test_materialized_fold_excludes_held_out_sources(self):
        run = SecretRun(1, ("train",), ("valid",), ("test",))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            examples = [
                {"state_index": index, "source_secret": secret, "split": "original"}
                for index, secret in enumerate(("train", "valid", "test"))
            ]
            with gzip.open(source / "examples.jsonl.gz", "wt", encoding="utf-8") as output:
                for example in examples:
                    output.write(json.dumps(example) + "\n")
            manifest = materialize_expert_fold(source, root / "fold", run)
            with gzip.open(root / "fold" / "examples.jsonl.gz", "rt", encoding="utf-8") as input_file:
                materialized = [json.loads(line) for line in input_file]

        self.assertEqual(manifest["held_out_examples"], 0)
        self.assertEqual(manifest["omitted_test_examples"], 1)
        self.assertEqual({item["source_secret"] for item in materialized}, {"train", "valid"})
        self.assertEqual({item["split"] for item in materialized}, {"train", "validation"})

    def test_combines_folds_and_compares_secrets_pairwise(self):
        runs = tuple(SecretRun(index + 1, (), (), (f"s{index}",)) for index in range(5))
        mode = EvaluationMode("benchmark", runs, (0,))
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for index in range(5):
                path = Path(directory) / f"fold-{index}.json"
                path.write_text(
                    json.dumps(
                        {
                            "results": [
                                {
                                    "secret": f"s{index}",
                                    "guesses": ["a"] * (index + 1),
                                    "won": index != 4,
                                    "invalid_guesses": int(index == 3),
                                }
                            ]
                        }
                    ),
                    encoding="utf-8",
                )
                paths.append(path)
            model_a = combine_fold_predictions(mode, paths)

        model_b = json.loads(json.dumps(model_a))
        model_b["results"][4]["won"] = True
        model_b["results"][4]["guesses"] = ["a"] * 4
        comparison = compare_paired_predictions(model_a, model_b)
        self.assertEqual(model_a["wins"], 4)
        self.assertEqual(model_a["invalid_guesses"], 1)
        self.assertEqual(comparison["outcomes"]["a_loses_b_wins"], 1)
        self.assertEqual(comparison["per_secret"][-1]["guess_delta_b_minus_a"], -1)

    def test_aggregates_seed_variation(self):
        aggregate = aggregate_seed_summaries(
            [
                {"wins": 700, "win_rate": 0.97, "average_guesses": 3.1, "average_attempts": 3.2, "invalid_guesses": 2},
                {"wins": 704, "win_rate": 0.98, "average_guesses": 3.0, "average_attempts": 3.1, "invalid_guesses": 4},
                {"wins": 702, "win_rate": 0.975, "average_guesses": 3.05, "average_attempts": 3.15, "invalid_guesses": 3},
            ]
        )
        self.assertEqual(aggregate["wins"]["mean"], 702.0)
        self.assertEqual(aggregate["wins"]["standard_deviation"], 2.0)


if __name__ == "__main__":
    unittest.main()
