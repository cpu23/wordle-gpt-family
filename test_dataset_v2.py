import gzip
import json
import tempfile
import unittest
from pathlib import Path

from dataset_v2 import build_v2_dataset
from tokenizer import FEEDBACK_SYMBOLS, LETTERS, TOKEN_TO_ID
from train import IGNORE_INDEX
from train_v2 import load_v2_split
from tokenizer_v2 import (
    MECHANICS_TOKEN,
    POLICY_TOKEN,
    SECRET_TOKEN,
    VOCABULARY_SIZE,
    decode,
    encode,
)
from wordle import score_guess


class DatasetV2Tests(unittest.TestCase):
    def test_builds_mechanics_and_expert_examples_with_only_observable_targets(self):
        words = ("crane", "slate", "drink", "colon")
        strategies = ("clever", "simple", "random", "partly-random")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_dir = root / "source"
            output_dir = root / "v2"
            source_dir.mkdir()
            with gzip.open(
                source_dir / "states.jsonl.gz", "wt", encoding="utf-8"
            ) as output:
                for state_index, strategy in enumerate(strategies):
                    answer = words[state_index]
                    guess = words[(state_index + 1) % len(words)]
                    state = {
                        "state_index": state_index,
                        "split": "validation" if state_index == 1 else "train",
                        "strategy": strategy,
                        "action_policy": (
                            strategy if strategy != "partly-random" else "random"
                        ),
                        "answer": answer,
                        "action": guess,
                        "feedback": score_guess(answer, guess),
                        "history": [],
                        "possible_answers": list(words),
                    }
                    output.write(json.dumps(state) + "\n")

            manifest = build_v2_dataset(source_dir, output_dir, words)
            with gzip.open(
                output_dir / "examples.jsonl.gz", "rt", encoding="utf-8"
            ) as source:
                examples = [json.loads(line) for line in source]
            validation_data = load_v2_split(output_dir, "validation")

        self.assertEqual(manifest["source_states"], 4)
        self.assertEqual(manifest["examples"], 8)
        self.assertEqual(set(manifest["source_strategy_counts"]), set(strategies))
        mechanics = [example for example in examples if example["example_type"] == "mechanics"]
        self.assertEqual(manifest["vocabulary_size"], VOCABULARY_SIZE)
        experts = [example for example in examples if example["example_type"] == "expert"]
        self.assertEqual(len(mechanics), 4)
        self.assertEqual(len(experts), 4)
        self.assertEqual(validation_data.supervised_token_count, 10)
        self.assertEqual(
            (validation_data.targets != IGNORE_INDEX).sum(dim=1).tolist(),
            [5, 5],
        )
        self.assertTrue(mechanics[0]["text"].startswith(MECHANICS_TOKEN + SECRET_TOKEN))
        self.assertTrue(experts[0]["text"].startswith(POLICY_TOKEN))
        self.assertEqual(decode(mechanics[0]["token_ids"]), mechanics[0]["text"])

        feedback_ids = {TOKEN_TO_ID[symbol] for symbol in FEEDBACK_SYMBOLS}
        letter_ids = {TOKEN_TO_ID[letter] for letter in LETTERS}
        for example in mechanics:
            start, stop = example["loss_ranges"][0]
            targets = example["token_ids"][1:]
            self.assertEqual(stop - start, 5)
            self.assertTrue(set(targets[start:stop]) <= feedback_ids)
        for example in experts:
            start, stop = example["loss_ranges"][0]
            targets = example["token_ids"][1:]
            self.assertEqual(stop - start, 5)
            self.assertTrue(set(targets[start:stop]) <= letter_ids)
            self.assertEqual(example["expert_policy"], "clever")

    def test_training_sampling_weights_do_not_change_validation_distribution(self):
        text = f"{POLICY_TOKEN}<G>crane<E>"
        record = {
            "example_type": "expert",
            "text": text,
            "token_ids": encode(text),
            "loss_ranges": [[1, 6]],
            "sampling_weight": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as output:
                output.write(json.dumps({**record, "split": "train"}) + "\n")
                output.write(json.dumps({**record, "split": "validation"}) + "\n")
            train_data = load_v2_split(directory, "train", example_type="expert")
            validation_data = load_v2_split(
                directory, "validation", example_type="expert"
            )
        self.assertEqual(len(train_data.inputs), 3)
        self.assertEqual(len(validation_data.inputs), 1)


if __name__ == "__main__":
    unittest.main()
