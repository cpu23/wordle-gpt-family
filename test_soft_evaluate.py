import json
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from model import WordleGPT
from soft_evaluate import _consistent_answers, _distribution_positions, _score_action, checkpoint_key, evaluate_soft_model
from tokenizer_v2 import VOCABULARY_SIZE, encode
from train_v2 import V2SplitData
from wordle import expected_survivors, load_words, score_guess


class ObservableDataset(SimpleNamespace):
    def __len__(self):
        return len(self.lengths)

    @property
    def source_ids(self):
        raise AssertionError("source secrets must not be consulted by evaluation")


class SoftEvaluationTests(unittest.TestCase):
    def test_distribution_panel_covers_sizes_despite_singleton_majority(self):
        counts = [1] * 100 + [2, 5, 6, 20, 21, 100, 101, 719]
        selected = _distribution_positions(counts, 5)
        self.assertEqual(selected, {0, 100, 102, 104, 106})
        self.assertEqual(_distribution_positions(counts, 4), {100, 102, 104, 106})
        self.assertEqual(_distribution_positions(counts, 200), set(range(len(counts))))
        self.assertEqual(_distribution_positions(counts, 0), set())

    def test_duplicate_feedback_recovers_all_consistent_answers(self):
        words = ("apple", "ample", "apply", "allee", "level")
        feedback = score_guess("apple", "allee")
        symbols = "".join({"X": "0", "Y": "1", "G": "2"}[mark] for mark in feedback)
        remaining = _consistent_answers("<P><G>allee<F>" + symbols + "<G>", words)
        self.assertEqual(remaining, ("apple", "ample"))
        self.assertEqual(_consistent_answers("<P><G>", words), words)

    def test_exhaustive_rank_uses_remaining_first_and_dictionary_ties(self):
        # Singleton costs all tie, but its remaining answer ranks first even
        # when a probe occurs earlier in dictionary order.
        words = ("cider", "ample", "apple")
        prompt = "<P><G>apple<F>20222<G>"
        action = _score_action(prompt, "cider", words)
        self.assertEqual(action["remaining_answer_count"], 1)
        self.assertEqual(action["selected_rank"], 2)
        self.assertEqual(action["regret"], 0.0)
        self.assertEqual(action["relative_quality"], 1.0)
        self.assertEqual(action["top_guesses"][0]["guess"], "ample")
        # Exhaustive scores/ranks are not restricted to the top eight.
        words = load_words()
        action = _score_action("<P><G>aback<F>22222<G>", "zebra", words)
        self.assertEqual(action["selected_rank"], len(words))

    def test_action_regret_and_relative_quality_use_expected_survivors(self):
        words = ("cider", "apple", "ample", "apply", "angle")
        costs = {word: expected_survivors(words, word) for word in words}
        selected = max(words, key=costs.get)
        action = _score_action("<P><G>", selected, words)
        best = min(costs.values())
        self.assertEqual(action["selected_cost"], costs[selected])
        self.assertEqual(action["best_cost"], best)
        self.assertEqual(action["regret"], costs[selected] - best)
        self.assertEqual(action["relative_quality"], costs[selected] / best)
        order = sorted(words, key=costs.get)
        self.assertEqual(action["selected_rank"], order.index(selected) + 1)

    def test_full_panel_distribution_gameplay_and_mechanics(self):
        words = load_words()
        prompts = [encode("<P><G>" + word + "<F>22222<G>") for word in words[:3]]
        ids = np.tile(np.arange(128), (3, 1))
        remaining = np.zeros((3, 128), dtype=bool)
        remaining[np.arange(3), np.arange(3)] = True
        ranks = np.stack([
            np.array([1 if index == row else index + 2 if index < row else index + 1 for index in range(128)])
            for row in range(3)
        ])
        dataset = ObservableDataset(
            words=words, prompts=np.array(prompts), lengths=np.array([len(prompt) for prompt in prompts]),
            candidate_ids=ids, costs=np.ones((3, 128)), ranks=ranks,
            remaining_counts=np.ones(3), is_remaining=remaining, state_ids=np.array([100, 101, 102]),
        )
        with torch.random.fork_rng():
            model = WordleGPT(vocab_size=VOCABULARY_SIZE, embedding_size=8, num_layers=1, num_heads=2, mlp_size=16)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.zero_()
            for letter, logit in zip("zebra", (10.0, 9.0, 8.0, 7.0, 6.0)):
                model.output.bias[encode(letter)[0]] = logit
        model.train()
        mechanics = V2SplitData(
            inputs=torch.tensor([[0, 1], [2, 3]]),
            targets=torch.tensor([[4, -100], [5, 6]]),
            example_type_ids=torch.zeros(2, dtype=torch.int8),
        )
        report = evaluate_soft_model(
            model, dataset, [2, 0, 1], 0.5, ["zebra", "aback"], words, mechanics, "cpu",
            batch_size=2, distribution_examples=2,
        )
        json.dumps(report, allow_nan=False)
        self.assertTrue(model.training)
        self.assertEqual(report["panel_indices"], [2, 0, 1])
        actions = report["action_quality"]["actions"]
        self.assertEqual([action["dataset_index"] for action in actions], [2, 0, 1])
        self.assertEqual([action["guess"] for action in actions], ["zebra"] * 3)
        self.assertTrue(all(action["selected_rank"] == 719 for action in actions))
        self.assertTrue(all(action["relative_quality"] == 1.0 for action in actions))
        gameplay = report["gameplay"]
        self.assertEqual([result["secret"] for result in gameplay["results"]], ["zebra", "aback"])
        self.assertEqual(gameplay["wins"], 1)
        self.assertEqual(gameplay["average_attempts"], 3.5)
        self.assertEqual(gameplay["average_guesses"], 1.0)
        examples = report["distribution_examples"]
        self.assertEqual([example["dataset_index"] for example in examples], [2, 0])
        raw_token_logps = model.output.bias.detach().log_softmax(-1)
        expected_logps = torch.stack([raw_token_logps[encode(word)].sum() for word in words[:128]])
        expected_student = expected_logps.softmax(-1)
        for example in examples:
            candidates = example["candidates"]
            self.assertEqual([candidate["word"] for candidate in candidates], list(words[:128]))
            self.assertAlmostEqual(sum(candidate["teacher_probability"] for candidate in candidates), 1.0)
            self.assertAlmostEqual(sum(candidate["student_probability"] for candidate in candidates), 1.0, places=5)
            answer = example["dataset_index"]
            self.assertEqual(candidates[answer]["teacher_probability"], 1.0)
            for candidate in candidates:
                self.assertAlmostEqual(candidate["sequence_logp"], float(expected_logps[candidate["id"]]), places=4)
        policy = report["policy"]
        expected_ce = float(-expected_logps.log_softmax(-1)[:3].mean())
        self.assertAlmostEqual(policy["cross_entropy"], expected_ce, places=4)
        self.assertAlmostEqual(policy["teacher_student_kl"], expected_ce, places=4)
        self.assertEqual(policy["teacher_entropy"], 0.0)
        self.assertAlmostEqual(policy["student_entropy"], float(-(expected_student * expected_logps.log_softmax(-1)).sum()), places=4)
        for key, limit in (("rank1_probability", 1), ("top3_probability", 3), ("top8_probability", 8)):
            expected = sum(float(expected_student[torch.tensor(row <= limit)].sum()) for row in ranks) / 3
            self.assertAlmostEqual(policy[key], expected, places=5)
        self.assertAlmostEqual(report["mechanics_validation_loss"], float(-raw_token_logps[[4, 5, 6]].mean()), places=5)

    def test_checkpoint_selection_is_lexicographic_gameplay_before_policy(self):
        def report(values):
            wins, attempts, guesses, regret, kl = values
            return {
                "gameplay": {"wins": wins, "average_attempts": attempts, "average_guesses": guesses},
                "action_quality": {"summary": {"mean_action_regret": regret}},
                "policy": {"teacher_student_kl": kl},
            }

        base = (30, 4.0, 3.0, 2.0, 1.0)
        self.assertEqual(checkpoint_key(report(base)), (30, -4.0, -3.0, -2.0, -1.0))
        for position in range(5):
            better = list(base)
            better[position] += 1 if position == 0 else -0.5
            for later in range(position + 1, 5):
                better[later] += 10
            self.assertGreater(checkpoint_key(report(better)), checkpoint_key(report(base)))


if __name__ == "__main__":
    unittest.main()
