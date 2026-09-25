import gzip
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from grpo_information_states import (
    BalancedStateSampler,
    build_information_pool,
    candidate_bucket,
    load_state_pool,
    make_state,
)
from wordle import score_guess


class InformationStateTests(unittest.TestCase):
    words = ("aaaaa", "aaaab", "aaaac", "zzzzz")

    def test_belief_uses_only_history_and_full_dictionary_not_source_secret(self):
        history = (("zzzzz", "XXXXX"),)
        train = make_state("aaaaa", history, "random", "train", self.words)
        held_out = make_state("aaaab", history, "random", "validation", self.words)
        self.assertEqual(train.candidates, ("aaaaa", "aaaab", "aaaac"))
        self.assertEqual(train.candidates, held_out.candidates)
        self.assertEqual(train.prompt, held_out.prompt)
        with self.assertRaises(ValueError):
            make_state("aaaaa", (("zzzzz", "GGGGG"),), "random", "bad", self.words)
        with self.assertRaises(ValueError):
            make_state("aaaaa", (("yyyyy", "XXXXX"),), "random", "bad", self.words)
        with self.assertRaises(ValueError):
            make_state("aaaaa", (("aaaaa", "GGGGG"),), "random", "solved", self.words)

    def _build_inputs(self, root, wrong_count=False):
        words_path = root / "words.txt"
        words_path.write_text("\n".join(self.words) + "\n", encoding="utf-8")
        mode_path = root / "mode.json"
        mode_path.write_text(json.dumps({"mode": "test", "runs": [
            {"train": ["aaaaa"], "validation": ["aaaab"], "test": ["aaaac"]},
            {"train": ["aaaac"], "validation": ["aaaaa"], "test": ["aaaab"]},
        ]}), encoding="utf-8")
        rows = []
        for secret in self.words:
            for behavior in ("entropy", "clever", "informative", "simple", "partly-random-0.30", "poor", "random"):
                for _ in range(7):
                    rows.append({
                        "state_index": len(rows), "source_secret": secret,
                        # The old source's split label must not override modes runs[0].
                        "split": "train", "source_behavior": behavior,
                        "history": [{"guess": "zzzzz", "feedback": score_guess(secret, "zzzzz")}],
                        "possible_answer_count": 2 if wrong_count else 3,
                    })
        for history in ([], [{"guess": "aaaaa", "feedback": "GGGGG"}],
                        [{"guess": "zzzzz", "feedback": "XXXXX"}] * 6):
            rows.append({
                "state_index": len(rows), "source_secret": "aaaaa", "source_behavior": "random",
                "history": history, "possible_answer_count": 3,
            })
        source = root / "examples.jsonl.gz"
        with gzip.open(source, "wt", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row) + "\n")
        return {"source": source, "mode": mode_path, "words": words_path, "train_cap": 3, "validation_cap": 2, "seed": 19}, len(rows)

    def test_reservoir_is_split_safe_capped_broad_and_reads_entire_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, rows = self._build_inputs(root)
            manifest = build_information_pool(root / "first", **inputs)
            build_information_pool(root / "second", **inputs)
            train = load_state_pool(root / "first" / "train.jsonl", self.words)
            validation = load_state_pool(root / "first" / "validation.jsonl", self.words)
            self.assertEqual({state.secret for state in train}, {"aaaaa"})
            self.assertEqual({state.secret for state in validation}, {"aaaab"})
            self.assertTrue(all("aaaac" in state.candidates for state in train + validation))
            expected_behaviors = {"optimal_entropy", "simple", "partly_random", "poor", "random"}
            self.assertEqual(Counter(state.behavior for state in train), {behavior: 3 for behavior in expected_behaviors})
            self.assertEqual(Counter(state.behavior for state in validation), {behavior: 2 for behavior in expected_behaviors})
            self.assertEqual(manifest["source"]["rows"], rows)
            self.assertEqual(manifest["sampling"]["excluded_rows"], {
                "test": 49, "outside_panel": 49,
                "train:empty_history": 1, "train:solved_history": 1, "train:depth_over_five": 1,
            })
            self.assertEqual(len(manifest["selected_counts"]["train"]["missing_strata"]), 120)
            for name in ("train.jsonl", "validation.jsonl", "manifest.json"):
                self.assertEqual((root / "first" / name).read_bytes(), (root / "second" / name).read_bytes())

    def test_exact_candidate_count_is_verified_before_writing_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            inputs, _ = self._build_inputs(root, wrong_count=True)
            with self.assertRaisesRegex(ValueError, "exact feedback gives 3"):
                build_information_pool(root / "pool", **inputs)
            self.assertFalse((root / "pool" / "train.jsonl").exists())


class BalancedStateSamplerTests(unittest.TestCase):
    words = ("cigar", "rebut", "sissy", "humph", "awake", "blush", "focal", "evade", "naval", "serve", "heath", "dwarf")

    def _states(self):
        # Both histories are reachable with cigar; one is narrow and one broad.
        states = []
        for guess in ("sissy", "humph"):
            for depth in (1, 3):
                history = ((guess, score_guess("cigar", guess)),) * depth
                for behavior, copies in (("random", 1), ("simple", 50)):
                    for index in range(copies):
                        states.append(make_state("cigar", history, behavior, f"{guess}:{depth}:{behavior}:{index}", self.words))
        return states

    def test_seeded_opening_schedule_is_exact_and_independent_of_batch_boundaries(self):
        states = self._states()
        first = BalancedStateSampler(states, self.words, seed=83)
        second = BalancedStateSampler(states, self.words, seed=83)
        draws = first.sample_batch(800)
        chunked = second.sample_batch(7) + second.sample_batch(193) + second.sample_batch(600)
        self.assertEqual(draws, chunked)
        self.assertEqual(sum(not state.history for state in draws), 60)
        for start in range(0, 800, 40):
            self.assertEqual(sum(not state.history for state in draws[start:start + 40]), 3)
        for state in draws:
            if not state.history:
                self.assertEqual(state.candidates, self.words)
                self.assertEqual(state.secret, "cigar")
        self.assertEqual(first.coverage()["total_draws"], 800)
        self.assertEqual(first.coverage()["opening_draws"], 60)
        self.assertEqual(sum(row["count"] for row in first.coverage()["draws"]["strata"]), 740)

    def test_hierarchy_balances_difficulty_depth_behavior_not_pool_multiplicity(self):
        states = self._states()
        self.assertEqual({candidate_bucket(len(state.candidates)) for state in states}, {"1-2", "6-20"})
        sampler = BalancedStateSampler(states, self.words, seed=41, opening_fraction=0)
        draws = sampler.sample_batch(16000)
        counts = Counter((candidate_bucket(len(state.candidates)), len(state.history), state.behavior) for state in draws)
        self.assertEqual(len(counts), 8)
        # Fifty times as many simple rows must not give simple fifty times the draws.
        for count in counts.values():
            self.assertLess(abs(count - 2000), 200)

    def test_sparse_coverage_and_model_refresh_preserve_corpus_and_draw_counters(self):
        corpus = self._states()[0]
        old = make_state("cigar", (("humph", score_guess("cigar", "humph")),), "model", "old", self.words)
        new = make_state("cigar", old.history * 2, "model", "new", self.words)
        sampler = BalancedStateSampler([corpus, old], self.words, seed=8, opening_fraction=0)
        sampler.sample_batch(20)
        sampler.replace_model_states([new])
        coverage = sampler.coverage()
        self.assertEqual(coverage["total_draws"], 20)
        self.assertEqual(coverage["pool_states"], 2)
        self.assertEqual(coverage["model_states"], 1)
        missing = {(row["candidate_bucket"], row["history_depth"], row["behavior"]) for row in coverage["pool"]["missing_strata"]}
        self.assertIn((candidate_bucket(len(old.candidates)), 1, "model"), missing)
        self.assertNotIn((candidate_bucket(len(new.candidates)), 2, "model"), missing)
        sampled = sampler.sample_batch(400)
        self.assertEqual({state.source_index for state in sampled}, {corpus.source_index, "new"})
        sampler.replace_model_states([])
        self.assertEqual(sampler.sample_batch(20), [corpus] * 20)


if __name__ == "__main__":
    unittest.main()
