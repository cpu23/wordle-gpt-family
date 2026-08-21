import random
import tempfile
import unittest
from pathlib import Path

from wordle import (
    GREEN,
    choose_answer,
    choose_informative_guess,
    filter_answers,
    load_words,
    score_guess,
    solve,
    top_informative_guesses,
)


class WordListTests(unittest.TestCase):
    def test_load_words_normalizes_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "words.txt")
            path.write_text("Apple\ncrane\nAPPLE\n\n", encoding="utf-8")
            self.assertEqual(load_words(path), ("apple", "crane"))

    def test_load_words_rejects_malformed_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "words.txt")
            path.write_text("apple\nfour\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "line 2"):
                load_words(path)

    def test_choose_answer_uses_supplied_random_source(self):
        words = ("apple", "crane", "slate")
        self.assertEqual(choose_answer(words, random.Random(7)), "crane")


class FeedbackTests(unittest.TestCase):
    def test_all_green(self):
        self.assertEqual(score_guess("crane", "crane"), "GGGGG")

    def test_duplicate_guess_letters_are_consumed_once(self):
        self.assertEqual(score_guess("apple", "allee"), "GYXXG")
        self.assertEqual(score_guess("cacao", "aaaaa"), "XGXGX")

    def test_greens_are_reserved_before_yellows(self):
        self.assertEqual(score_guess("abbey", "babes"), "YYGGX")

    def test_filter_keeps_exactly_matching_duplicate_patterns(self):
        possible = ("apple", "ample", "addle", "alley", "angle")
        feedback = score_guess("apple", "allee")
        expected = tuple(word for word in possible if score_guess(word, "allee") == feedback)
        self.assertEqual(filter_answers(possible, "allee", feedback), expected)
        self.assertEqual(expected, ("apple", "ample", "addle", "angle"))


class SolverTests(unittest.TestCase):
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

    def test_simple_solver_only_guesses_remaining_answers(self):
        possible = self.WORDS
        turns = solve("apple", self.WORDS, "simple", random.Random(3))
        for turn in turns:
            self.assertIn(turn.guess, possible)
            possible = filter_answers(possible, turn.guess, turn.feedback)
            self.assertEqual(turn.remaining, len(possible))
        self.assertEqual(turns[-1].feedback, GREEN * 5)

    def test_informative_guess_minimizes_expected_survivors(self):
        possible = self.WORDS[2:10]
        chosen = choose_informative_guess(possible, self.WORDS)

        def bucket_cost(guess):
            counts = {}
            for answer in possible:
                pattern = score_guess(answer, guess)
                counts[pattern] = counts.get(pattern, 0) + 1
            return sum(size * size for size in counts.values())

        self.assertEqual(bucket_cost(chosen), min(map(bucket_cost, self.WORDS)))

    def test_both_solvers_finish_from_full_candidate_set(self):
        for strategy in ("simple", "clever"):
            with self.subTest(strategy=strategy):
                turns = solve("frame", self.WORDS, strategy, random.Random(11))
                self.assertEqual(turns[-1].guess, "frame")
                self.assertEqual(turns[-1].feedback, GREEN * 5)
                self.assertTrue(all(turn.remaining >= 1 for turn in turns))

    def test_top_guesses_match_best_and_rank_by_expected_survivors(self):
        possible = self.WORDS[2:10]
        top = top_informative_guesses(possible, self.WORDS, count=8)
        self.assertEqual(top[0][0], choose_informative_guess(possible, self.WORDS))

        def bucket_cost(guess):
            counts = {}
            for answer in possible:
                pattern = score_guess(answer, guess)
                counts[pattern] = counts.get(pattern, 0) + 1
            return sum(size * size for size in counts.values())

        possible_set = set(possible)
        candidates = list(possible)
        candidates.extend(word for word in self.WORDS if word not in possible_set)
        costs = {word: bucket_cost(word) for word in candidates}
        self.assertEqual(
            top,
            sorted(
                ((word, costs[word] / len(possible)) for word in candidates),
                key=lambda item: (item[1], candidates.index(item[0])),
            )[:8],
        )

    def test_top_guesses_respect_count_and_single_answer_state(self):
        possible = self.WORDS[2:10]
        self.assertEqual(len(top_informative_guesses(possible, self.WORDS, 3)), 3)
        single = top_informative_guesses(("apple",), self.WORDS, 8)
        self.assertEqual(single[0], ("apple", 1.0))
        self.assertEqual(len(single), 8)
        self.assertTrue(all(score == 1.0 for _, score in single))


if __name__ == "__main__":
    unittest.main()
