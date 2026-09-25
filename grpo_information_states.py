from __future__ import annotations

import argparse
import gzip
import json
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from grpo_corpus import DEFAULT_MODE, DEFAULT_SOURCE, _sha256_file, _split_secrets
from grpo_states import _cached_feedback, _filter_with_feedback
from tokenizer import END_TOKEN, GUESS_TOKEN, serialize_trajectory
from tokenizer_v2 import POLICY_TOKEN, encode
from wordle import DEFAULT_WORDS, GREEN, _validate_word, load_words

DEFAULT_OUTPUT_DIR = Path("data/grpo-information")
BEHAVIORS = ("optimal_entropy", "simple", "partly_random", "poor", "random")
CANDIDATE_BUCKETS = ("1-2", "3-5", "6-20", "21-100", "101+")


def candidate_bucket(count: int) -> str:
    if count < 1:
        raise ValueError("a reachable state must have at least one candidate")
    for upper, name in zip((2, 5, 20, 100), CANDIDATE_BUCKETS):
        if count <= upper:
            return name
    return CANDIDATE_BUCKETS[-1]


def classify_behavior(behavior: str) -> str:
    if behavior in ("entropy", "clever", "informative", "optimal_entropy"):
        return "optimal_entropy"
    if behavior.startswith("partly-random-") or behavior == "partly_random":
        return "partly_random"
    if behavior in BEHAVIORS:
        return behavior
    raise ValueError(f"unknown corpus behavior: {behavior!r}")


@lru_cache(maxsize=8)
def _word_set(words: tuple[str, ...]) -> frozenset[str]:
    if not words or any(_validate_word(word) != word for word in words):
        raise ValueError("words must be nonempty normalized five-letter words")
    allowed = frozenset(words)
    if len(allowed) != len(words):
        raise ValueError("words must be unique")
    return allowed


@lru_cache(maxsize=8192)
def _infer_candidates(
    words: tuple[str, ...], history: tuple[tuple[str, str], ...]
) -> tuple[str, ...]:
    if not history:
        return words
    guess, feedback = history[-1]
    return _filter_with_feedback(_infer_candidates(words, history[:-1]), guess, feedback)


@dataclass(frozen=True)
class InformationState:
    secret: str
    history: tuple[tuple[str, str], ...]
    prompt: tuple[int, ...]
    candidates: tuple[str, ...]
    behavior: str
    source_index: str


def make_state(
    secret: str,
    history: Sequence[tuple[str, str]],
    behavior: str,
    source_index: str,
    words: Sequence[str],
) -> InformationState:
    """Infer belief from observations and the FULL dictionary, never from the secret split."""
    words = tuple(words)
    allowed = _word_set(words)
    history = tuple((guess, feedback) for guess, feedback in history)
    if secret not in allowed:
        raise ValueError(f"secret is outside the dictionary: {secret!r}")
    if len(history) > 5:
        raise ValueError("pre-action histories must have at most five turns")
    if behavior not in (*BEHAVIORS, "model", "opening"):
        raise ValueError(f"unknown state behavior: {behavior!r}")
    for guess, feedback in history:
        if guess not in allowed:
            raise ValueError(f"history contains an illegal guess: {guess!r}")
        if feedback != _cached_feedback(secret, guess):
            raise ValueError(f"history feedback does not match secret for {guess!r}")
        if feedback == GREEN * 5:
            raise ValueError("solved histories are not pre-action states")
    candidates = _infer_candidates(words, history)
    if secret not in candidates:
        raise ValueError("history eliminated its own secret")
    serialized = serialize_trajectory(
        tuple({"guess": guess, "feedback": feedback} for guess, feedback in history)
    )
    prompt = tuple(encode(POLICY_TOKEN + serialized[: -len(END_TOKEN)] + GUESS_TOKEN))
    return InformationState(secret, history, prompt, candidates, behavior, str(source_index))


def load_state_pool(path: str | Path, words: Sequence[str]) -> list[InformationState]:
    states = []
    with Path(path).open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                row = json.loads(line)
                states.append(make_state(
                    row["secret"], row["history"], row["behavior"], row["source_index"], words
                ))
    return states


def _stratum(state: InformationState) -> tuple[str, int, str]:
    return candidate_bucket(len(state.candidates)), len(state.history), state.behavior


def _coverage(counts: Counter, behaviors: Sequence[str]) -> dict:
    return {
        "strata": [
            {"candidate_bucket": bucket, "history_depth": depth, "behavior": behavior, "count": count}
            for (bucket, depth, behavior), count in sorted(counts.items())
        ],
        "missing_strata": [
            {"candidate_bucket": bucket, "history_depth": depth, "behavior": behavior}
            for bucket in CANDIDATE_BUCKETS
            for depth in range(1, 6)
            for behavior in behaviors
            if not counts[bucket, depth, behavior]
        ],
    }


class BalancedStateSampler:
    """Uniform available bucket, then depth, then behavior, then state.

    Opening groups use a seeded integer accumulator rather than a Bernoulli draw.
    Missing strata are reported, never manufactured or filled by another source.
    Caller-supplied pools must contain only training source secrets; inference still
    uses the full dictionary, including held-out possible answers.
    """

    def __init__(
        self, states: Sequence[InformationState], words: Sequence[str], seed: int,
        opening_fraction: float = 0.075,
    ) -> None:
        if not 0 <= opening_fraction <= 1:
            raise ValueError("opening_fraction must be between zero and one")
        self.words = tuple(words)
        _word_set(self.words)
        self.rng = random.Random(seed)
        self.opening_fraction = opening_fraction
        fraction = Fraction(str(opening_fraction))
        self._opening_numerator = fraction.numerator
        self._opening_denominator = fraction.denominator
        self._opening_accumulator = self.rng.randrange(fraction.denominator)
        self._corpus_states = tuple(state for state in states if state.behavior != "model")
        self._model_states = tuple(state for state in states if state.behavior == "model")
        self._draws = Counter()
        self._opening_draws = 0
        self._total_draws = 0
        self._rebuild()

    def _rebuild(self) -> None:
        states = self._corpus_states + self._model_states
        self._secrets = tuple(sorted({state.secret for state in states}))
        if not self._secrets:
            raise ValueError("at least one training-pool secret is required")
        cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        self._pool_counts = Counter()
        for state in states:
            if not state.history:
                continue
            bucket, depth, behavior = _stratum(state)
            if depth > 5:
                raise ValueError("state pool contains a six-turn or longer history")
            cells[bucket][depth][behavior].append(state)
            self._pool_counts[bucket, depth, behavior] += 1
        if not cells and self.opening_fraction < 1:
            raise ValueError("no nonempty states are available for continuation groups")
        self._cells = dict(cells)
        self._buckets = tuple(bucket for bucket in CANDIDATE_BUCKETS if bucket in cells)
        self._depths = {bucket: tuple(sorted(cells[bucket])) for bucket in self._buckets}
        self._behaviors = {
            (bucket, depth): tuple(sorted(cells[bucket][depth]))
            for bucket in self._buckets for depth in self._depths[bucket]
        }

    def replace_model_states(self, states: Sequence[InformationState]) -> None:
        replacement = tuple(states)
        if any(state.behavior != "model" for state in replacement):
            raise ValueError("replacement states must all have behavior='model'")
        previous = self._model_states
        self._model_states = replacement
        try:
            self._rebuild()
        except ValueError:
            self._model_states = previous
            self._rebuild()
            raise

    def sample_batch(self, count: int) -> list[InformationState]:
        if count < 0:
            raise ValueError("batch count must be nonnegative")
        result = []
        for _ in range(count):
            self._opening_accumulator += self._opening_numerator
            if self._opening_accumulator >= self._opening_denominator:
                self._opening_accumulator -= self._opening_denominator
                state = make_state(
                    self.rng.choice(self._secrets), (), "opening",
                    f"opening:{self._total_draws}", self.words,
                )
                self._opening_draws += 1
            else:
                bucket = self.rng.choice(self._buckets)
                depth = self.rng.choice(self._depths[bucket])
                behavior = self.rng.choice(self._behaviors[bucket, depth])
                state = self.rng.choice(self._cells[bucket][depth][behavior])
                self._draws[bucket, depth, behavior] += 1
            self._total_draws += 1
            result.append(state)
        return result

    def coverage(self) -> dict:
        return {
            "pool": _coverage(self._pool_counts, (*BEHAVIORS, "model")),
            "draws": _coverage(self._draws, (*BEHAVIORS, "model")),
            "pool_states": sum(self._pool_counts.values()),
            "model_states": len(self._model_states),
            "training_pool_secrets": len(self._secrets),
            "total_draws": self._total_draws,
            "opening_draws": self._opening_draws,
            "opening_fraction": self.opening_fraction,
            "opening_accumulator": self._opening_accumulator,
            "opening_denominator": self._opening_denominator,
        }


def build_information_pool(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR, *,
    source: str | Path = DEFAULT_SOURCE, mode: str | Path = DEFAULT_MODE,
    words: str | Path = DEFAULT_WORDS, train_cap: int = 256,
    validation_cap: int = 32, seed: int = 20260924,
) -> dict:
    """Read the entire source and reservoir-sample each split/behavior/bucket/depth."""
    if train_cap < 1 or validation_cap < 1:
        raise ValueError("reservoir capacities must be positive")
    output_dir, source, mode, words = map(Path, (output_dir, source, mode, words))
    outputs = {name: output_dir / name for name in ("train.jsonl", "validation.jsonl", "manifest.json")}
    if any(path.exists() for path in outputs.values()):
        raise FileExistsError(f"refusing to overwrite existing information pool in {output_dir}")
    dictionary = load_words(words)
    splits, first_run, mode_name = _split_secrets(mode, set(dictionary))
    split_by_secret = {secret: split for split, secrets in splits.items() for secret in secrets}
    caps = {"train": train_cap, "validation": validation_cap}
    rngs = {"train": random.Random(seed), "validation": random.Random(seed + 1)}
    reservoirs = defaultdict(list)
    eligible = Counter()
    source_splits = Counter()
    raw_behaviors = Counter()
    exclusions = Counter()
    total_rows = 0
    with gzip.open(source, "rt", encoding="utf-8") as source_file:
        for line_number, line in enumerate(source_file, 1):
            if not line.strip():
                continue
            total_rows += 1
            row = json.loads(line)
            split = split_by_secret.get(row["source_secret"], "outside_panel")
            source_splits[split] += 1
            if split not in caps:
                exclusions[split] += 1
                continue
            raw_history = row["history"]
            depth = len(raw_history)
            if not depth or depth > 5:
                exclusions[f"{split}:{'empty_history' if not depth else 'depth_over_five'}"] += 1
                continue
            if any(turn["feedback"] == GREEN * 5 or turn["guess"] == row["source_secret"] for turn in raw_history):
                exclusions[f"{split}:solved_history"] += 1
                continue
            behavior = classify_behavior(row["source_behavior"])
            raw_behaviors[row["source_behavior"]] += 1
            count = row["possible_answer_count"]
            if not isinstance(count, int) or isinstance(count, bool) or count > len(dictionary):
                raise ValueError(f"source line {line_number} has an invalid candidate count")
            key = split, candidate_bucket(count), depth, behavior
            eligible[key] += 1
            reservoir = reservoirs[key]
            replacement = len(reservoir)
            if replacement >= caps[split]:
                replacement = rngs[split].randrange(eligible[key])
                if replacement >= caps[split]:
                    continue
            # Do not retain large expert targets/token arrays in the reservoir.
            item = {
                "secret": row["source_secret"],
                "history": tuple((turn["guess"], turn["feedback"]) for turn in raw_history),
                "behavior": behavior,
                "source_index": str(row["state_index"]),
                "candidate_count": count,
            }
            if replacement == len(reservoir):
                reservoir.append(item)
            else:
                reservoir[replacement] = item

    selected = {"train": [], "validation": []}
    actual_counts = {split: Counter() for split in caps}
    for (split, bucket, depth, behavior), reservoir in sorted(reservoirs.items()):
        for row in reservoir:
            state = make_state(row["secret"], row["history"], behavior, row["source_index"], dictionary)
            if len(state.candidates) != row["candidate_count"]:
                raise ValueError(
                    f"source state {state.source_index}: advertised {row['candidate_count']} candidates, "
                    f"exact feedback gives {len(state.candidates)}"
                )
            selected[split].append(state)
            actual_counts[split][_stratum(state)] += 1

    manifest = {
        "schema_version": 1,
        "source": {"path": str(source), "sha256": _sha256_file(source), "rows": total_rows},
        "mode": {"path": str(mode), "sha256": _sha256_file(mode), "name": mode_name, "run": first_run.get("run", 1)},
        "words": {"path": str(words), "sha256": _sha256_file(words), "count": len(dictionary)},
        "seed": seed,
        "sampling": {
            "algorithm": "reservoir per mode source split, behavior, candidate bucket, history depth",
            "caps": caps, "source_rows_by_split": dict(source_splits),
            "eligible_source_behaviors": dict(sorted(raw_behaviors.items())),
            "excluded_rows": dict(sorted(exclusions.items())),
            "eligible_strata": {
                split: _coverage(Counter({key[1:]: count for key, count in eligible.items() if key[0] == split}), BEHAVIORS)
                for split in caps
            },
        },
        "selected_counts": {split: _coverage(counts, BEHAVIORS) for split, counts in actual_counts.items()},
        "held_out_exclusion": {
            "train_secrets": sorted(splits["train"]),
            "validation_secrets": sorted(splits["validation"]),
            "test_secrets": sorted(splits["test"]),
            "held_out_secrets_excluded_from_training": sorted(splits["validation"] + splits["test"]),
            "test_source_states_included": False,
            "candidate_sets_use_all_dictionary_words": True,
        },
        "files": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, states in selected.items():
        path = outputs[f"{split}.jsonl"]
        with path.open("x", encoding="utf-8") as destination:
            for state in states:
                row = {"secret": state.secret, "history": state.history, "behavior": state.behavior, "source_index": state.source_index}
                destination.write(json.dumps(row, separators=(",", ":")) + "\n")
        manifest["files"][path.name] = {"rows": len(states), "sha256": _sha256_file(path)}
    with outputs["manifest.json"].open("x", encoding="utf-8") as destination:
        json.dump(manifest, destination, indent=2, sort_keys=True)
        destination.write("\n")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build difficulty/depth/behavior-balanced information-GRPO state pools.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--mode", type=Path, default=DEFAULT_MODE)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--train-cap", type=int, default=256)
    parser.add_argument("--validation-cap", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args(argv)
    manifest = build_information_pool(**vars(args))
    print(json.dumps(manifest["files"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
