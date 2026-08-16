from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO

from dataset import state_fingerprint
from tokenizer import END_TOKEN, GUESS_TOKEN, serialize_trajectory
from tokenizer_v2 import POLICY_TOKEN, VOCABULARY_SIZE, encode
from wordle import (
    DEFAULT_WORDS,
    GREEN,
    _feedback_code,
    choose_informative_guess,
    filter_answers,
    load_words,
    score_guess,
)

SCHEMA_VERSION = 1
DEFAULT_OUTPUT_DIR = Path("data/wordle-v2-diverse")
DEFAULT_SPLITS_PATH = Path("data/wordle-100k/secret-splits.json")
DEFAULT_PREFIXES = (10_000, 50_000, 100_000, 200_000)
PARTLY_RANDOM_RATES = (0.05, 0.15, 0.30, 0.50, 0.75, 0.90)
BEHAVIORS = (
    "random",
    "simple",
    "entropy",
    *(f"partly-random-{rate:.2f}" for rate in PARTLY_RANDOM_RATES),
    "poor",
)


def _open_gzip_text(path: Path) -> IO[str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed_file = gzip.GzipFile(filename=str(path), mode="wb", mtime=0)
    return io.TextIOWrapper(compressed_file, encoding="utf-8", newline="\n")


def _word_list_hash(words: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(words) + "\n").encode("ascii")).hexdigest()


def _available_words(words: Sequence[str], used: set[str]) -> list[str]:
    available = [word for word in words if word not in used]
    return available or list(words)


def _partition_cost(possible_answers: Sequence[str], guess: str) -> int:
    buckets: dict[int, int] = {}
    cost = 0
    for answer in possible_answers:
        code = _feedback_code(answer, guess)
        old_size = buckets.get(code, 0)
        buckets[code] = old_size + 1
        cost += 2 * old_size + 1
    return cost


def choose_entropy_guess(
    possible_answers: Sequence[str], allowed_guesses: Sequence[str]
) -> str:
    """Choose the legal guess with maximum Shannon feedback entropy."""
    if not possible_answers or not allowed_guesses:
        raise ValueError("entropy choice requires answers and legal guesses")
    if len(possible_answers) == 1:
        return possible_answers[0]
    best_guess = allowed_guesses[0]
    best_score = math.inf
    for guess in allowed_guesses:
        counts: dict[int, int] = {}
        for answer in possible_answers:
            code = _feedback_code(answer, guess)
            counts[code] = counts.get(code, 0) + 1
        score = sum(count * math.log(count) for count in counts.values())
        if score < best_score:
            best_guess = guess
            best_score = score
    return best_guess


def choose_poor_legal_guess(
    possible_answers: Sequence[str],
    allowed_guesses: Sequence[str],
    used_guesses: set[str],
    rng: random.Random,
    *,
    candidate_count: int = 64,
) -> str:
    """Choose the least informative member of a reproducible legal sample."""
    available = _available_words(allowed_guesses, used_guesses)
    candidates = (
        available
        if len(available) <= candidate_count
        else rng.sample(available, candidate_count)
    )
    return max(candidates, key=lambda guess: _partition_cost(possible_answers, guess))


def _source_guess(
    behavior: str,
    possible_answers: tuple[str, ...],
    words: Sequence[str],
    used_guesses: set[str],
    rng: random.Random,
    clever_cache: dict[tuple[str, ...], str],
    entropy_cache: dict[tuple[str, ...], str],
) -> tuple[str, str]:
    available = _available_words(words, used_guesses)
    if behavior == "random":
        return rng.choice(available), "random-legal"
    if behavior == "simple":
        candidates = [word for word in possible_answers if word not in used_guesses]
        return rng.choice(candidates or list(possible_answers)), "simple-consistent"
    if behavior == "entropy":
        guess = entropy_cache.get(possible_answers)
        if guess is None or guess in used_guesses:
            guess = choose_entropy_guess(possible_answers, available)
            if not used_guesses:
                entropy_cache[possible_answers] = guess
        return guess, "maximum-entropy"
    if behavior == "poor":
        return (
            choose_poor_legal_guess(possible_answers, words, used_guesses, rng),
            "poor-sampled-max-survivors",
        )
    if behavior.startswith("partly-random-"):
        random_rate = float(behavior.removeprefix("partly-random-"))
        if rng.random() < random_rate:
            return rng.choice(available), f"random-legal@{random_rate:.2f}"
        guess = clever_cache.get(possible_answers)
        if guess is None or guess in used_guesses:
            guess = choose_informative_guess(possible_answers, available)
            if not used_guesses:
                clever_cache[possible_answers] = guess
        return guess, f"clever@{1.0 - random_rate:.2f}"
    raise ValueError(f"unknown behavior: {behavior!r}")


def _expert_example(
    *,
    state_index: int,
    split: str,
    source_secret: str,
    source_behavior: str,
    source_action_policy: str,
    history: Sequence[dict[str, str]],
    possible_answers: tuple[str, ...],
    words: Sequence[str],
    expert_cache: dict[tuple[str, ...], str],
) -> dict[str, object]:
    expert_guess = expert_cache.get(possible_answers)
    if expert_guess is None:
        expert_guess = choose_informative_guess(possible_answers, words)
        expert_cache[possible_answers] = expert_guess
    history_text = serialize_trajectory(history)
    text = (
        POLICY_TOKEN
        + history_text[: -len(END_TOKEN)]
        + GUESS_TOKEN
        + expert_guess
        + END_TOKEN
    )
    target_start = 1 + len(history) * 12
    return {
        "schema_version": SCHEMA_VERSION,
        "state_index": state_index,
        "state_fingerprint": state_fingerprint(history),
        "example_type": "expert",
        "split": split,
        "history": list(history),
        "source_secret": source_secret,
        "source_behavior": source_behavior,
        "source_action_policy": source_action_policy,
        "expert_policy": "clever",
        "possible_answer_count": len(possible_answers),
        "desired_guess": expert_guess,
        "text": text,
        "token_ids": encode(text),
        "sampling_weight": 10_000 if not history else 1,
        "loss_ranges": [[target_start, target_start + 5]],
        "supervised_target": "expert_guess_letters",
    }


def build_diverse_expert_dataset(
    output_dir: str | Path,
    words: Sequence[str],
    split_secrets: Mapping[str, Sequence[str]],
    *,
    total_states: int = 200_000,
    prefixes: Sequence[int] = DEFAULT_PREFIXES,
    behaviors: Sequence[str] = BEHAVIORS,
    seed: int = 20260815,
    max_turns: int = 6,
) -> dict[str, object]:
    """Generate unique off-policy histories and label every state with clever."""
    if total_states < 1 or max_turns < 1:
        raise ValueError("total_states and max_turns must be positive")
    if not behaviors or len(set(behaviors)) != len(behaviors):
        raise ValueError("behaviors must be nonempty and unique")
    unknown = set(behaviors) - set(BEHAVIORS)
    if unknown:
        raise ValueError(f"unknown behaviors: {', '.join(sorted(unknown))}")
    normalized_prefixes = tuple(sorted(set(prefixes)))
    if not normalized_prefixes or normalized_prefixes[-1] != total_states:
        raise ValueError("prefixes must end at total_states")
    if normalized_prefixes[0] < 1:
        raise ValueError("prefixes must be positive")

    secret_to_split: dict[str, str] = {}
    source_secrets: list[str] = []
    for split, secrets in split_secrets.items():
        for secret in secrets:
            if secret in secret_to_split:
                raise ValueError(f"secret appears in multiple splits: {secret}")
            if secret not in words:
                raise ValueError(f"secret is absent from word list: {secret}")
            secret_to_split[secret] = split
            source_secrets.append(secret)
    if not source_secrets:
        raise ValueError("at least one source secret is required")

    output = Path(output_dir)
    examples_path = output / "examples.jsonl.gz"
    rng = random.Random(seed)
    clever_cache: dict[tuple[str, ...], str] = {}
    entropy_cache: dict[tuple[str, ...], str] = {}
    fingerprints: set[str] = set()
    behavior_counts: Counter[str] = Counter()
    policy_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    prefix_statistics: dict[int, dict[str, object]] = {}
    trajectory_count = 0

    with _open_gzip_text(examples_path) as destination:
        while len(fingerprints) < total_states:
            behavior = behaviors[trajectory_count % len(behaviors)]
            secret = rng.choice(source_secrets)
            split = secret_to_split[secret]
            possible_answers = tuple(words)
            used_guesses: set[str] = set()
            history: list[dict[str, str]] = []
            source_policy = "initial"

            for _ in range(max_turns):
                fingerprint = state_fingerprint(history)
                if fingerprint not in fingerprints:
                    state_index = len(fingerprints)
                    example = _expert_example(
                        state_index=state_index,
                        split=split,
                        source_secret=secret,
                        source_behavior=behavior,
                        source_action_policy=source_policy,
                        history=history,
                        possible_answers=possible_answers,
                        words=words,
                        expert_cache=clever_cache,
                    )
                    destination.write(json.dumps(example, separators=(",", ":")) + "\n")
                    fingerprints.add(fingerprint)
                    behavior_counts[behavior] += 1
                    policy_counts[source_policy] += 1
                    split_counts[split] += 1
                    current_size = len(fingerprints)
                    if current_size in normalized_prefixes:
                        prefix_statistics[current_size] = {
                            "unique_states": current_size,
                            "split_counts": dict(sorted(split_counts.items())),
                            "source_behavior_counts": dict(sorted(behavior_counts.items())),
                        }
                    if current_size == total_states:
                        break

                guess, source_policy = _source_guess(
                    behavior,
                    possible_answers,
                    words,
                    used_guesses,
                    rng,
                    clever_cache,
                    entropy_cache,
                )
                used_guesses.add(guess)
                feedback = score_guess(secret, guess)
                history.append({"guess": guess, "feedback": feedback})
                possible_answers = tuple(
                    filter_answers(possible_answers, guess, feedback)
                )
                if feedback == GREEN * 5:
                    break
            trajectory_count += 1

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "word_count": len(words),
        "word_list_sha256": _word_list_hash(words),
        "vocabulary_size": VOCABULARY_SIZE,
        "file": examples_path.name,
        "total_states": total_states,
        "unique_states": len(fingerprints),
        "trajectory_count": trajectory_count,
        "max_turns": max_turns,
        "behaviors": list(behaviors),
        "partly_random_rates": list(PARTLY_RANDOM_RATES),
        "source_secret_counts": {
            split: len(secrets) for split, secrets in split_secrets.items()
        },
        "split_counts": dict(sorted(split_counts.items())),
        "source_behavior_counts": dict(sorted(behavior_counts.items())),
        "source_action_policy_counts": dict(sorted(policy_counts.items())),
        "sampling": {
            "unique_state_storage": True,
            "canonical_empty_history_weight": 10_000,
            "training_only": True,
        },
        "expert_policy": "clever/minimum expected survivors",
        "nested_datasets": {
            str(size): {
                "file": examples_path.name,
                "state_index_range": [0, size],
                **prefix_statistics[size],
            }
            for size in normalized_prefixes
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_diverse_expert_dataset(
    data_dir: str | Path,
    words: Sequence[str],
    split_secrets: Mapping[str, Sequence[str]],
) -> dict[str, object]:
    """Validate every history, unique identity, clever target, and loss mask."""
    secret_to_split = {
        secret: split
        for split, secrets in split_secrets.items()
        for secret in secrets
    }
    fingerprints: set[str] = set()
    expert_cache: dict[tuple[str, ...], str] = {}
    split_counts: Counter[str] = Counter()
    behavior_counts: Counter[str] = Counter()
    examples_path = Path(data_dir) / "examples.jsonl.gz"
    with gzip.open(examples_path, "rt", encoding="utf-8") as source:
        for expected_index, line in enumerate(source):
            example = json.loads(line)
            if example["state_index"] != expected_index:
                raise ValueError("state indices are not contiguous")
            history = example["history"]
            fingerprint = state_fingerprint(history)
            if example["state_fingerprint"] != fingerprint:
                raise ValueError(f"state {expected_index} has an invalid fingerprint")
            if fingerprint in fingerprints:
                raise ValueError(f"state {expected_index} duplicates an earlier history")
            fingerprints.add(fingerprint)
            secret = example["source_secret"]
            if secret_to_split.get(secret) != example["split"]:
                raise ValueError(f"state {expected_index} has an invalid secret split")

            possible_answers = tuple(words)
            for turn in history:
                if turn["guess"] not in words:
                    raise ValueError(f"state {expected_index} has an illegal source guess")
                if score_guess(secret, turn["guess"]) != turn["feedback"]:
                    raise ValueError(f"state {expected_index} has incorrect feedback")
                possible_answers = tuple(
                    filter_answers(
                        possible_answers,
                        turn["guess"],
                        turn["feedback"],
                    )
                )
            if secret not in possible_answers:
                raise ValueError(f"state {expected_index} eliminated its source secret")
            expert_guess = expert_cache.get(possible_answers)
            if expert_guess is None:
                expert_guess = choose_informative_guess(possible_answers, words)
                expert_cache[possible_answers] = expert_guess
            expected_weight = 10_000 if not history else 1
            if example.get("sampling_weight") != expected_weight:
                raise ValueError(f"state {expected_index} has an invalid sampling weight")
            if example["desired_guess"] != expert_guess:
                raise ValueError(f"state {expected_index} has an incorrect clever target")

            history_text = serialize_trajectory(history)
            expected_text = (
                POLICY_TOKEN
                + history_text[: -len(END_TOKEN)]
                + GUESS_TOKEN
                + expert_guess
                + END_TOKEN
            )
            if example["text"] != expected_text or example["token_ids"] != encode(
                expected_text
            ):
                raise ValueError(f"state {expected_index} has invalid serialization")
            target_start = 1 + len(history) * 12
            if example["loss_ranges"] != [[target_start, target_start + 5]]:
                raise ValueError(f"state {expected_index} has an invalid loss mask")
            split_counts[example["split"]] += 1
            behavior_counts[example["source_behavior"]] += 1

    manifest = json.loads(
        (Path(data_dir) / "manifest.json").read_text(encoding="utf-8")
    )
    if len(fingerprints) != manifest["unique_states"]:
        raise ValueError("manifest unique-state count does not match examples")
    if dict(sorted(split_counts.items())) != manifest["split_counts"]:
        raise ValueError("manifest split counts do not match examples")
    return {
        "examples": len(fingerprints),
        "unique_histories": len(fingerprints),
        "unique_answer_sets": len(expert_cache),
        "split_counts": dict(sorted(split_counts.items())),
        "source_behavior_counts": dict(sorted(behavior_counts.items())),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate unique off-policy Wordle states with clever targets."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--splits", type=Path, default=DEFAULT_SPLITS_PATH)
    parser.add_argument("--total-states", type=int, default=200_000)
    parser.add_argument("--prefixes", type=int, nargs="+")
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument(
        "--include-test",
        action="store_true",
        help="Include fixed test secrets only when building a universal CV pool.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    split_payload = json.loads(args.splits.read_text(encoding="utf-8"))["splits"]
    source_split_names = (
        ("train", "validation", "test")
        if args.include_test
        else ("train", "validation")
    )
    source_splits = {split: split_payload[split] for split in source_split_names}
    words = load_words(args.words)
    if args.validate_only:
        result = validate_diverse_expert_dataset(
            args.output_dir,
            words,
            source_splits,
        )
    else:
        prefixes = tuple(args.prefixes or (*DEFAULT_PREFIXES[:-1], args.total_states))
        result = build_diverse_expert_dataset(
            args.output_dir,
            words,
            source_splits,
            total_states=args.total_states,
            prefixes=prefixes,
            seed=args.seed,
            max_turns=args.max_turns,
        )
    manifest = result
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
