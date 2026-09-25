"""Observable-state exhaustive Wordle teachers and persisted soft-action data."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from tokenizer import END_TOKEN, GUESS_TOKEN, serialize_trajectory
from tokenizer_v2 import POLICY_TOKEN, encode
from wordle import DEFAULT_WORDS, _feedback_code, load_words

DEFAULT_SOURCE = Path("data/wordle-v2-diverse-1m/examples.jsonl.gz")
ARRAY_NAMES = ("prompts", "lengths", "candidate_ids", "costs", "ranks",
               "remaining_counts", "is_remaining", "source_ids", "state_ids", "weights")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def feedback_matrix(words: Sequence[str]) -> np.ndarray:
    """Rows are guesses, columns are answers; duplicate-letter semantics are exact."""
    return np.asarray([[_feedback_code(answer, guess) for answer in words]
                       for guess in words], dtype=np.uint8)


def exhaustive_scores(matrix: np.ndarray, remaining: Sequence[int]):
    """Return integer partition numerators and exhaustive 1-based solver ranks."""
    remaining = np.asarray(remaining, dtype=np.int64)
    if not len(remaining):
        raise ValueError("observable history has no possible answers")
    vocabulary = len(matrix)
    codes = matrix[:, remaining].astype(np.int64)
    codes += np.arange(vocabulary, dtype=np.int64)[:, None] * 243
    buckets = np.bincount(codes.ravel(), minlength=vocabulary * 243).reshape(vocabulary, 243)
    numerators = np.einsum("ij,ij->i", buckets, buckets).astype(np.uint32)
    membership = np.zeros(vocabulary, dtype=bool)
    membership[remaining] = True
    # Inference preserves dictionary order, matching the existing solver.
    tie_order = np.concatenate((remaining, np.flatnonzero(~membership)))
    order = tie_order[np.argsort(numerators[tie_order], kind="stable")]
    ranks = np.empty(vocabulary, dtype=np.uint16)
    ranks[order] = np.arange(1, vocabulary + 1)
    return numerators, ranks


def select_candidates(ranks, remaining, stored_top, desired_guess, *, seed, observable_key):
    """32 best, 32 rank-33..256, 64 others, with mandatory guesses retained."""
    ranks = np.asarray(ranks)
    if len(ranks) < 128:
        raise ValueError("128 distinct teacher candidates require at least 128 words")
    order = np.argsort(ranks, kind="stable")
    mandatory = set(map(int, stored_top)) | {int(order[0])}
    if desired_guess in remaining:
        mandatory.add(int(desired_guess))
    if any(word < 0 or word >= len(ranks) for word in mandatory):
        raise ValueError("mandatory candidate is outside the dictionary")
    seed_bytes = hashlib.sha256(str(seed).encode("ascii") + b":" + bytes(observable_key)).digest()
    rng = np.random.default_rng(int.from_bytes(seed_bytes[:16], "little"))
    mandatory_mask = np.zeros(len(ranks), dtype=bool)
    mandatory_mask[list(mandatory)] = True
    best = order[:32]
    middle_pool = order[32:min(256, len(order))]
    middle = middle_pool[mandatory_mask[middle_pool]]
    middle_available = middle_pool[~mandatory_mask[middle_pool]]
    middle = np.concatenate((middle, rng.choice(middle_available, size=max(0, min(32 - len(middle), len(middle_available))), replace=False)))
    selected_mask = np.zeros(len(ranks), dtype=bool)
    selected_mask[best] = True
    selected_mask[middle] = True
    # Mandatory replacements consume their own tier's quota before random filling.
    extra = order[mandatory_mask[order] & ~selected_mask[order]]
    selected_mask[extra] = True
    selected = np.concatenate((best, middle, extra))
    if len(selected) > 128:
        raise ValueError("mandatory candidates exceed candidate capacity")
    remainder = order[~selected_mask[order]]
    return np.concatenate((selected, rng.choice(remainder, size=128 - len(selected), replace=False))).astype(np.uint16)


def teacher_probabilities(costs, remaining_counts, is_remaining, temperature):
    """Score ties share mass; singleton states target only the known answer."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    with torch.no_grad():
        costs = costs.detach()
        if not costs.is_floating_point():
            costs = costs.float()
        logits = -torch.log(costs / costs.amin(dim=-1, keepdim=True)) / temperature
        targets = logits.softmax(dim=-1)
        singleton = remaining_counts.eq(1).unsqueeze(-1)
        known = is_remaining.to(dtype=targets.dtype)
        return torch.where(singleton, known, targets).detach()


class TeacherDataset:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        self.words = tuple(self.manifest["words"])
        for name in ARRAY_NAMES:
            setattr(self, name, np.load(self.root / f"{name}.npy", mmap_mode="r"))
        if any(len(getattr(self, name)) != self.manifest["rows"] for name in ARRAY_NAMES):
            raise ValueError("teacher array row counts do not match manifest")

    def __len__(self):
        return len(self.lengths)


def _score_worker_init(matrix_path):
    global _WORKER_MATRIX
    _WORKER_MATRIX = np.load(matrix_path, mmap_mode="r")


def _score_chunk(items):
    return [(index, *exhaustive_scores(_WORKER_MATRIX, remaining)) for index, remaining in items]


def build_teacher_dataset(output_dir, *, source=DEFAULT_SOURCE, words=DEFAULT_WORDS,
                          seed=20260924, workers=1, expected_count=1_000_000):
    """Consume every source row, infer from the full vocabulary, and score each set once."""
    source, output_dir, words_path = Path(source), Path(output_dir), Path(words)
    dictionary = load_words(words_path)
    if len(dictionary) < 128 or len(dictionary) > np.iinfo(np.uint16).max:
        raise ValueError("dictionary must contain 128..65535 unique words")
    if expected_count < 1 or workers < 1:
        raise ValueError("expected_count and workers must be positive")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ids = {word: index for index, word in enumerate(dictionary)}
    longest_history = [{"guess": dictionary[0], "feedback": "XXXXX"}] * 5
    max_prompt_length = len(encode(POLICY_TOKEN + serialize_trajectory(longest_history)[:-len(END_TOKEN)] + GUESS_TOKEN))
    matrix = feedback_matrix(dictionary)
    np.save(output_dir / "feedback_matrix.npy", matrix)
    shapes = {
        "prompts": (np.uint8, (expected_count, max_prompt_length)),
        "lengths": (np.uint8, (expected_count,)),
        "candidate_ids": (np.uint16, (expected_count, 128)),
        "costs": (np.float64, (expected_count, 128)),
        "ranks": (np.uint16, (expected_count, 128)),
        "remaining_counts": (np.uint16, (expected_count,)),
        "is_remaining": (np.bool_, (expected_count, 128)),
        "source_ids": (np.uint16, (expected_count,)),
        "state_ids": (np.int64, (expected_count,)),
        "weights": (np.float32, (expected_count,)),
        "answer_set_ids": (np.uint32, (expected_count,)),
        "_mandatory": (np.int32, (expected_count, 9)),
    }
    arrays = {name: np.lib.format.open_memmap(output_dir / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
              for name, (dtype, shape) in shapes.items()}
    arrays["prompts"][:] = 0
    arrays["_mandatory"][:] = -1
    answer_sets = [tuple(range(len(dictionary)))]
    set_ids = {answer_sets[0]: 0}

    @lru_cache(maxsize=131072)
    def transition(set_id, guess_id, feedback):
        previous = answer_sets[set_id]
        remaining = tuple(word for word in previous if matrix[guess_id, word] == feedback)
        if not remaining:
            raise ValueError("observable history eliminates the full dictionary")
        found = set_ids.get(remaining)
        if found is None:
            found = len(answer_sets)
            set_ids[remaining] = found
            answer_sets.append(remaining)
        return found

    rows = 0
    opener = gzip.open if source.suffix == ".gz" else open
    with opener(source, "rt", encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            if rows >= expected_count:
                raise ValueError(f"source exceeds expected full count {expected_count}")
            row = json.loads(line)
            history = row["history"]
            if len(history) > 5:
                raise ValueError("pre-action history exceeds five turns")
            secret_id = ids[row["source_secret"]]
            set_id = 0
            for turn in history:
                guess_id = ids[turn["guess"]]
                feedback = turn["feedback"]
                if len(feedback) != 5 or any(mark not in "XYG" for mark in feedback):
                    raise ValueError("invalid observed feedback")
                code = 0
                for mark in feedback:
                    code = code * 3 + "XYG".index(mark)
                # The secret participates only in this truthful-history audit.
                if code == 242 or int(matrix[guess_id, secret_id]) != code:
                    raise ValueError("source history is solved or untruthful")
                set_id = transition(set_id, guess_id, code)
            remaining = answer_sets[set_id]
            if len(remaining) != row["possible_answer_count"]:
                raise ValueError("stored answer count disagrees with full-dictionary inference")
            prompt = encode(POLICY_TOKEN + serialize_trajectory(history)[:-len(END_TOKEN)] + GUESS_TOKEN)
            if len(prompt) > max_prompt_length:
                raise ValueError("serialized prompt exceeds the five-turn pre-action limit")
            arrays["prompts"][rows, :len(prompt)] = prompt
            arrays["lengths"][rows] = len(prompt)
            arrays["remaining_counts"][rows] = len(remaining)
            arrays["source_ids"][rows] = secret_id
            arrays["state_ids"][rows] = int(row["state_index"])
            weight = float(row.get("sampling_weight", 1))
            if not math.isfinite(weight) or weight <= 0:
                raise ValueError("sampling_weight must be finite and positive")
            arrays["weights"][rows] = weight
            arrays["answer_set_ids"][rows] = set_id
            top = row.get("top_guesses", [])[:8]
            arrays["_mandatory"][rows, :len(top)] = [ids[item["guess"]] for item in top]
            arrays["_mandatory"][rows, 8] = ids[row["desired_guess"]]
            rows += 1
            if rows % 100_000 == 0:
                print(json.dumps({"phase": "infer", "rows": rows, "answer_sets": len(answer_sets)}), flush=True)
    if rows != expected_count:
        raise ValueError(f"source has {rows} rows, expected the full {expected_count}")
    transition.cache_clear()
    set_ids.clear()
    cache_shape = (len(answer_sets), len(dictionary))
    numerators = np.lib.format.open_memmap(output_dir / "answer_set_numerators.npy", mode="w+", dtype=np.uint32, shape=cache_shape)
    cached_ranks = np.lib.format.open_memmap(output_dir / "answer_set_ranks.npy", mode="w+", dtype=np.uint16, shape=cache_shape)
    # Bound queued work: executor.map in older Python otherwise queues all sets.
    if workers == 1:
        for index, remaining in enumerate(answer_sets):
            numerators[index], cached_ranks[index] = exhaustive_scores(matrix, remaining)
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_score_worker_init,
                                 initargs=(str(output_dir / "feedback_matrix.npy"),)) as pool:
            for start in range(0, len(answer_sets), workers * 128):
                chunks = [list(enumerate(answer_sets[offset:offset + 128], offset))
                          for offset in range(start, min(start + workers * 128, len(answer_sets)), 128)]
                for result in pool.map(_score_chunk, chunks):
                    for index, scores, ranks in result:
                        numerators[index], cached_ranks[index] = scores, ranks
    offsets = np.zeros(len(answer_sets) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([len(remaining) for remaining in answer_sets])
    np.save(output_dir / "answer_set_offsets.npy", offsets)
    np.save(output_dir / "answer_set_members.npy", np.fromiter((word for remaining in answer_sets for word in remaining), dtype=np.uint16, count=int(offsets[-1])))
    for index in range(rows):
        set_id = int(arrays["answer_set_ids"][index])
        remaining = answer_sets[set_id]
        mandatory = arrays["_mandatory"][index]
        candidates = select_candidates(cached_ranks[set_id], remaining, mandatory[:8][mandatory[:8] >= 0],
                                       int(mandatory[8]), seed=seed,
                                       observable_key=arrays["prompts"][index, :arrays["lengths"][index]].tobytes())
        arrays["candidate_ids"][index] = candidates
        arrays["costs"][index] = numerators[set_id, candidates].astype(np.float64) / len(remaining)
        arrays["ranks"][index] = cached_ranks[set_id, candidates]
        arrays["is_remaining"][index] = np.isin(candidates, remaining)
        if (index + 1) % 100_000 == 0:
            print(json.dumps({"phase": "candidates", "rows": index + 1}), flush=True)
    for array in (*arrays.values(), numerators, cached_ranks):
        array.flush()
    del arrays["_mandatory"]
    (output_dir / "_mandatory.npy").unlink()
    manifest = {
        "schema_version": 1, "rows": rows, "seed": seed, "words": list(dictionary),
        "source": {"path": str(source), "sha256": _hash_file(source), "rows": rows, "subset": False},
        "dictionary": {"path": str(words_path), "sha256": _hash_file(words_path), "count": len(dictionary)},
        "answer_sets": len(answer_sets), "prompt_format": "<P> observable history <G>; right-padded, lengths exclude padding",
        "cost": "sum_feedback_buckets(bucket_size**2) / full_dictionary_remaining_count",
        "ranking": "exact ascending numerator; ties remaining answers then dictionary order; 1-based exhaustive ranks",
        "candidate_construction": {"count": 128, "top_ranked": 32, "middle_count": 32,
            "middle_ranks": [33, min(256, len(dictionary))], "random_remainder": 64,
            "mandatory": ["solver rank 1", "stored top 8", "desired_guess when a remaining answer"],
            "mandatory_policy": "consume middle quota first, otherwise random quota; deterministic refill",
            "rng": "numpy PCG64 seeded by first 128 bits SHA256(ascii(seed) + colon + unpadded observable prompt bytes)",
            "source_secret_used_for": "truthful-history audit and source_ids metadata only"},
        "teacher": "T>0: softmax(-log(cost/min_cost)/T); singleton: one-hot remaining answer; exact ties equal mass otherwise",
        "arrays": {},
    }
    for path in sorted(output_dir.glob("*.npy")):
        array = np.load(path, mmap_mode="r")
        manifest["arrays"][path.stem] = {"shape": list(array.shape), "dtype": str(array.dtype), "sha256": _hash_file(path)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def distribution_statistics(dataset, temperatures=(0.25, 0.5, 1.0), mode=None, chunk_size=8192):
    """Reweight saved costs only: no feedback inference or simulation is repeated."""
    groups = {"all": np.ones(len(dataset), dtype=bool)}
    mode_provenance = None
    if mode is not None:
        from cross_validation import load_mode
        evaluation_mode = load_mode(mode)
        if len(evaluation_mode.runs) != 1:
            raise ValueError("source-split temperature statistics require a development mode")
        run = evaluation_mode.runs[0]
        splits = {"train": run.train, "validation": run.validation, "test": run.test}
        mode_name = evaluation_mode.name
        ids = {word: index for index, word in enumerate(dataset.words)}
        for split, secrets in splits.items():
            groups[f"source_{split}"] = np.isin(dataset.source_ids, [ids[word] for word in secrets])
        mode_provenance = {"path": str(mode), "sha256": _hash_file(Path(mode)), "mode": mode_name}
    report = {"rows": len(dataset), "weighting": "unweighted corpus states", "mode": mode_provenance, "temperatures": {}}
    singleton = dataset.remaining_counts == 1
    for temperature in temperatures:
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        metrics = {name: np.empty(len(dataset), dtype=np.float64) for name in
                   ("entropy", "rank1_mass", "top3_mass", "top8_mass", "effective_candidates", "count_mass_ge_0.01")}
        for start in range(0, len(dataset), chunk_size):
            stop = min(start + chunk_size, len(dataset))
            costs = np.asarray(dataset.costs[start:stop])
            logits = -np.log(costs / costs.min(axis=1, keepdims=True)) / temperature
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            one = singleton[start:stop]
            probabilities[one] = dataset.is_remaining[start:stop][one]
            logp = np.zeros_like(probabilities)
            np.log(probabilities, out=logp, where=probabilities > 0)
            entropy = -(probabilities * logp).sum(axis=1)
            metrics["entropy"][start:stop] = entropy
            metrics["effective_candidates"][start:stop] = np.exp(entropy)
            metrics["count_mass_ge_0.01"][start:stop] = (probabilities >= 0.01).sum(axis=1)
            for cutoff in (1, 3, 8):
                metrics[f"{'rank1' if cutoff == 1 else f'top{cutoff}'}_mass"][start:stop] = (probabilities * (dataset.ranks[start:stop] <= cutoff)).sum(axis=1)
        summaries = {}
        for name, mask in groups.items():
            summaries[name] = {}
            for population, selected in (("all", mask), ("singleton", mask & singleton), ("non_singleton", mask & ~singleton)):
                count = int(selected.sum())
                summaries[name][population] = {"count": count, **{
                    key: {"mean": float(values[selected].mean()) if count else None,
                          "median": float(np.median(values[selected])) if count else None}
                    for key, values in metrics.items()}}
        report["temperatures"][str(temperature)] = summaries
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--workers", type=int, default=max(1, min(8, os.cpu_count() or 1)))
    parser.add_argument("--expected-count", type=int, default=1_000_000)
    parser.add_argument("--mode", type=Path)
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    parser.add_argument("--stats-only", action="store_true")
    args = parser.parse_args(argv)
    if not args.stats_only:
        build_teacher_dataset(args.output_dir, source=args.source, words=args.words, seed=args.seed,
                              workers=args.workers, expected_count=args.expected_count)
    report = distribution_statistics(TeacherDataset(args.output_dir), args.temperatures, args.mode)
    (args.output_dir / "teacher-statistics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
