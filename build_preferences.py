from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import io
import json
from collections import Counter
from pathlib import Path
from typing import Iterator, Mapping, Sequence

from wordle import DEFAULT_WORDS, load_words

RULES = {
    "clear": {"chosen_ranks": [1, 2], "rejected_ranks": [5, 6, 7, 8], "minimum_score_ratio": 1.25},
    "hard": {"chosen_ranks": [1, 2], "rejected_ranks": [3, 4], "require_strictly_better": True},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preference_candidates(record: Mapping[str, object], allowed_words: set[str]) -> Iterator[dict[str, object]]:
    text = str(record["text"])
    desired = str(record["desired_guess"])
    suffix = desired + "<E>"
    if not text.endswith(suffix):
        raise ValueError("expert record text does not end in its desired completion")
    prompt = text[: -len(suffix)]
    if not prompt.endswith("<G>"):
        raise ValueError("preference prompt must terminate at the next-guess marker")
    guesses = record.get("top_guesses")
    if not isinstance(guesses, list):
        return
    for pair_type, rule in RULES.items():
        for chosen_rank in rule["chosen_ranks"]:
            for rejected_rank in rule["rejected_ranks"]:
                if max(chosen_rank, rejected_rank) > len(guesses):
                    continue
                chosen = guesses[chosen_rank - 1]
                rejected = guesses[rejected_rank - 1]
                chosen_guess, rejected_guess = str(chosen["guess"]), str(rejected["guess"])
                chosen_score = float(chosen["expected_survivors"])
                rejected_score = float(rejected["expected_survivors"])
                if chosen_guess == rejected_guess or chosen_guess not in allowed_words or rejected_guess not in allowed_words:
                    continue
                if pair_type == "clear" and rejected_score < chosen_score * 1.25:
                    continue
                if pair_type == "hard" and not rejected_score > chosen_score:
                    continue
                difference = rejected_score - chosen_score
                yield {
                    "schema_version": 1,
                    "pair_type": pair_type,
                    "split": record["split"],
                    "prompt": prompt,
                    "chosen_guess": chosen_guess,
                    "rejected_guess": rejected_guess,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "chosen_rank": chosen_rank,
                    "rejected_rank": rejected_rank,
                    "score_ratio": rejected_score / chosen_score,
                    "score_difference": difference,
                    "relative_score_gap": difference / chosen_score,
                    "source_state_id": record["state_fingerprint"],
                    "source_state_index": record["state_index"],
                    "source_secret": record["source_secret"],
                    "remaining_answer_count": record["possible_answer_count"],
                    "top_guesses": guesses,
                }


def _priority(seed: int, pair: Mapping[str, object]) -> int:
    key = f"{seed}|{pair['prompt']}|{pair['chosen_guess']}|{pair['rejected_guess']}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
def deterministic_gzip_writer(path: Path):
    raw = path.open("wb")
    compressed = gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0)
    return io.TextIOWrapper(compressed, encoding="utf-8")




def build_preferences(
    source: str | Path,
    output_dir: str | Path,
    *,
    allowed_words: Sequence[str],
    seed: int = 0,
    train_target: int = 450_000,
    validation_target: int = 50_000,
    clear_fraction: float = 0.8,
) -> dict[str, object]:
    if train_target < 1 or validation_target < 1 or not 0 < clear_fraction < 1:
        raise ValueError("targets must be positive and clear_fraction must be between zero and one")
    source, output = Path(source), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    targets = {"train": train_target, "validation": validation_target}
    quotas = {
        split: {"clear": round(target * clear_fraction), "hard": target - round(target * clear_fraction)}
        for split, target in targets.items()
    }
    legal_words = set(allowed_words)
    heaps: dict[tuple[str, str], list[tuple[int, int, dict[str, object]]]] = {(s, t): [] for s in targets for t in RULES}
    eligible = Counter()
    seen: set[bytes] = set()
    sequence = 0
    with gzip.open(source, "rt", encoding="utf-8") as records:
        for line in records:
            record = json.loads(line)
            split = record.get("split")
            if split not in targets:
                continue
            for pair in preference_candidates(record, legal_words):
                identity = hashlib.sha256(
                    f"{pair['prompt']}|{pair['chosen_guess']}|{pair['rejected_guess']}".encode()
                ).digest()[:16]
                if identity in seen:
                    continue
                seen.add(identity)
                pair_type = str(pair["pair_type"])
                eligible[(split, pair_type)] += 1
                priority = _priority(seed, pair)
                heap = heaps[(split, pair_type)]
                item = (-priority, sequence, pair)
                sequence += 1
                quota = quotas[split][pair_type]
                if len(heap) < quota:
                    heapq.heappush(heap, item)
                elif priority < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    counts: dict[str, dict[str, int]] = {}
    prompt_counts: dict[str, int] = {}
    for split in targets:
        selected = [item for pair_type in RULES for item in heaps[(split, pair_type)]]
        selected.sort(key=lambda item: (_priority(seed + 1, item[2]), item[1]))
        path = output / f"{split}.jsonl.gz"
        prompts: set[str] = set()
        by_type = Counter()
        with deterministic_gzip_writer(path) as destination:
            for _, _, pair in selected:
                destination.write(json.dumps(pair, sort_keys=True) + "\n")
                prompts.add(str(pair["prompt"]))
                by_type[str(pair["pair_type"])] += 1
        counts[split] = {"pairs": len(selected), **dict(by_type)}
        prompt_counts[split] = len(prompts)
    manifest = {
        "schema_version": 1,
        "source": str(source),
        "source_sha256": sha256(source),
        "allowed_words_sha256": hashlib.sha256("\n".join(allowed_words).encode()).hexdigest(),
        "seed": seed,
        "rules": RULES,
        "requested_targets": targets,
        "requested_clear_fraction": clear_fraction,
        "eligible_pairs": {f"{split}_{kind}": eligible[(split, kind)] for split in targets for kind in RULES},
        "counts": counts,
        "unique_prompts": prompt_counts,
        "unique_pairs": sum(value["pairs"] for value in counts.values()),
        "files": {split: f"{split}.jsonl.gz" for split in targets},
        "file_sha256": {
            split: sha256(output / f"{split}.jsonl.gz")
            for split in targets
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Build deterministic solver-ranked Wordle DPO pairs.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-target", type=int, default=450_000)
    parser.add_argument("--validation-target", type=int, default=50_000)
    args = parser.parse_args()
    print(json.dumps(build_preferences(args.source, args.output_dir, allowed_words=load_words(args.words), seed=args.seed, train_target=args.train_target, validation_target=args.validation_target), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
