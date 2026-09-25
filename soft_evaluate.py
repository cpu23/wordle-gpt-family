from __future__ import annotations

import math
import re
import statistics
from collections.abc import Sequence
from dataclasses import asdict
from functools import lru_cache
from itertools import zip_longest

import torch

from evaluate_v2 import evaluate_model
from experiments_v2 import evaluate_objective_loss
from tokenizer import FEEDBACK_TO_SYMBOL, GUESS_TOKEN
from tokenizer_v2 import POLICY_TOKEN, decode, encode
from train import generate_constrained_guess
from wordle import filter_answers, top_informative_guesses

_SYMBOL_TO_FEEDBACK = {symbol: mark for mark, symbol in FEEDBACK_TO_SYMBOL.items()}
_HISTORY_TURN = re.compile(r"([a-z]{5})<F>([012]{5})<G>")


def _consistent_answers(prompt: str, words: tuple[str, ...]) -> tuple[str, ...]:
    """Recover the full answer set using only serialized observable feedback."""
    prefix = POLICY_TOKEN + GUESS_TOKEN
    if not prompt.startswith(prefix):
        raise ValueError("evaluation requires an observable policy prompt")
    history = prompt[len(prefix) :]
    possible = words
    offset = 0
    while offset < len(history):
        turn = _HISTORY_TURN.match(history, offset)
        if turn is None:
            raise ValueError("malformed policy history")
        guess, symbols = turn.groups()
        feedback = "".join(_SYMBOL_TO_FEEDBACK[symbol] for symbol in symbols)
        possible = filter_answers(possible, guess, feedback)
        offset = turn.end()
    if not possible:
        raise ValueError("observable policy history has no consistent answers")
    return possible


@lru_cache(maxsize=1024)
def _exhaustive_ranking(
    possible: tuple[str, ...], words: tuple[str, ...]
) -> tuple[tuple[str, float], ...]:
    # Requesting the entire dictionary preserves the solver's exact score and
    # remaining-answer-first/dictionary-order tie breaking, including singletons.
    return tuple(top_informative_guesses(possible, words, len(words)))


def _score_action(prompt: str, guess: str, words: tuple[str, ...]) -> dict[str, object]:
    possible = _consistent_answers(prompt, words)
    ranking = _exhaustive_ranking(possible, words)
    selected_rank = next(rank for rank, (word, _) in enumerate(ranking, 1) if word == guess)
    selected_cost = ranking[selected_rank - 1][1]
    best_cost = ranking[0][1]
    return {
        "guess": guess,
        "remaining_answer_count": len(possible),
        "selected_rank": selected_rank,
        "selected_cost": selected_cost,
        "best_cost": best_cost,
        "regret": selected_cost - best_cost,
        "relative_quality": selected_cost / best_cost,
        "top_guesses": [
            {"guess": word, "expected_survivors": cost, "rank": rank}
            for rank, (word, cost) in enumerate(ranking[:8], 1)
        ],
    }


def _action_summary(actions: list[dict[str, object]]) -> dict[str, object]:
    regrets = [float(action["regret"]) for action in actions]
    relative = [float(action["relative_quality"]) for action in actions]
    ranks = [int(action["selected_rank"]) for action in actions]
    count = len(actions)
    return {
        "actions": count,
        "mean_action_regret": statistics.fmean(regrets),
        "median_action_regret": statistics.median(regrets),
        "rank_1_fraction": sum(rank == 1 for rank in ranks) / count,
        "top_3_fraction": sum(rank <= 3 for rank in ranks) / count,
        "top_8_fraction": sum(rank <= 8 for rank in ranks) / count,
        "mean_relative_quality": statistics.fmean(relative),
        "median_relative_quality": statistics.median(relative),
    }


def _distribution_positions(remaining_counts: Sequence[int], limit: int) -> set[int]:
    """Balance observable-state sizes, taking nonsingletons first each round."""
    buckets: list[list[int]] = [[] for _ in range(5)]
    for position, count in enumerate(remaining_counts):
        bucket = 4 if count == 1 else 0 if count <= 5 else 1 if count <= 20 else 2 if count <= 100 else 3
        buckets[bucket].append(position)
    selected: set[int] = set()
    for positions in zip_longest(*buckets):
        for position in positions:
            if len(selected) >= limit:
                return selected
            if position is not None:
                selected.add(position)
    return selected


def evaluate_soft_model(
    model,
    dataset,
    indices: Sequence[int],
    temperature: float,
    secrets: Sequence[str],
    words: Sequence[str],
    mechanics_validation,
    device: str | torch.device,
    *,
    batch_size: int = 4,
    distribution_examples: int = 12,
) -> dict[str, object]:
    """Evaluate the complete caller-selected panel and supplied validation games.

    Candidate probabilities condition raw-vocabulary five-letter sequence
    probabilities on the cached candidate set. Action quality instead evaluates
    full-dictionary constrained token-greedy actions, never candidate argmaxes.
    Neither teacher targets nor action scoring inspect source-secret metadata.
    """
    # Keep the evaluator importable independently of cache/training construction.
    from soft_policy import candidate_sequence_logps, distillation_loss
    from soft_teacher import teacher_probabilities

    panel = [int(index) for index in indices]
    words = tuple(words)
    if not panel:
        raise ValueError("validation panel must not be empty")
    if any(index < 0 or index >= len(dataset) for index in panel):
        raise IndexError("validation panel index outside teacher dataset")
    if batch_size < 1 or distribution_examples < 0:
        raise ValueError("batch_size must be positive and distribution_examples nonnegative")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if not secrets:
        raise ValueError("validation gameplay secrets must not be empty")
    if words != tuple(dataset.words):
        raise ValueError("evaluation words must match the full teacher dictionary in order")

    allowed = frozenset(words)
    word_tokens = torch.tensor([encode(word) for word in words], dtype=torch.long, device=device)
    metric_totals: dict[str, float] = {}
    actions: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []
    example_positions = _distribution_positions(dataset.remaining_counts[panel], distribution_examples)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            for start in range(0, len(panel), batch_size):
                batch_indices = panel[start : start + batch_size]
                lengths = torch.tensor(dataset.lengths[batch_indices], dtype=torch.long, device=device)
                prompts = torch.tensor(
                    dataset.prompts[batch_indices, : int(lengths.max().item())],
                    dtype=torch.long,
                    device=device,
                )
                candidate_ids = torch.tensor(dataset.candidate_ids[batch_indices], dtype=torch.long, device=device)
                costs = torch.tensor(dataset.costs[batch_indices], dtype=torch.float32, device=device)
                ranks = torch.tensor(dataset.ranks[batch_indices], dtype=torch.long, device=device)
                remaining_counts = torch.tensor(dataset.remaining_counts[batch_indices], dtype=torch.long, device=device)
                is_remaining = torch.tensor(dataset.is_remaining[batch_indices], dtype=torch.bool, device=device)
                targets = teacher_probabilities(costs, remaining_counts, is_remaining, temperature)
                sequence_logps = candidate_sequence_logps(model, prompts, lengths, word_tokens[candidate_ids])
                _, metrics = distillation_loss(sequence_logps, targets, ranks)
                for key, value in metrics.items():
                    metric_totals[key] = metric_totals.get(key, 0.0) + float(value.item()) * len(batch_indices)

                student = sequence_logps.softmax(dim=-1)
                prompt_rows = prompts.cpu().tolist()
                length_rows = lengths.cpu().tolist()
                # Transfer complete distributions only for the fixed example panel.
                example_rows = [row for row in range(len(batch_indices)) if start + row in example_positions]
                example_offsets = {row: offset for offset, row in enumerate(example_rows)}
                if example_rows:
                    example_arrays = [
                        tensor[example_rows].cpu().tolist()
                        for tensor in (candidate_ids, costs, ranks, is_remaining, targets, student, sequence_logps)
                    ]
                for row, index in enumerate(batch_indices):
                    prefix = prompt_rows[row][: length_rows[row]]
                    prompt = decode(prefix)
                    guess = decode(generate_constrained_guess(model, prefix, allowed))
                    action = _score_action(prompt, guess, words)
                    action.update({"dataset_index": index, "state_id": int(dataset.state_ids[index]), "prompt": prompt})
                    actions.append(action)
                    if row in example_offsets:
                        ids, row_costs, row_ranks, remaining, teacher, probabilities, logps = (
                            values[example_offsets[row]] for values in example_arrays
                        )
                        examples.append({
                            "dataset_index": index,
                            "state_id": int(dataset.state_ids[index]),
                            "prompt": prompt,
                            "remaining_answer_count": action["remaining_answer_count"],
                            "candidates": [
                                {
                                    "word": words[word_id], "id": word_id,
                                    "rank": rank, "cost": cost, "is_remaining": remains,
                                    "teacher_probability": target, "student_probability": probability,
                                    "sequence_logp": logp,
                                }
                                for word_id, cost, rank, remains, target, probability, logp in zip(
                                    ids, row_costs, row_ranks, remaining, teacher, probabilities, logps
                                )
                            ],
                        })
            gameplay = asdict(evaluate_model(model, secrets, words, decode="constrained"))
            mechanics_loss = evaluate_objective_loss(model, mechanics_validation)
    finally:
        model.train(was_training)

    return {
        "temperature": float(temperature),
        "panel_indices": panel,
        "gameplay": gameplay,
        "mechanics_validation_loss": mechanics_loss,
        "policy": {key: total / len(panel) for key, total in metric_totals.items()},
        "action_quality": {"summary": _action_summary(actions), "actions": actions},
        "distribution_examples": examples,
    }


def checkpoint_key(report: dict[str, object]) -> tuple[float, float, float, float, float]:
    """Maximize gameplay first, then exhaustive regret, then teacher/student KL."""
    gameplay = report["gameplay"]
    return (
        int(gameplay["wins"]),
        -float(gameplay["average_attempts"]),
        -float(gameplay["average_guesses"]),
        -float(report["action_quality"]["summary"]["mean_action_regret"]),
        -float(report["policy"]["teacher_student_kl"]),
    )
