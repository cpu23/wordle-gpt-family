from __future__ import annotations

import math
import statistics
from collections.abc import Sequence

import torch

from action_regret import analyze_action_regret
from evaluate_v2 import GameplaySummary, evaluate_model
from model import WordleGPT
from wordle import filter_answers, score_guess


def _gameplay_payload(summary: GameplaySummary) -> dict[str, object]:
    results: list[dict[str, object]] = []
    repeated_guess_count = 0
    games_with_repeated_guesses = 0
    winning_attempts: list[int] = []

    for result in summary.results:
        attempts = len(result.guesses)
        repeats = attempts - len(set(result.guesses))
        repeated_guess_count += repeats
        games_with_repeated_guesses += repeats > 0
        if result.won:
            winning_attempts.append(attempts)
        results.append({
            "secret": result.secret,
            "guesses": list(result.guesses),
            "won": result.won,
            "attempts": attempts,
            "winning_guess": result.guesses[-1] if result.won else None,
            "winning_attempts": attempts if result.won else None,
            "invalid_guesses": result.invalid_guesses,
            "repeated_guess_count": repeats,
        })

    return {
        "checkpoint": summary.checkpoint,
        "decode": summary.decode,
        "games": summary.games,
        "wins": summary.wins,
        "win_rate": summary.win_rate,
        "average_attempts": summary.average_attempts,
        "average_guesses_to_win": summary.average_guesses,
        "winning_attempts": winning_attempts,
        "invalid_guesses": summary.invalid_guesses,
        "games_with_invalid_guesses": sum(
            result.invalid_guesses > 0 for result in summary.results
        ),
        "repeated_guess_count": repeated_guess_count,
        "games_with_repeated_guesses": games_with_repeated_guesses,
        "results": results,
    }


def _replay_candidate_rewards(
    gameplay: dict[str, object],
    words: tuple[str, ...],
    solve_bonus: float,
) -> dict[str, object]:
    allowed = frozenset(words)
    all_actions: list[dict[str, object]] = []
    game_traces: list[dict[str, object]] = []

    for result in gameplay["results"]:
        secret = str(result["secret"])
        possible = words
        actions: list[dict[str, object]] = []
        legal_rewards: list[float] = []
        for turn, guess_value in enumerate(result["guesses"], start=1):
            guess = str(guess_value)
            candidate_count_before = len(possible)
            if guess not in allowed:
                action = {
                    "secret": secret,
                    "turn": turn,
                    "guess": guess,
                    "legal": False,
                    "feedback": None,
                    "solved": False,
                    "candidate_count_before": candidate_count_before,
                    "candidate_count_after": None,
                    "absolute_reduction": None,
                    "proportional_reduction": None,
                    "log_reduction": None,
                    "solve_bonus_awarded": None,
                    "reward": None,
                }
                actions.append(action)
                all_actions.append(action)
                break

            feedback = score_guess(secret, guess)
            remaining = filter_answers(possible, guess, feedback)
            candidate_count_after = len(remaining)
            if candidate_count_after == 0:
                raise RuntimeError(
                    "true Wordle feedback eliminated its validation secret from "
                    "the candidate universe"
                )
            absolute_reduction = candidate_count_before - candidate_count_after
            proportional_reduction = absolute_reduction / candidate_count_before
            log_reduction = math.log(candidate_count_before / candidate_count_after)
            solved = guess == secret
            solve_bonus_awarded = solve_bonus if solved else 0.0
            reward = log_reduction + solve_bonus_awarded
            action = {
                "secret": secret,
                "turn": turn,
                "guess": guess,
                "legal": True,
                "feedback": feedback,
                "solved": solved,
                "candidate_count_before": candidate_count_before,
                "candidate_count_after": candidate_count_after,
                "absolute_reduction": absolute_reduction,
                "proportional_reduction": proportional_reduction,
                "log_reduction": log_reduction,
                "solve_bonus_awarded": solve_bonus_awarded,
                "reward": reward,
            }
            actions.append(action)
            all_actions.append(action)
            legal_rewards.append(reward)
            possible = remaining
            if solved:
                break

        game_traces.append({
            "secret": secret,
            "won": bool(result["won"]),
            "attempts": int(result["attempts"]),
            "rewarded_actions": len(legal_rewards),
            "total_reward": sum(legal_rewards) if legal_rewards else None,
            "actions": actions,
        })

    legal_actions = [action for action in all_actions if action["legal"]]
    proportional_reductions = [
        float(action["proportional_reduction"]) for action in legal_actions
    ]
    log_reductions = [float(action["log_reduction"]) for action in legal_actions]
    rewards = [float(action["reward"]) for action in legal_actions]
    return {
        "summary": {
            "games": len(game_traces),
            "actions": len(legal_actions),
            "illegal_actions": len(all_actions) - len(legal_actions),
            "mean_absolute_reduction": (
                statistics.fmean(int(action["absolute_reduction"]) for action in legal_actions)
                if legal_actions else 0.0
            ),
            "mean_proportional_reduction": (
                statistics.fmean(proportional_reductions) if proportional_reductions else 0.0
            ),
            "mean_log_reduction": statistics.fmean(log_reductions) if log_reductions else 0.0,
            "total_solve_bonus_awarded": sum(
                float(action["solve_bonus_awarded"]) for action in legal_actions
            ),
            "total_reward": sum(rewards),
            "mean_reward_per_legal_action": statistics.fmean(rewards) if rewards else 0.0,
            "mean_reward_per_game": (
                sum(rewards) / len(game_traces) if game_traces else 0.0
            ),
        },
        "games": game_traces,
        "actions": all_actions,
    }


def evaluate_grpo(
    model: WordleGPT,
    validation_secrets: Sequence[str],
    words: Sequence[str],
    *,
    solve_bonus: float = 5.0,
    checkpoint: str = "in-memory",
) -> dict[str, object]:
    """Evaluate raw and constrained Wordle gameplay and replay GRPO rewards.

    Candidate reductions are measured against the full ``words`` universe and
    use true feedback for each validation secret. Per-action ``reward`` is the
    natural-log candidate reduction plus ``solve_bonus`` on a winning guess;
    proportional reduction is reported separately. Illegal guesses have no
    feedback, reductions, or reward.
    """
    validation = tuple(validation_secrets)
    candidate_words = tuple(dict.fromkeys(words))
    if not validation:
        raise ValueError("validation_secrets must contain at least one secret")
    if not candidate_words:
        raise ValueError("words must contain the full non-empty candidate universe")
    candidate_set = frozenset(candidate_words)
    missing_secrets = [secret for secret in validation if secret not in candidate_set]
    if missing_secrets:
        raise ValueError("every validation secret must be present in words")

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            raw_summary = evaluate_model(
                model, validation, candidate_words, checkpoint=checkpoint, decode="raw"
            )
            constrained_summary = evaluate_model(
                model, validation, candidate_words, checkpoint=checkpoint, decode="constrained"
            )
    finally:
        model.train(was_training)

    raw_gameplay = _gameplay_payload(raw_summary)
    constrained_gameplay = _gameplay_payload(constrained_summary)
    raw_rewards = _replay_candidate_rewards(raw_gameplay, candidate_words, solve_bonus)
    constrained_rewards = _replay_candidate_rewards(
        constrained_gameplay, candidate_words, solve_bonus
    )
    raw_regret = analyze_action_regret(raw_gameplay, candidate_words)
    constrained_regret = analyze_action_regret(constrained_gameplay, candidate_words)

    return {
        "checkpoint": checkpoint,
        "solve_bonus": solve_bonus,
        "candidate_universe_size": len(candidate_words),
        "reward_definition": (
            "For each legal action, reward = ln(candidate_count_before / "
            "candidate_count_after) + solve_bonus if the guess solves the game. "
            "Illegal actions receive no feedback, candidate reduction, or reward."
        ),
        "gameplay": {
            "raw": raw_gameplay,
            "constrained": constrained_gameplay,
        },
        "candidate_reduction": {
            "definition": (
                "Candidate sets start from the full words universe and are filtered "
                "using the validation secret's true feedback after each legal guess."
            ),
            "raw": raw_rewards,
            "constrained": constrained_rewards,
        },
        "action_regret": {
            "definition": (
                "Expected-survivor regret is the selected legal guess's expected "
                "remaining-candidate count minus the minimum expected count among "
                "all allowed words for the same candidate set; it is a heuristic "
                "diagnostic, not the GRPO reward."
            ),
            "raw": raw_regret,
            "constrained": constrained_regret,
        },
    }
