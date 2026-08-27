from __future__ import annotations

import json
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path

from wordle import expected_survivors, filter_answers, score_guess, top_informative_guesses


def analyze_action_regret(
    gameplay: Mapping[str, object],
    allowed_words: Sequence[str],
) -> dict[str, object]:
    actions: list[dict[str, object]] = []
    allowed = tuple(allowed_words)
    for game in gameplay["results"]:
        secret = str(game["secret"])
        possible = allowed
        for turn, guess_value in enumerate(game["guesses"], start=1):
            guess = str(guess_value)
            if guess not in allowed:
                actions.append({"secret": secret, "turn": turn, "guess": guess, "legal": False})
                break
            top = top_informative_guesses(possible, allowed, 8)
            model_score = expected_survivors(possible, guess)
            ranked = {candidate: rank for rank, (candidate, _) in enumerate(top, start=1)}
            optimal_score = top[0][1]
            actions.append({
                "secret": secret, "turn": turn, "guess": guess, "legal": True,
                "remaining_answer_count": len(possible), "model_score": model_score,
                "optimal_score": optimal_score, "regret": model_score - optimal_score,
                "selected_rank": ranked.get(guess), "top_guesses": [
                    {"guess": candidate, "expected_survivors": score, "rank": rank}
                    for rank, (candidate, score) in enumerate(top, start=1)
                ],
            })
            if guess == secret:
                break
            possible = filter_answers(possible, guess, score_guess(secret, guess))
    legal_actions = [action for action in actions if action.get("legal")]
    regrets = [float(action["regret"]) for action in legal_actions]
    ranks = [action.get("selected_rank") for action in legal_actions]
    count = len(legal_actions)
    summary = {
        "games": len(gameplay["results"]), "actions": count,
        "illegal_actions": len(actions) - count,
        "mean_action_regret": statistics.fmean(regrets) if regrets else 0.0,
        "median_action_regret": statistics.median(regrets) if regrets else 0.0,
        "rank_1_fraction": sum(rank == 1 for rank in ranks) / count if count else 0.0,
        "top_3_fraction": sum(rank is not None and rank <= 3 for rank in ranks) / count if count else 0.0,
        "top_8_fraction": sum(rank is not None for rank in ranks) / count if count else 0.0,
    }
    return {"summary": summary, "actions": actions}


def analyze_file(gameplay_path: str | Path, output_path: str | Path, allowed_words: Sequence[str]) -> dict[str, object]:
    result = analyze_action_regret(json.loads(Path(gameplay_path).read_text(encoding="utf-8")), allowed_words)
    Path(output_path).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result
