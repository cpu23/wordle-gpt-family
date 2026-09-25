from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cross_validation import load_mode
from evaluate_v2 import evaluate_model, load_v2_model
from experiments_v2 import evaluate_objective_loss
from action_regret import analyze_action_regret
from train_dpo import evaluate_preferences, load_preferences, reference_logps, train_dpo
from train_v2 import load_v2_split
from wordle import DEFAULT_WORDS, load_words

BETA = 0.20
LAMBDAS = (0.0, 0.1, 0.5, 1.0)
PRIMARY_LR = 3e-6
FOLLOWUP_LR = 1e-6
SEED = 0
EVALUATION_PASSES = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

def write_development_table(path: Path, rows: list[dict[str, object]]) -> None:
    headers = (
        "Beta", "Lambda SFT", "LR", "Pass", "Constrained wins", "Raw wins", "Invalid",
        "Pref acc", "Chosen Δlogp", "Rejected Δlogp", "Mean regret", "Mechanics loss",
    )
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---:" for _ in headers) + "|"]
    for row in rows:
        preference = row.get("validation", row.get("preference", {}))
        regret = row["action_regret"]
        values = (
            "SFT" if row["beta"] is None else f"{float(row['beta']):.2f}",
            "—" if row["lambda_sft"] is None else f"{float(row['lambda_sft']):.1f}",
            "—" if row["learning_rate"] is None else f"{float(row['learning_rate']):.0e}",
            f"{float(row['pass']):.2f}",
            str(row["constrained_wins"]), str(row["raw_wins"]), str(row["raw_invalid_guesses"]),
            f"{float(preference['preference_accuracy']):.4f}",
            f"{float(preference['chosen_delta_logp']):.4f}",
            f"{float(preference['rejected_delta_logp']):.4f}",
            f"{float(regret['mean_action_regret']):.6f}",
            f"{float(row['mechanics_validation_loss']):.6f}",
        )
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gameplay_key(record: dict[str, object]) -> tuple[float, ...]:
    return (
        float(record["constrained_wins"]),
        -float(record["constrained_average_attempts"]),
        -float(record["constrained_average_guesses"]),
        float(record["raw_wins"]),
        -float(record["raw_invalid_guesses"]),
    )


def run_experiment(
    preference_dir: Path,
    mechanics_dir: Path,
    base_checkpoint: Path,
    mode_path: Path,
    output: Path,
    words: list[str],
    device: str,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=True)
    train_path = preference_dir / "train.jsonl.gz"
    validation_path = preference_dir / "validation.jsonl.gz"
    train_preferences = load_preferences(train_path)
    validation_preferences = load_preferences(validation_path)
    mechanics_validation = load_v2_split(mechanics_dir, "validation", example_type="mechanics")
    mode = load_mode(mode_path)
    run = mode.runs[0]
    if len(run.validation) != 72:
        raise RuntimeError(f"expected 72 development validation secrets, found {len(run.validation)}")

    baseline_model = load_v2_model(base_checkpoint, device)
    parameter_count = sum(parameter.numel() for parameter in baseline_model.parameters())
    if parameter_count != 7_162_403:
        raise RuntimeError(f"expected the 7,162,403-parameter checkpoint, found {parameter_count:,}")
    references = reference_logps(baseline_model, validation_preferences, 256, torch.device(device))
    baseline_preference = evaluate_preferences(
        baseline_model, validation_preferences, references, BETA, 256, torch.device(device)
    )
    baseline_mechanics = evaluate_objective_loss(baseline_model, mechanics_validation, batch_size=256)
    baseline_constrained = evaluate_model(baseline_model, run.validation, words, decode="constrained")
    baseline_raw = evaluate_model(baseline_model, run.validation, words, decode="raw")
    baseline_regret = analyze_action_regret(asdict(baseline_constrained), words)
    baseline = {
        "label": "sft", "beta": None, "lambda_sft": None, "learning_rate": None,
        "pass": 0.0, "constrained_wins": baseline_constrained.wins,
        "constrained_average_attempts": baseline_constrained.average_attempts,
        "constrained_average_guesses": baseline_constrained.average_guesses,
        "raw_wins": baseline_raw.wins, "raw_invalid_guesses": baseline_raw.invalid_guesses,
        "preference": asdict(baseline_preference), "mechanics_validation_loss": baseline_mechanics,
        "action_regret": baseline_regret["summary"],
    }
    baseline_dir = output / "sft-baseline"
    write_json(baseline_dir / "validation-constrained.json", asdict(baseline_constrained))
    write_json(baseline_dir / "validation-raw.json", asdict(baseline_raw))
    write_json(baseline_dir / "action-regret.json", baseline_regret)
    write_json(baseline_dir / "summary.json", baseline)
    del baseline_model
    if device == "cuda":
        torch.cuda.empty_cache()

    manifest = {
        "experiment": "dpo-rescue-chosen-likelihood-anchor", "beta": BETA,
        "lambdas": list(LAMBDAS), "primary_learning_rate": PRIMARY_LR,
        "conditional_followup_learning_rate": FOLLOWUP_LR, "seed": SEED,
        "evaluation_passes": list(EVALUATION_PASSES), "effective_batch_size": 128,
        "base_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": sha256(base_checkpoint),
        "parameter_count": parameter_count, "train_preferences_sha256": sha256(train_path),
        "validation_preferences_sha256": sha256(validation_path),
        "preference_manifest": json.loads((preference_dir / "manifest.json").read_text(encoding="utf-8")),
        "validation_secrets": list(run.validation), "test_evaluated": False,
    }
    write_json(output / "manifest.json", manifest)

    histories: dict[str, list[dict[str, object]]] = {}

    def train_configuration(lambda_sft: float, learning_rate: float, replay: float = 0.0) -> None:
        suffix = f"-mechanics-{replay:.2f}" if replay else ""
        label = f"lambda-{lambda_sft:.1f}-lr-{learning_rate:.0e}{suffix}"
        run_dir = output / label
        complete = run_dir / "training-complete.json"
        metrics_path = run_dir / "metrics.jsonl"
        if not complete.exists():
            checkpoint, records = train_dpo(
                train_preferences, validation_preferences, mechanics_validation, run_dir,
                base_checkpoint=base_checkpoint, validation_secrets=run.validation,
                allowed_words=words, beta=BETA, lambda_sft=lambda_sft, learning_rate=learning_rate,
                evaluation_passes=EVALUATION_PASSES, collapse_wins=max(0, baseline_constrained.wins - 10),
                seed=SEED, device=device,
                mechanics_train=load_v2_split(mechanics_dir, "train", example_type="mechanics") if replay else None,
                mechanics_replay_fraction=replay,
            )
            write_json(complete, {"checkpoint": str(checkpoint), "evaluations": len(records)})
        if device == "cuda":
            torch.cuda.empty_cache()
        checkpoint_path = run_dir / "checkpoints" / "best.pt"
        histories[label] = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines()]
        write_json(complete, {
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": sha256(checkpoint_path),
            "metrics": str(metrics_path),
            "metrics_sha256": sha256(metrics_path),
            "evaluations": len(histories[label]),
        })

    nonzero_best = max(
        ((label, record) for label, history in histories.items() if not label.startswith("lambda-0.0")
         for record in history if float(record["effective_passes"]) > 0),
        key=lambda item: gameplay_key(item[1]),
    )
    best_lambda = float(nonzero_best[0].split("-")[1])
    train_configuration(best_lambda, FOLLOWUP_LR)

    candidates = [(label, record) for label, history in histories.items() for record in history if float(record["effective_passes"]) > 0]
    best_label, best_record = max(candidates, key=lambda item: gameplay_key(item[1]))
    mechanics_damage = float(best_record["mechanics_validation_loss"]) > baseline_mechanics * 1.20
    if mechanics_damage and not best_label.startswith("lambda-0.0"):
        parts = best_label.split("-")
        train_configuration(float(parts[1]), float(parts[3]), replay=0.05)

    rows: list[dict[str, object]] = [baseline]
    for label, history in histories.items():
        run_config = json.loads((output / label / "run.json").read_text(encoding="utf-8"))
        for record in history:
            rows.append({"label": label, "beta": BETA, "lambda_sft": run_config["lambda_sft"],
                         "learning_rate": run_config["learning_rate"], "pass": record["effective_passes"], **record})
    trained = [row for row in rows[1:] if float(row["effective_passes"]) > 0]
    best = max(trained, key=gameplay_key)
    baseline_key = gameplay_key(baseline)
    gameplay_beaten = gameplay_key(best) > baseline_key
    regret_improved = (
        best["constrained_wins"] == baseline["constrained_wins"]
        and float(best["action_regret"]["mean_action_regret"]) < float(baseline["action_regret"]["mean_action_regret"]) * 0.95
        and float(best["validation"]["chosen_delta_logp"]) > -5.0
    )
    if gameplay_beaten:
        interpretation = "anchor fixes gameplay"
    elif regret_improved:
        interpretation = "anchor preserves gameplay and materially improves action regret"
    elif int(best["constrained_wins"]) >= int(baseline["constrained_wins"]):
        interpretation = "anchor preserves but does not improve gameplay"
    else:
        interpretation = "anchor still harms gameplay"
    result = {
        "manifest": manifest, "baseline": baseline, "rows": rows, "best_trained": best,
        "success": gameplay_beaten or regret_improved, "interpretation": interpretation,
        "mechanics_replay_triggered": mechanics_damage,
        "next_step": "retain DPO candidate" if gameplay_beaten or regret_improved else "stop DPO and proceed to GRPO / online environment optimization",
    }
    write_json(output / "development-summary.json", result)
    write_development_table(output / "development-summary.md", rows)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the chosen-likelihood anchored DPO rescue experiment.")
    parser.add_argument("--preferences", type=Path, default=Path("data/wordle-dpo-dev"))
    parser.add_argument("--mechanics-data", type=Path, default=Path("data/wordle-dev-1m/fold-1/mechanics"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt"))
    parser.add_argument("--mode", type=Path, default=Path("data/wordle-development.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dpo-rescue-anchor"))
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    result = run_experiment(args.preferences, args.mechanics_data, args.base_checkpoint, args.mode, args.output_dir, list(load_words(args.words)), args.device)
    print(json.dumps({"success": result["success"], "interpretation": result["interpretation"], "next_step": result["next_step"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
