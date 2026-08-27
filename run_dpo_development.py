from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import torch

from action_regret import analyze_action_regret
from cross_validation import load_mode
from evaluate_v2 import evaluate_checkpoint
from experiments_v2 import evaluate_objective_loss
from train_dpo import evaluate_preferences, load_preferences, reference_logps, train_dpo
from train_v2 import load_v2_split
from wordle import DEFAULT_WORDS, load_words
from evaluate_v2 import load_v2_model

BETAS = (0.05, 0.10, 0.20)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_sweep(
    preference_dir: str | Path,
    mechanics_dir: str | Path,
    base_checkpoint: str | Path,
    mode_path: str | Path,
    output_dir: str | Path,
    words: list[str],
    *,
    device: str = "cuda",
) -> dict[str, object]:
    preference_dir, base_checkpoint, output = Path(preference_dir), Path(base_checkpoint), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_preferences = load_preferences(preference_dir / "train.jsonl.gz")
    validation_preferences = load_preferences(preference_dir / "validation.jsonl.gz")
    mechanics = load_v2_split(mechanics_dir, "validation", example_type="mechanics")
    run = load_mode(mode_path).runs[0]
    manifest = {
        "experiment": "wordle-dpo-development-beta-sweep", "betas": list(BETAS), "seed": 0,
        "base_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": sha256(base_checkpoint),
        "preference_manifest": json.loads((preference_dir / "manifest.json").read_text()),
        "validation_secrets": list(run.validation), "test_evaluated": False,
    }
    write_json(output / "manifest.json", manifest)

    summaries: dict[str, object] = {}
    baseline_dir = output / "sft-baseline"
    baseline_dir.mkdir(exist_ok=True)
    baseline_model = load_v2_model(base_checkpoint, device)
    validation_reference = reference_logps(baseline_model, validation_preferences, 256, torch.device(device))
    baseline_preference = evaluate_preferences(baseline_model, validation_preferences, validation_reference, 0.1, 256, torch.device(device))
    baseline_mechanics = evaluate_objective_loss(baseline_model, mechanics, batch_size=256)
    for decode in ("raw", "constrained"):
        result = evaluate_checkpoint(base_checkpoint, run.validation, words, device=device, decode=decode)
        write_json(baseline_dir / f"validation-{decode}.json", asdict(result))
    constrained_payload = json.loads((baseline_dir / "validation-constrained.json").read_text())
    write_json(baseline_dir / "action-regret.json", analyze_action_regret(constrained_payload, words))
    raw_payload = json.loads((baseline_dir / "validation-raw.json").read_text())
    summaries["sft"] = {
        "beta": None, "constrained": constrained_payload, "raw": raw_payload,
        "preference": asdict(baseline_preference), "mechanics_validation_loss": baseline_mechanics,
        "action_regret": json.loads((baseline_dir / "action-regret.json").read_text())["summary"],
    }
    del baseline_model
    torch.cuda.empty_cache()

    for beta in BETAS:
        label = f"beta-{beta:.2f}"
        run_dir = output / label
        complete = run_dir / "training-complete.json"
        metrics = run_dir / "metrics.jsonl"
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not complete.exists() or not metrics.exists() or metrics.stat().st_size == 0:
            checkpoint, records = train_dpo(
                train_preferences, validation_preferences, mechanics, run_dir,
                base_checkpoint=base_checkpoint, validation_secrets=run.validation,
                allowed_words=words, beta=beta, device=device,
            )
            write_json(complete, {"checkpoint": str(checkpoint), "evaluations": len(records)})
        for decode in ("raw", "constrained"):
            result = evaluate_checkpoint(checkpoint, run.validation, words, device=device, decode=decode)
            write_json(run_dir / f"validation-{decode}.json", asdict(result))
        constrained_payload = json.loads((run_dir / "validation-constrained.json").read_text())
        raw_payload = json.loads((run_dir / "validation-raw.json").read_text())
        write_json(run_dir / "action-regret.json", analyze_action_regret(constrained_payload, words))
        best = json.loads((run_dir / "best.json").read_text())
        summaries[label] = {
            "beta": beta, "constrained": constrained_payload, "raw": raw_payload,
            "selected": best, "preference": best["validation"],
            "mechanics_validation_loss": best["mechanics_validation_loss"],
            "action_regret": json.loads((run_dir / "action-regret.json").read_text())["summary"],
        }
        torch.cuda.empty_cache()
    aggregate = {"manifest": manifest, "models": summaries}
    catastrophic = []
    for beta in BETAS:
        metrics_path = output / f"beta-{beta:.2f}" / "metrics.jsonl"
        history = [json.loads(line) for line in metrics_path.read_text().splitlines()]
        if max(record["mechanics_validation_loss"] for record in history) >= baseline_mechanics * 10:
            catastrophic.append(beta)
    if catastrophic:
        beta = catastrophic[0]
        label = f"beta-{beta:.2f}-mechanics-5pct"
        run_dir = output / label
        complete = run_dir / "training-complete.json"
        checkpoint = run_dir / "checkpoints" / "best.pt"
        if not complete.exists():
            checkpoint, records = train_dpo(
                train_preferences, validation_preferences, mechanics, run_dir,
                base_checkpoint=base_checkpoint, validation_secrets=run.validation,
                allowed_words=words, beta=beta, device=device,
                mechanics_train=load_v2_split(mechanics_dir, "train", example_type="mechanics"),
                mechanics_replay_fraction=0.05,
            )
            write_json(complete, {"checkpoint": str(checkpoint), "evaluations": len(records)})
        for decode in ("raw", "constrained"):
            result = evaluate_checkpoint(checkpoint, run.validation, words, device=device, decode=decode)
            write_json(run_dir / f"validation-{decode}.json", asdict(result))
        constrained_payload = json.loads((run_dir / "validation-constrained.json").read_text())
        raw_payload = json.loads((run_dir / "validation-raw.json").read_text())
        write_json(run_dir / "action-regret.json", analyze_action_regret(constrained_payload, words))
        best = json.loads((run_dir / "best.json").read_text())
        summaries[label] = {
            "beta": beta, "mechanics_replay_fraction": 0.05,
            "constrained": constrained_payload, "raw": raw_payload,
            "selected": best, "preference": best["validation"],
            "mechanics_validation_loss": best["mechanics_validation_loss"],
            "action_regret": json.loads((run_dir / "action-regret.json").read_text())["summary"],
        }
    write_json(output / "aggregate.json", aggregate)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the controlled Wordle DPO beta sweep.")
    parser.add_argument("--preferences", type=Path, default=Path("data/wordle-dpo-dev"))
    parser.add_argument("--mechanics-data", type=Path, default=Path("data/wordle-dev-1m/fold-1/mechanics"))
    parser.add_argument("--base-checkpoint", type=Path, default=Path("runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt"))
    parser.add_argument("--mode", type=Path, default=Path("data/wordle-development.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/dpo-dev"))
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    print(json.dumps(run_sweep(args.preferences, args.mechanics_data, args.base_checkpoint, args.mode, args.output_dir, list(load_words(args.words)), device=args.device), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
