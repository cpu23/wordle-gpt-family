from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor
from torch.nn import functional as F
from action_regret import analyze_action_regret

from evaluate_v2 import evaluate_model, load_v2_model
from experiments_v2 import evaluate_objective_loss
from model import WordleGPT
from train import calculate_loss
from tokenizer_v2 import VOCABULARY_SIZE, encode
from train_v2 import V2SplitData, load_v2_split
from wordle import DEFAULT_WORDS, load_words


@dataclass(frozen=True)
class PreferenceData:
    prompts: Tensor
    prompt_lengths: Tensor
    chosen: Tensor
    rejected: Tensor
    pair_types: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.prompt_lengths)


@dataclass(frozen=True)
class DPOStats:
    loss: float
    chosen_nll: float
    total_loss: float
    policy_chosen_logp: float
    policy_rejected_logp: float
    reference_chosen_logp: float
    reference_rejected_logp: float
    policy_margin: float
    reference_margin: float
    preference_accuracy: float
    reference_deviation: float
    chosen_delta_logp: float
    rejected_delta_logp: float


@dataclass(frozen=True)
class DPORecord:
    epoch: int
    optimizer_steps: int
    pairs_seen: int
    effective_passes: float
    train: DPOStats | None
    validation: DPOStats
    constrained_wins: int
    constrained_average_attempts: float
    constrained_average_guesses: float
    raw_wins: int
    raw_invalid_guesses: int
    mechanics_validation_loss: float
    action_regret: dict[str, object]
    gradient_norm: float | None
    learning_rate: float
    wall_clock_seconds: float
    peak_gpu_memory_bytes: int | None
    improved: bool


def load_preferences(path: str | Path, context_length: int = 96) -> PreferenceData:
    rows: list[tuple[list[int], list[int], list[int], str]] = []
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            prompt = encode(record["prompt"])
            chosen = encode(record["chosen_guess"])
            rejected = encode(record["rejected_guess"])
            if len(chosen) != 5 or len(rejected) != 5 or len(prompt) + 5 > context_length:
                raise ValueError("DPO examples require two five-letter completions within context")
            rows.append((prompt, chosen, rejected, record["pair_type"]))
    if not rows:
        raise ValueError("preference dataset is empty")
    prompts = torch.zeros((len(rows), context_length), dtype=torch.int16)
    lengths = torch.empty(len(rows), dtype=torch.int16)
    chosen = torch.empty((len(rows), 5), dtype=torch.int16)
    rejected = torch.empty((len(rows), 5), dtype=torch.int16)
    for index, (prompt, good, bad, _) in enumerate(rows):
        prompts[index, : len(prompt)] = torch.tensor(prompt, dtype=torch.int16)
        lengths[index] = len(prompt)
        chosen[index] = torch.tensor(good, dtype=torch.int16)
        rejected[index] = torch.tensor(bad, dtype=torch.int16)
    return PreferenceData(prompts, lengths, chosen, rejected, tuple(row[3] for row in rows))


def completion_logps(model: WordleGPT, prompts: Tensor, lengths: Tensor, completions: Tensor) -> Tensor:
    """Sum autoregressive log-probabilities over exactly five completion letters."""
    batch_size = len(lengths)
    inputs = prompts.to(dtype=torch.long).clone()
    positions = lengths.to(dtype=torch.long).unsqueeze(1) + torch.arange(5, device=lengths.device)
    inputs.scatter_(1, positions, completions.to(dtype=torch.long))
    logits = model(inputs)
    prediction_positions = positions - 1
    selected_logits = logits.gather(1, prediction_positions.unsqueeze(2).expand(-1, -1, logits.size(2)))
    token_logps = F.log_softmax(selected_logits, dim=-1).gather(2, completions.to(dtype=torch.long).unsqueeze(2)).squeeze(2)
    if not torch.isfinite(token_logps).all():
        raise FloatingPointError("non-finite completion log-probability")
    return token_logps.sum(dim=1)


def dpo_loss(policy_chosen: Tensor, policy_rejected: Tensor, reference_chosen: Tensor, reference_rejected: Tensor, beta: float) -> Tensor:
    if beta <= 0:
        raise ValueError("beta must be positive")
    logits = beta * ((policy_chosen - policy_rejected) - (reference_chosen - reference_rejected))
    loss = -F.logsigmoid(logits)
    if not torch.isfinite(loss).all():
        raise FloatingPointError("non-finite DPO loss")
    return loss

def anchored_dpo_loss(
    policy_chosen: Tensor,
    policy_rejected: Tensor,
    reference_chosen: Tensor,
    reference_rejected: Tensor,
    beta: float,
    lambda_sft: float,
) -> tuple[Tensor, Tensor, Tensor]:
    if lambda_sft < 0:
        raise ValueError("lambda_sft must be nonnegative")
    dpo = dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta)
    chosen_nll = -policy_chosen
    return dpo + lambda_sft * chosen_nll, dpo, chosen_nll


def _stats(pc: Tensor, pr: Tensor, rc: Tensor, rr: Tensor, beta: float, lambda_sft: float = 0.0) -> DPOStats:
    dpo = float(dpo_loss(pc, pr, rc, rr, beta).mean())
    chosen_nll = float(-pc.mean())
    return DPOStats(
        loss=dpo,
        chosen_nll=chosen_nll,
        total_loss=dpo + lambda_sft * chosen_nll,
        policy_chosen_logp=float(pc.mean()),
        policy_rejected_logp=float(pr.mean()),
        reference_chosen_logp=float(rc.mean()),
        reference_rejected_logp=float(rr.mean()),
        policy_margin=float((pc - pr).mean()),
        reference_margin=float((rc - rr).mean()),
        preference_accuracy=float((pc > pr).float().mean()),
        reference_deviation=float((((pc - rc) + (pr - rr)) * 0.5).mean()),
        chosen_delta_logp=float((pc - rc).mean()),
        rejected_delta_logp=float((pr - rr).mean()),
    )


def _batch(data: PreferenceData, indices: Tensor, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return (
        data.prompts.index_select(0, indices).to(device),
        data.prompt_lengths.index_select(0, indices).to(device),
        data.chosen.index_select(0, indices).to(device),
        data.rejected.index_select(0, indices).to(device),
    )

@torch.no_grad()
def reference_logps(model: WordleGPT, data: PreferenceData, batch_size: int, device: torch.device) -> Tensor:
    model.eval()
    result = torch.empty((len(data), 2))
    for start in range(0, len(data), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data)))
        prompts, lengths, chosen, rejected = _batch(data, indices, device)
        result[indices, 0] = completion_logps(model, prompts, lengths, chosen).cpu()
        result[indices, 1] = completion_logps(model, prompts, lengths, rejected).cpu()
    return result

@torch.no_grad()
def evaluate_preferences(
    model: WordleGPT,
    data: PreferenceData,
    references: Tensor,
    beta: float,
    batch_size: int,
    device: torch.device,
    lambda_sft: float = 0.0,
) -> DPOStats:
    model.eval()
    pc, pr = [], []
    for start in range(0, len(data), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data)))
        prompts, lengths, chosen, rejected = _batch(data, indices, device)
        pc.append(completion_logps(model, prompts, lengths, chosen).cpu())
        pr.append(completion_logps(model, prompts, lengths, rejected).cpu())
    return _stats(torch.cat(pc), torch.cat(pr), references[:, 0], references[:, 1], beta, lambda_sft)


def checkpoint_better(candidate: DPORecord, reference: DPORecord | None) -> bool:
    if reference is None:
        return True
    candidate_key = (
        candidate.constrained_wins,
        -candidate.constrained_average_attempts,
        -candidate.constrained_average_guesses,
        candidate.raw_wins,
        -candidate.raw_invalid_guesses,
        candidate.validation.preference_accuracy,
        -candidate.validation.loss,
    )
    reference_key = (
        reference.constrained_wins,
        -reference.constrained_average_attempts,
        -reference.constrained_average_guesses,
        reference.raw_wins,
        -reference.raw_invalid_guesses,
        reference.validation.preference_accuracy,
        -reference.validation.loss,
    )
    return candidate_key > reference_key


def _save_checkpoint(path: Path, model: WordleGPT, record: DPORecord, base_checkpoint: Path, beta: float, seed: int) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save({
        "format_version": 1, "model_state_dict": model.state_dict(), "model_config": dict(model.config),
        "vocabulary_size": VOCABULARY_SIZE, "dpo": asdict(record), "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(), "beta": beta, "seed": seed,
    }, temporary)
    temporary.replace(path)


def train_dpo(
    train_data: PreferenceData,
    validation_data: PreferenceData,
    mechanics_validation: V2SplitData,
    output_dir: str | Path,
    *,
    base_checkpoint: str | Path,
    validation_secrets: Sequence[str],
    allowed_words: Sequence[str],
    mechanics_train: V2SplitData | None = None,
    mechanics_replay_fraction: float = 0.0,
    beta: float,
    lambda_sft: float = 0.0,
    learning_rate: float = 3e-6,
    physical_batch_size: int = 64,
    gradient_accumulation_steps: int = 2,
    eval_batch_size: int = 256,
    evaluation_passes: Sequence[float] = (0.0, 0.10, 0.25, 0.50, 0.75, 1.0),
    collapse_wins: int | None = None,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[Path, list[DPORecord]]:
    if beta <= 0 or lambda_sft < 0:
        raise ValueError("beta must be positive and lambda_sft nonnegative")
    if mechanics_replay_fraction < 0 or mechanics_replay_fraction >= 1:
        raise ValueError("mechanics replay fraction must be in [0, 1)")
    if (mechanics_train is None) != (mechanics_replay_fraction == 0):
        raise ValueError("mechanics training data and a positive replay fraction must be provided together")
    if physical_batch_size * gradient_accumulation_steps != 128:
        raise ValueError("effective preference batch size must equal 128")
    schedule = tuple(sorted(set(float(value) for value in evaluation_passes)))
    if not schedule or schedule[0] != 0.0 or schedule[-1] > 1.0 or any(value < 0 for value in schedule):
        raise ValueError("evaluation_passes must begin at 0 and remain within one pass")

    selected_device = torch.device(device)
    torch.manual_seed(seed)
    base_checkpoint = Path(base_checkpoint)
    policy = load_v2_model(base_checkpoint, device)
    reference = load_v2_model(base_checkpoint, device)
    reference.requires_grad_(False)
    reference.eval()
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise RuntimeError("reference model must be frozen")
    policy_state = policy.state_dict()
    reference_state = reference.state_dict()
    if policy_state.keys() != reference_state.keys() or any(
        not torch.equal(policy_state[name], reference_state[name]) for name in policy_state
    ):
        raise RuntimeError("policy and reference parameters differ at initialization")

    sample_indices = torch.arange(min(8, len(validation_data)))
    sample = _batch(validation_data, sample_indices, selected_device)
    with torch.no_grad():
        policy_chosen = completion_logps(policy, *sample[:2], sample[2])
        policy_rejected = completion_logps(policy, *sample[:2], sample[3])
        reference_chosen = completion_logps(reference, *sample[:2], sample[2])
        reference_rejected = completion_logps(reference, *sample[:2], sample[3])
        if not torch.equal(policy_chosen, reference_chosen) or not torch.equal(policy_rejected, reference_rejected):
            raise RuntimeError("policy and reference log-probabilities differ at initialization")
        initial_dpo = float(dpo_loss(policy_chosen, policy_rejected, reference_chosen, reference_rejected, beta).mean())
        if not math.isclose(initial_dpo, math.log(2), abs_tol=1e-6):
            raise RuntimeError(f"initial DPO loss is {initial_dpo}, expected log(2)")

    train_reference = reference_logps(reference, train_data, eval_batch_size, selected_device).to(selected_device)
    validation_reference = reference_logps(reference, validation_data, eval_batch_size, selected_device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    output = Path(output_dir)
    checkpoints = output / "checkpoints"
    evaluations = output / "evaluations"
    checkpoints.mkdir(parents=True, exist_ok=True)
    evaluations.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    base_hash = hashlib.sha256(base_checkpoint.read_bytes()).hexdigest()
    config = {
        "objective": "dpo_loss + lambda_sft * chosen_nll",
        "beta": beta, "lambda_sft": lambda_sft, "learning_rate": learning_rate, "optimizer": "AdamW",
        "physical_batch_size": physical_batch_size, "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": 128, "eval_batch_size": eval_batch_size, "evaluation_passes": list(schedule),
        "collapse_wins": collapse_wins, "seed": seed, "base_checkpoint": str(base_checkpoint),
        "base_checkpoint_sha256": base_hash, "train_pairs": len(train_data), "validation_pairs": len(validation_data),
        "completion_log_probability": "sum over exactly five guess-letter tokens",
        "initialization": {"parameters_equal": True, "chosen_logps_equal": True, "rejected_logps_equal": True, "dpo_loss": initial_dpo},
        "mechanics_replay_fraction": mechanics_replay_fraction,
    }
    (output / "run.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selected_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.perf_counter()
    records: list[DPORecord] = []
    best: DPORecord | None = None
    best_path = checkpoints / "best.pt"
    steps = pairs_seen = 0
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train_data), generator=generator)
    cursor = 0
    mechanics_generator = torch.Generator().manual_seed(seed + 1_000_003)
    preference_updates = 0
    last_train: DPOStats | None = None
    last_gradient: float | None = None

    def evaluate_at(target_pass: float) -> DPORecord:
        nonlocal best
        validation = evaluate_preferences(
            policy, validation_data, validation_reference, beta, eval_batch_size, selected_device, lambda_sft
        )
        mechanics_loss = evaluate_objective_loss(policy, mechanics_validation, batch_size=eval_batch_size)
        constrained = evaluate_model(policy, validation_secrets, allowed_words, decode="constrained")
        raw = evaluate_model(policy, validation_secrets, allowed_words, decode="raw")
        constrained_payload = asdict(constrained)
        raw_payload = asdict(raw)
        regret = analyze_action_regret(constrained_payload, allowed_words)
        label = f"pass-{target_pass:.2f}"
        (evaluations / f"{label}-constrained.json").write_text(
            json.dumps(constrained_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (evaluations / f"{label}-raw.json").write_text(
            json.dumps(raw_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (evaluations / f"{label}-action-regret.json").write_text(
            json.dumps(regret, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        record = DPORecord(
            epoch=0, optimizer_steps=steps, pairs_seen=pairs_seen, effective_passes=pairs_seen / len(train_data),
            train=last_train, validation=validation, constrained_wins=constrained.wins,
            constrained_average_attempts=constrained.average_attempts,
            constrained_average_guesses=constrained.average_guesses, raw_wins=raw.wins,
            raw_invalid_guesses=raw.invalid_guesses, mechanics_validation_loss=mechanics_loss,
            action_regret=regret["summary"], gradient_norm=last_gradient,
            learning_rate=float(optimizer.param_groups[0]["lr"]), wall_clock_seconds=time.perf_counter() - started,
            peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(selected_device) if selected_device.type == "cuda" else None,
            improved=False,
        )
        improved = checkpoint_better(record, best)
        record = replace(record, improved=improved)
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as metrics:
            metrics.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        if improved:
            best = record
            _save_checkpoint(best_path, policy, record, base_checkpoint, beta, seed)
            (output / "best.json").write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(
            f"pass={record.effective_passes:.3f} constrained={constrained.wins}/{constrained.games} "
            f"raw={raw.wins}/{raw.games} invalid={raw.invalid_guesses} val_dpo={validation.loss:.4f} "
            f"val_nll={validation.chosen_nll:.4f} chosen_delta={validation.chosen_delta_logp:.4f} "
            f"rejected_delta={validation.rejected_delta_logp:.4f} mechanics={mechanics_loss:.4f}",
            flush=True,
        )
        return record

    evaluate_at(0.0)
    optimizer.zero_grad(set_to_none=True)
    for target_pass in schedule[1:]:
        target_pairs = min(len(train_data), math.ceil(target_pass * len(train_data)))
        sums = torch.zeros(13)
        count = 0
        gradient_sum = 0.0
        gradient_count = 0
        segment_microbatches = 0
        while pairs_seen < target_pairs:
            indices = order[cursor : min(cursor + physical_batch_size, target_pairs)]
            cursor += len(indices)
            prompts, lengths, chosen, rejected = _batch(train_data, indices, selected_device)
            pc = completion_logps(policy, prompts, lengths, chosen)
            pr = completion_logps(policy, prompts, lengths, rejected)
            refs = train_reference.index_select(0, indices.to(selected_device))
            objective, dpo, chosen_nll = anchored_dpo_loss(
                pc, pr, refs[:, 0], refs[:, 1], beta, lambda_sft
            )
            (objective.mean() / gradient_accumulation_steps).backward()
            segment_microbatches += 1
            batch_count = len(indices)
            batch_stats = _stats(pc.detach(), pr.detach(), refs[:, 0], refs[:, 1], beta, lambda_sft)
            sums += torch.tensor(list(asdict(batch_stats).values())) * batch_count
            count += batch_count
            pairs_seen += batch_count
            final_partial = pairs_seen == target_pairs
            accumulated = segment_microbatches % gradient_accumulation_steps
            if accumulated == 0 or final_partial:
                if final_partial and accumulated:
                    for parameter in policy.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(gradient_accumulation_steps / accumulated)
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
                if not torch.isfinite(norm):
                    raise FloatingPointError("non-finite DPO gradient norm")
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                gradient_sum += float(norm)
                gradient_count += 1
                steps += 1
                preference_updates += 1
                if mechanics_train is not None:
                    replay_interval = max(1, round((1.0 - mechanics_replay_fraction) / mechanics_replay_fraction))
                    if preference_updates % replay_interval == 0:
                        mechanics_indices = torch.randint(
                            len(mechanics_train.inputs), (128,), generator=mechanics_generator
                        )
                        mechanics_inputs = mechanics_train.inputs.index_select(0, mechanics_indices).to(selected_device)
                        mechanics_targets = mechanics_train.targets.index_select(0, mechanics_indices).to(selected_device)
                        replay_loss = calculate_loss(policy(mechanics_inputs), mechanics_targets)
                        replay_loss.backward()
                        replay_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
                        if not torch.isfinite(replay_norm):
                            raise FloatingPointError("non-finite mechanics replay gradient norm")
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        steps += 1
        last_train = DPOStats(*map(float, sums / count))
        last_gradient = gradient_sum / gradient_count
        record = evaluate_at(target_pass)
        if collapse_wins is not None and record.constrained_wins < collapse_wins:
            (output / "early-stop.json").write_text(
                json.dumps({"reason": "constrained gameplay collapse", "threshold": collapse_wins, "record": asdict(record)}, indent=2) + "\n",
                encoding="utf-8",
            )
            break
    return best_path, records


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Wordle policy with reference-relative DPO.")
    parser.add_argument("--train-preferences", type=Path, required=True)
    parser.add_argument("--validation-preferences", type=Path, required=True)
    parser.add_argument("--mechanics-data", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--mode", type=Path, default=Path("data/wordle-development.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--words", type=Path, default=DEFAULT_WORDS)
    parser.add_argument("--beta", type=float, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    from cross_validation import load_mode
    mode = load_mode(args.mode)
    run = mode.runs[0]
    train_dpo(
        load_preferences(args.train_preferences), load_preferences(args.validation_preferences),
        load_v2_split(args.mechanics_data, "validation", example_type="mechanics"), args.output_dir,
        base_checkpoint=args.base_checkpoint, validation_secrets=run.validation, allowed_words=load_words(args.words), beta=args.beta, device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
