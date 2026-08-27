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
    policy_chosen_logp: float
    policy_rejected_logp: float
    reference_chosen_logp: float
    reference_rejected_logp: float
    policy_margin: float
    reference_margin: float
    preference_accuracy: float
    reference_deviation: float


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


def _stats(pc: Tensor, pr: Tensor, rc: Tensor, rr: Tensor, beta: float) -> DPOStats:
    return DPOStats(
        loss=float(dpo_loss(pc, pr, rc, rr, beta).mean()),
        policy_chosen_logp=float(pc.mean()), policy_rejected_logp=float(pr.mean()),
        reference_chosen_logp=float(rc.mean()), reference_rejected_logp=float(rr.mean()),
        policy_margin=float((pc - pr).mean()), reference_margin=float((rc - rr).mean()),
        preference_accuracy=float((pc > pr).float().mean()),
        reference_deviation=float((((pc - rc) + (pr - rr)) * 0.5).mean()),
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
def evaluate_preferences(model: WordleGPT, data: PreferenceData, references: Tensor, beta: float, batch_size: int, device: torch.device) -> DPOStats:
    model.eval()
    pc, pr = [], []
    for start in range(0, len(data), batch_size):
        indices = torch.arange(start, min(start + batch_size, len(data)))
        prompts, lengths, chosen, rejected = _batch(data, indices, device)
        pc.append(completion_logps(model, prompts, lengths, chosen).cpu())
        pr.append(completion_logps(model, prompts, lengths, rejected).cpu())
    return _stats(torch.cat(pc), torch.cat(pr), references[:, 0], references[:, 1], beta)


def checkpoint_better(candidate: DPORecord, reference: DPORecord | None) -> bool:
    if reference is None:
        return True
    candidate_key = (
        candidate.constrained_wins,
        -candidate.constrained_average_attempts,
        -candidate.constrained_average_guesses,
        -candidate.raw_invalid_guesses,
        candidate.validation.preference_accuracy,
        -candidate.validation.loss,
    )
    reference_key = (
        reference.constrained_wins,
        -reference.constrained_average_attempts,
        -reference.constrained_average_guesses,
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
    learning_rate: float = 1e-5,
    physical_batch_size: int = 64,
    gradient_accumulation_steps: int = 2,
    eval_batch_size: int = 256,
    max_passes: int = 10,
    patience: int = 3,
    seed: int = 0,
    device: str = "cuda",
) -> tuple[Path, list[DPORecord]]:
    if mechanics_replay_fraction < 0 or mechanics_replay_fraction >= 1:
        raise ValueError("mechanics replay fraction must be in [0, 1)")
    if (mechanics_train is None) != (mechanics_replay_fraction == 0):
        raise ValueError("mechanics training data and a positive replay fraction must be provided together")
    if physical_batch_size * gradient_accumulation_steps != 128:
        raise ValueError("effective preference batch size must equal 128")
    selected_device = torch.device(device)
    torch.manual_seed(seed)
    base_checkpoint = Path(base_checkpoint)
    policy = load_v2_model(base_checkpoint, device)
    reference = load_v2_model(base_checkpoint, device)
    reference.requires_grad_(False)
    reference.eval()
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise RuntimeError("reference model must be frozen")
    sample_indices = torch.arange(min(8, len(validation_data)))
    sample = _batch(validation_data, sample_indices, selected_device)
    with torch.no_grad():
        if not torch.equal(completion_logps(policy, *sample[:2], sample[2]), completion_logps(reference, *sample[:2], sample[2])):
            raise RuntimeError("policy and reference do not begin with identical log-probabilities")
    train_reference = reference_logps(reference, train_data, eval_batch_size, selected_device)
    validation_reference = reference_logps(reference, validation_data, eval_batch_size, selected_device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)
    output = Path(output_dir)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "metrics.jsonl"
    metrics_path.write_text("", encoding="utf-8")
    config = {
        "beta": beta, "learning_rate": learning_rate, "optimizer": "AdamW", "physical_batch_size": physical_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps, "effective_batch_size": 128,
        "eval_batch_size": eval_batch_size, "max_effective_passes": max_passes, "patience": patience,
        "seed": seed, "base_checkpoint": str(base_checkpoint), "base_checkpoint_sha256": hashlib.sha256(base_checkpoint.read_bytes()).hexdigest(),
        "train_pairs": len(train_data), "validation_pairs": len(validation_data), "completion_log_probability": "sum over exactly five guess-letter tokens",
        "mechanics_replay": mechanics_replay_fraction > 0,
        "mechanics_replay_fraction": mechanics_replay_fraction,
    }
    (output / "run.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if selected_device.type == "cuda": torch.cuda.reset_peak_memory_stats(selected_device)
    started = time.perf_counter()
    records: list[DPORecord] = []
    best: DPORecord | None = None
    best_path = checkpoints / "best.pt"
    checks_without_progress = 0
    steps = pairs_seen = 0
    generator = torch.Generator().manual_seed(seed)
    last_train: DPOStats | None = None
    last_gradient: float | None = None

    mechanics_generator = torch.Generator().manual_seed(seed + 1_000_003)
    for epoch in range(max_passes + 1):
        validation = evaluate_preferences(policy, validation_data, validation_reference, beta, eval_batch_size, selected_device)
        mechanics_loss = evaluate_objective_loss(policy, mechanics_validation, batch_size=eval_batch_size)
        constrained = evaluate_model(policy, validation_secrets, allowed_words, decode="constrained")
        raw = evaluate_model(policy, validation_secrets, allowed_words, decode="raw")
        record = DPORecord(
            epoch, steps, pairs_seen, pairs_seen / len(train_data), last_train, validation,
            constrained.wins, constrained.average_attempts, constrained.average_guesses,
            raw.wins, raw.invalid_guesses, mechanics_loss, last_gradient,
            float(optimizer.param_groups[0]["lr"]), time.perf_counter() - started,
            torch.cuda.max_memory_allocated(selected_device) if selected_device.type == "cuda" else None, False,
        )
        improved = checkpoint_better(record, best)
        record = replace(record, improved=improved)
        records.append(record)
        with metrics_path.open("a", encoding="utf-8") as metrics:
            metrics.write(json.dumps(asdict(record), sort_keys=True) + "\n")
        if improved:
            best, checks_without_progress = record, 0
            _save_checkpoint(best_path, policy, record, base_checkpoint, beta, seed)
            (output / "best.json").write_text(json.dumps(asdict(record), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        elif epoch:
            checks_without_progress += 1
        print(f"pass={epoch} constrained={constrained.wins}/{constrained.games} raw={raw.wins}/{raw.games} invalid={raw.invalid_guesses} val_dpo={validation.loss:.4f} pref_acc={validation.preference_accuracy:.3f} mechanics={mechanics_loss:.4f} patience={checks_without_progress}/{patience}", flush=True)
        if epoch == max_passes or checks_without_progress >= patience:
            break

        order = torch.randperm(len(train_data), generator=generator)
        policy.train()
        sums = torch.zeros(9)
        count = 0
        gradient_sum = 0.0
        gradient_count = 0
        optimizer.zero_grad(set_to_none=True)
        microbatches = math.ceil(len(order) / physical_batch_size)
        for batch_number, start in enumerate(range(0, len(order), physical_batch_size)):
            indices = order[start : start + physical_batch_size]
            prompts, lengths, chosen, rejected = _batch(train_data, indices, selected_device)
            pc = completion_logps(policy, prompts, lengths, chosen)
            pr = completion_logps(policy, prompts, lengths, rejected)
            refs = train_reference.index_select(0, indices).to(selected_device)
            losses = dpo_loss(pc, pr, refs[:, 0], refs[:, 1], beta)
            (losses.mean() / gradient_accumulation_steps).backward()
            batch_count = len(indices)
            values = torch.tensor([
                float(losses.detach().mean()), float(pc.detach().mean()), float(pr.detach().mean()),
                float(refs[:, 0].mean()), float(refs[:, 1].mean()),
                float((pc.detach() - pr.detach()).mean()), float((refs[:, 0] - refs[:, 1]).mean()),
                float((pc.detach() > pr.detach()).float().mean()),
                float((((pc.detach() - refs[:, 0]) + (pr.detach() - refs[:, 1])) * 0.5).mean()),
            ])
            sums += values * batch_count
            count += batch_count
            final = batch_number + 1 == microbatches
            if (batch_number + 1) % gradient_accumulation_steps == 0 or final:
                accumulated = (batch_number % gradient_accumulation_steps) + 1
                if final and accumulated < gradient_accumulation_steps:
                    for parameter in policy.parameters():
                        if parameter.grad is not None: parameter.grad.mul_(gradient_accumulation_steps / accumulated)
                norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
                if not torch.isfinite(norm): raise FloatingPointError("non-finite DPO gradient norm")
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                gradient_sum += float(norm)
                gradient_count += 1
                steps += 1
                if mechanics_train is not None:
                    interval = round((1.0 - mechanics_replay_fraction) / mechanics_replay_fraction)
                    if gradient_count % interval == 0:
                        mechanics_indices = torch.randint(
                            len(mechanics_train.inputs), (128,), generator=mechanics_generator
                        )
                        mechanics_inputs = mechanics_train.inputs.index_select(0, mechanics_indices).to(selected_device)
                        mechanics_targets = mechanics_train.targets.index_select(0, mechanics_indices).to(selected_device)
                        replay_loss = calculate_loss(policy(mechanics_inputs), mechanics_targets)
                        replay_loss.backward()
                        mechanics_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), float("inf"))
                        if not torch.isfinite(mechanics_norm):
                            raise FloatingPointError("non-finite mechanics replay gradient norm")
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        steps += 1
            pairs_seen += batch_count
        means = sums / count
        last_train = DPOStats(*map(float, means))
        last_gradient = gradient_sum / gradient_count
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
