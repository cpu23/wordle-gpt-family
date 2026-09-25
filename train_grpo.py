"""One-guess GRPO: fresh reachable states/actions, one optimizer step per batch.

Distinct actions use token masking with previously drawn words removed. The
stored masks define the same conditional policy during sampling and updating.
KL is exact categorical KL on sampled token prefixes (not full trajectory KL).
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from evaluate_v2 import load_v2_model
from grpo_evaluate import evaluate_grpo
from grpo_states import ReachableStateSampler, action_reward
from tokenizer_v2 import VOCABULARY_SIZE, decode
from wordle import DEFAULT_WORDS, load_words

BASE_CHECKPOINT = Path("runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt")
MILESTONES = (0, 100, 250, 500, 1_000, 2_500, 5_000, 10_000)


@dataclass
class ActionBatch:
    inputs: Tensor
    positions: Tensor
    actions: Tensor
    masks: Tensor
    old_logps: Tensor
    rewards: Tensor
    reductions: Tensor
    guesses: list[list[str]]


@torch.no_grad()
def sample_actions(model, states, words, group_size=8, solve_bonus=5.0):
    """Sequential draws without replacement, using the model's masked logits."""
    device = next(model.parameters()).device
    count = len(states)
    lengths = torch.tensor([len(s.prompt) for s in states], device=device)
    width = int(lengths.max()) + 5
    prompts = torch.zeros((count, width), dtype=torch.long, device=device)
    for row, state in enumerate(states):
        prompts[row, :len(state.prompt)] = torch.tensor(state.prompt, device=device)
    remaining = [set(words) for _ in states]
    if group_size < 2 or any(len(pool) < group_size for pool in remaining):
        raise ValueError("each group requires at least two distinct legal guesses")
    rows = torch.arange(count, device=device)
    all_inputs, all_actions, all_masks, all_logps = [], [], [], []
    guesses = [[] for _ in states]
    rewards, reductions = [], []
    for _ in range(group_size):
        inputs = prompts.clone()
        pools = [list(pool) for pool in remaining]
        # Sorting makes seeded sampling independent of Python hash randomization.
        pools = [sorted(pool) for pool in pools]
        letters, masks, logps = [], [], []
        for offset in range(5):
            logits = model(inputs[:, :int(lengths.max()) + offset])[rows, lengths + offset - 1]
            mask = torch.zeros_like(logits, dtype=torch.bool)
            for row, pool in enumerate(pools):
                ids = [ord(c) - ord('a') for c in sorted({w[offset] for w in pool})]
                mask[row, ids] = True
            distribution = F.log_softmax(logits.masked_fill(~mask, -torch.inf), dim=-1)
            token = torch.multinomial(distribution.exp(), 1).squeeze(1)
            inputs[rows, lengths + offset] = token
            letters.append(token)
            masks.append(mask)
            logps.append(distribution.gather(1, token[:, None]).squeeze(1))
            for row, value in enumerate(token.tolist()):
                pools[row] = [w for w in pools[row] if ord(w[offset]) - ord('a') == value]
        actions = torch.stack(letters, dim=1)
        rank_rewards, rank_reductions = [], []
        for row, state in enumerate(states):
            guess = decode(actions[row].tolist())
            remaining[row].remove(guess)
            guesses[row].append(guess)
            reward, after = action_reward(state, guess, solve_bonus)
            rank_rewards.append(reward)
            rank_reductions.append(1 - after / len(state.candidates))
        all_inputs.append(inputs)
        all_actions.append(actions)
        all_masks.append(torch.stack(masks, dim=1))
        all_logps.append(torch.stack(logps, dim=1).sum(1))
        rewards.append(rank_rewards)
        reductions.append(rank_reductions)
    # State-major, then action-rank ordering throughout.
    return ActionBatch(
        torch.stack(all_inputs, 1).flatten(0, 1),
        (lengths[:, None] + torch.arange(5, device=device) - 1).repeat_interleave(group_size, 0),
        torch.stack(all_actions, 1).flatten(0, 1),
        torch.stack(all_masks, 1).flatten(0, 1),
        torch.stack(all_logps, 1),
        torch.tensor(rewards, device=device).T.contiguous(),
        torch.tensor(reductions, device=device).T.contiguous(), guesses,
    )


def relative_advantages(rewards):
    centered = rewards - rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, correction=0, keepdim=True)
    identical = rewards.amax(1, keepdim=True) == rewards.amin(1, keepdim=True)
    return torch.where(identical, torch.zeros_like(centered), centered / std.clamp_min(1e-8))


def selected_logits(model, batch):
    logits = model(batch.inputs)
    return logits.gather(1, batch.positions[:, :, None].expand(-1, -1, logits.shape[-1]))


def grpo_loss(policy_logits, reference_logits, batch, kl_beta=0.02, clip=0.2):
    raw_logp = F.log_softmax(policy_logits, -1)
    reference_raw = F.log_softmax(reference_logits, -1)
    # Raw KL also anchors probability mass outside the constrained dictionary.
    raw_kl = (raw_logp.exp() * (raw_logp - reference_raw)).sum(-1).mean()
    policy = F.log_softmax(policy_logits.masked_fill(~batch.masks, -torch.inf), -1)
    reference = F.log_softmax(reference_logits.masked_fill(~batch.masks, -torch.inf), -1)
    difference = policy.masked_fill(~batch.masks, 0) - reference.masked_fill(~batch.masks, 0)
    constrained_kl = (policy.exp() * difference).sum(-1).mean()
    logps = policy.gather(2, batch.actions[:, :, None]).squeeze(2).sum(1).view_as(batch.rewards)
    ratio = (logps - batch.old_logps).exp()
    advantages = relative_advantages(batch.rewards)
    surrogate = torch.minimum(ratio * advantages, ratio.clamp(1-clip, 1+clip) * advantages)
    loss = -surrogate.mean() + kl_beta * raw_kl
    return loss, {"raw_prefix_kl_from_sft": float(raw_kl.detach()),
                  "constrained_prefix_kl_from_sft": float(constrained_kl.detach()),
                  "mean_importance_ratio": float(ratio.detach().mean())}


def group_diagnostics(batch, states, solve_bonus=5.0):
    """Rewards from actual outcomes; diversity across states, not forced group size."""
    rewards = batch.rewards
    guesses = [guess for group in batch.guesses for guess in group]
    first_guesses = [group[0] for group in batch.guesses]
    counts, first_counts = Counter(guesses), Counter(first_guesses)
    solved = torch.tensor(
        [[guess == state.secret for guess in group] for state, group in zip(states, batch.guesses)],
        device=rewards.device,
    )
    std = rewards.std(1, correction=0)
    identical = (rewards.amax(1) == rewards.amin(1)).float().mean()
    return {
        "groups": len(states), "sampled_actions": len(guesses),
        "mean_group_reward": float(rewards.mean()),
        "mean_log_candidate_reduction": float((rewards - solve_bonus * solved).mean()),
        "mean_candidate_reduction": float(batch.reductions.mean()),
        "immediate_solve_fraction": float(solved.float().mean()),
        "reward_std": float(rewards.std(correction=0)),
        "mean_group_reward_std": float(std.mean()),
        "identical_reward_group_rate": float(identical),
        "unique_action_count": len(counts),
        "unique_action_fraction": len(counts) / len(guesses),
        "first_draw_unique_action_count": len(first_counts),
        "first_draw_unique_action_fraction": len(first_counts) / len(first_guesses),
        "empirical_action_entropy_nats": -sum(
            (n / len(guesses)) * math.log(n / len(guesses)) for n in counts.values()
        ),
        "first_draw_empirical_action_entropy_nats": -sum(
            (n / len(first_guesses)) * math.log(n / len(first_guesses)) for n in first_counts.values()
        ),
    }


@torch.no_grad()
def policy_diagnostics(policy_logits, reference_logits, batch):
    """Exact token entropy/KL, averaged over sampled prefixes; units are nats."""
    raw = F.log_softmax(policy_logits, -1)
    ref_raw = F.log_softmax(reference_logits, -1)
    masked = F.log_softmax(policy_logits.masked_fill(~batch.masks, -torch.inf), -1)
    ref_masked = F.log_softmax(reference_logits.masked_fill(~batch.masks, -torch.inf), -1)
    finite = masked.masked_fill(~batch.masks, 0)
    ref_finite = ref_masked.masked_fill(~batch.masks, 0)
    raw_kl = (raw.exp() * (raw - ref_raw)).sum(-1).mean().clamp_min(0)
    constrained_kl = (masked.exp() * (finite - ref_finite)).sum(-1).mean().clamp_min(0)
    token_entropy = -(masked.exp() * finite).sum(-1)
    group_shape = batch.rewards.shape
    action_entropy = token_entropy.sum(1).view(group_shape)
    return {
        "raw_prefix_kl_from_sft": float(raw_kl),
        "constrained_prefix_kl_from_sft": float(constrained_kl),
        "raw_token_entropy_nats": float(-(raw.exp() * raw).sum(-1).mean()),
        "constrained_token_entropy_nats": float(token_entropy.mean()),
        "conditional_action_entropy_nats": float(action_entropy.mean()),
        "first_draw_action_entropy_nats": float(action_entropy[:, 0].mean()),
    }


def checkpoint_rank(report):
    """Lexicographic validation rank, with fixed-SFT-prefix KL as last tie-break."""
    gameplay = report["gameplay"]["constrained"]
    return (
        gameplay["wins"],
        -gameplay["average_attempts"],
        -report["action_regret"]["constrained"]["summary"]["mean_action_regret"],
        -report["diagnostics"]["fixed_sft_prefix_kl_from_sft"],
    )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(args):
    if args.updates < 1 or args.states_per_update < 1 or args.diagnostic_states < 1:
        raise ValueError("updates, states per update, and diagnostic states must be positive")
    if args.eval_every < 0:
        raise ValueError("extra evaluation interval must be nonnegative (zero disables it)")
    if args.lr <= 0 or args.kl_beta <= 0 or args.solve_bonus < 0:
        raise ValueError("learning rate/KL must be positive; solve bonus nonnegative")
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    words = tuple(load_words(args.words))
    split = json.loads(args.mode.read_text())["runs"][0]
    if set(split['train']) & (set(split['validation']) | set(split['test'])):
        raise ValueError("training secrets overlap held-out secrets")
    source = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    if "dpo" in source or "grpo" in source:
        raise ValueError("base must be the original SFT checkpoint, not DPO/GRPO")
    model = load_v2_model(args.base_checkpoint, args.device)
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 7_162_403:
        raise ValueError(f"expected the 7.2M SFT architecture, got {parameters} parameters")
    model.eval()  # Gradients remain enabled; rollout/update share deterministic behavior.
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sampler = ReachableStateSampler(words, split['train'], args.seed)
    diagnostic_seed = args.seed + 1_000_003
    diagnostic_sampler = ReachableStateSampler(words, split['validation'], diagnostic_seed)
    diagnostic_states = [diagnostic_sampler.sample() for _ in range(args.diagnostic_states)]
    rng_devices = [torch.cuda.current_device()] if args.device == 'cuda' else []
    # Fixed reference prefixes make KL ranking comparable across checkpoints.
    # fork_rng prevents checkpoint frequency from changing the training stream.
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(diagnostic_seed)
        probe = sample_actions(reference, diagnostic_states, words, solve_bonus=args.solve_bonus)
    with torch.no_grad():
        probe_reference_logits = selected_logits(reference, probe)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'manifest.json').exists():
        raise ValueError("output already contains a run; use a new output directory")
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest()
    write_json(output / 'manifest.json', {
        **config, 'parameters': parameters, 'base_checkpoint_sha256': provenance,
        'group_size': 8, 'reward': 'log(before/after) + solve_bonus * actual_solve',
        'reference': 'frozen original SFT', 'kl_regularizer': 'exact raw categorical KL on sampled prefixes',
        'sampling': 'autoregressive token masking with prior group guesses excluded',
        'optimizer_steps_per_fresh_batch': 1, 'milestones': list(MILESTONES),
        'scheduled_checkpoints': sorted({0, args.updates, *(s for s in MILESTONES if s <= args.updates)}),
        'checkpoint_ranking': ['validation_wins', 'lower_average_attempts',
                               'lower_mean_action_regret', 'lower_fixed_sft_prefix_kl'],
        'diagnostic_seed': diagnostic_seed,
        'diagnostic_panel': 'fixed validation-secret reachable states; RNG isolated from training',
        'entropy_definition': 'categorical token entropy summed over five sampled prefixes; first draw has no group exclusions',
        'diversity_definition': 'empirical word frequencies across panel; within-group uniqueness is forced',
        'test_evaluated': False, 'training_secrets': split['train'], 'validation_secrets': split['validation'],
    })
    write_json(output / 'diagnostic-states.json', [
        {'secret': state.secret, 'prompt': state.prompt, 'history': state.history,
         'candidates': state.candidates} for state in diagnostic_states
    ])
    started = time.monotonic()
    best_key = None
    ranking = []
    def save(step, path):
        temporary = path.with_suffix('.tmp')
        torch.save({'format_version': 1, 'model_state_dict': model.state_dict(),
                    'model_config': model.config, 'vocabulary_size': VOCABULARY_SIZE,
                    'grpo': {**config, 'updates': step}, 'base_checkpoint_sha256': provenance,
                    'base_checkpoint': str(args.base_checkpoint), 'optimizer_state_dict': optimizer.state_dict()}, temporary)
        temporary.replace(path)
    def evaluate(step):
        nonlocal best_key
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            torch.manual_seed(diagnostic_seed)
            diagnostic_batch = sample_actions(model, diagnostic_states, words, solve_bonus=args.solve_bonus)
            diagnostics = group_diagnostics(diagnostic_batch, diagnostic_states, args.solve_bonus)
            diagnostics.update(policy_diagnostics(
                selected_logits(model, diagnostic_batch),
                selected_logits(reference, diagnostic_batch), diagnostic_batch,
            ))
            fixed = policy_diagnostics(selected_logits(model, probe), probe_reference_logits, probe)
            diagnostics['fixed_sft_prefix_kl_from_sft'] = fixed['raw_prefix_kl_from_sft']
        report = evaluate_grpo(model, split['validation'], words, solve_bonus=args.solve_bonus,
                               checkpoint=f'update-{step}')
        report.update(updates=step, diagnostics=diagnostics, wall_clock_seconds=time.monotonic()-started)
        write_json(output / f'validation-{step}.json', report)
        save(step, checkpoints / f'update-{step}.pt')
        # Choose by validation performance only, never training reward or test games.
        gameplay = report['gameplay']['constrained']
        key = checkpoint_rank(report)
        ranking.append({'updates': step, 'validation_key': key, 'checkpoint': f'checkpoints/update-{step}.pt'})
        ranking.sort(key=lambda entry: entry['validation_key'], reverse=True)
        write_json(output / 'checkpoint-ranking.json', ranking)
        if best_key is None or key > best_key:
            best_key = key
            save(step, checkpoints / 'best.pt')
            write_json(output / 'best.json', {'updates': step, 'validation_key': key})
        summary = {k: v for k, v in gameplay.items() if k not in ('results', 'winning_attempts')}
        print(json.dumps({'event': 'evaluation', 'updates': step, 'constrained': summary,
                          'diagnostics': diagnostics, 'validation_key': key,
                          'elapsed_seconds': time.monotonic()-started}), flush=True)
    evaluate(0)
    with (output / 'metrics.jsonl').open('w') as metrics:
        for step in range(1, args.updates + 1):
            states = [sampler.sample() for _ in range(args.states_per_update)]
            batch = sample_actions(model, states, words, solve_bonus=args.solve_bonus)
            with torch.no_grad():
                ref_logits = selected_logits(reference, batch)
            optimizer.zero_grad(set_to_none=True)
            loss, stats = grpo_loss(selected_logits(model, batch), ref_logits, batch, args.kl_beta)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite GRPO loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            # Post-update KL makes even a single-update run's measured drift visible.
            with torch.no_grad():
                _, after = grpo_loss(selected_logits(model, batch), ref_logits, batch, args.kl_beta)
            record = {'updates': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                      **group_diagnostics(batch, states, args.solve_bonus),
                      'mean_history_depth': sum(len(s.history) for s in states)/len(states),
                      'learning_rate': args.lr, 'pre_update': stats, 'post_update': after,
                      'wall_clock_seconds': time.monotonic()-started}
            metrics.write(json.dumps(record) + '\n')
            metrics.flush()
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(record), flush=True)
            if (args.eval_every and step % args.eval_every == 0) or step in MILESTONES or step == args.updates:
                evaluate(step)
    write_json(output / 'training-complete.json', {'updates': args.updates, 'elapsed_seconds': time.monotonic()-started})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, default=BASE_CHECKPOINT)
    parser.add_argument('--mode', type=Path, default=Path('data/wordle-development.json'))
    parser.add_argument('--words', type=Path, default=DEFAULT_WORDS)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/grpo-dev-dense'))
    parser.add_argument('--updates', type=int, default=10_000)
    parser.add_argument('--states-per-update', type=int, default=16)
    parser.add_argument('--diagnostic-states', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--kl-beta', type=float, default=0.02)
    parser.add_argument('--solve-bonus', type=float, default=5.0)
    parser.add_argument('--eval-every', type=int, default=0,
                        help='Optional extra evaluation interval; zero uses only the dense schedule and final update.')
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args()
    if args.log_every < 1:
        parser.error('--log-every must be positive')
    print(run(args))


if __name__ == '__main__':
    main()
