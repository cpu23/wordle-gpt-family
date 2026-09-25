"""Complete-game or reachable-state continuation GRPO from original 7.2M SFT."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch

from action_regret import analyze_action_regret
from evaluate_v2 import evaluate_model, load_v2_model
from grpo_corpus import load_corpus
from grpo_rollouts import GROUP_SIZE, MAX_TURNS, LegalWordDecoder, sample_games
from grpo_trajectory_loss import (
    rollout_metrics, trajectory_logits, trajectory_loss, trajectory_policy_metrics,
    trajectory_gradient_metrics,
)
from tokenizer_v2 import VOCABULARY_SIZE
from train_grpo import BASE_CHECKPOINT, MILESTONES, checkpoint_rank, write_json
from wordle import DEFAULT_WORDS, load_words


def rollout_traces(batch):
    """Persist every sampled game with its group secret and terminal reward."""
    wins = batch.won.flatten().tolist()
    attempts = batch.attempts.flatten().tolist()
    rewards = batch.rewards.flatten().tolist()
    return [
        {'group': row // GROUP_SIZE, 'member': row % GROUP_SIZE,
         'secret': batch.secrets[row // GROUP_SIZE], 'guesses': guesses,
         'won': wins[row], 'attempts': attempts[row], 'reward': rewards[row]}
        for row, guesses in enumerate(batch.guesses)
    ]


@torch.no_grad()
def evaluate_games(model, validation_secrets, words, step):
    """Greedy validation uses the same terminal reward, never reduction rewards."""
    report = {'updates': step, 'gameplay': {}, 'action_regret': {}}
    for mode in ('raw', 'constrained'):
        gameplay = asdict(evaluate_model(
            model, validation_secrets, words, checkpoint=f'update-{step}', decode=mode,
        ))
        gameplay['mean_trajectory_reward'] = sum(
            7 - len(game['guesses']) if game['won'] else 0 for game in gameplay['results']
        ) / gameplay['games']
        report['gameplay'][mode] = gameplay
        report['action_regret'][mode] = analyze_action_regret(gameplay, words)
    return report


@torch.no_grad()
def evaluate_continuations(model, reference, states, decoder):
    """Evaluate a fixed state panel in bounded GPU batches."""
    totals = {}
    traces = []
    winning_attempts = 0.0
    trajectory_count = 0
    for start in range(0, len(states), 32):
        chunk = states[start:start + 32]
        batch = sample_games(
            model, [state.secret for state in chunk], decoder,
            histories=[state.history for state in chunk],
        )
        metrics = rollout_metrics(batch)
        winning_attempts += (
            (metrics['mean_guesses_for_winning_rollouts'] or 0) * metrics['winning_rollouts']
        )
        for key in ('winning_rollouts', 'sampled_rollouts'):
            totals[key] = totals.get(key, 0) + metrics[key]
        for key in ('mean_trajectory_reward', 'mean_group_reward_std', 'identical_reward_group_rate'):
            totals[key] = totals.get(key, 0.0) + metrics[key] * len(chunk)
        count = batch.inputs.shape[0]
        kls = trajectory_policy_metrics(
            trajectory_logits(model, batch), trajectory_logits(reference, batch), batch,
        )
        for key, value in kls.items():
            totals[key] = totals.get(key, 0.0) + value * count
        trajectory_count += count
        for trace in rollout_traces(batch):
            state = chunk[trace['group']]
            trace.update(source_index=state.source_index, starting_history=state.history)
            trace['group'] += start
            traces.append(trace)
    for key in ('mean_trajectory_reward', 'mean_group_reward_std', 'identical_reward_group_rate'):
        totals[key] /= len(states)
    for key in kls:
        totals[key] /= trajectory_count
    totals['sampled_rollout_win_fraction'] = totals['winning_rollouts'] / totals['sampled_rollouts']
    totals['mean_guesses_for_winning_rollouts'] = (
        winning_attempts / totals['winning_rollouts'] if totals['winning_rollouts'] else None
    )
    return {'diagnostics': totals, 'sampled_rollouts': traces}


def run(args):
    if min(args.updates, args.groups_per_update, args.log_every) < 1:
        raise ValueError('updates, groups per update, and log interval must be positive')
    if args.lr <= 0 or args.kl_beta <= 0 or args.eval_every < 0:
        raise ValueError('LR/KL must be positive; extra evaluation interval nonnegative')
    output = args.output_dir
    if (output / 'manifest.json').exists():
        raise ValueError('output already contains a run; use a new output directory')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    secret_rng = random.Random(args.seed)
    words = tuple(load_words(args.words))
    split = json.loads(args.mode.read_text())['runs'][0]
    train_secrets = tuple(split['train'])
    validation_secrets = tuple(split['validation'])
    if not train_secrets or not validation_secrets:
        raise ValueError('training and validation splits must be nonempty')
    if set(train_secrets) & (set(validation_secrets) | set(split['test'])):
        raise ValueError('training secrets overlap held-out secrets')
    if len(set(train_secrets)) != len(train_secrets):
        raise ValueError('duplicate training secrets would bias uniform sampling')
    training_states = load_corpus(args.state_corpus) if args.state_corpus else None
    validation_states = load_corpus(args.validation_corpus) if args.validation_corpus else None
    if bool(training_states) != bool(validation_states):
        raise ValueError('provide nonempty training and validation state corpora together')
    for states, allowed in ((training_states, train_secrets), (validation_states, validation_secrets)):
        if states is not None and any(state.secret not in allowed for state in states):
            raise ValueError('state corpus contains secrets outside its designated split')
    training_panel = (
        random.Random(args.seed + 2_000_003).sample(training_states, min(512, len(training_states)))
        if training_states else None
    )
    source = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True)
    if 'dpo' in source or 'grpo' in source:
        raise ValueError('initialize from original SFT, not a DPO or GRPO checkpoint')
    model = load_v2_model(args.base_checkpoint, args.device).eval()
    parameters = sum(p.numel() for p in model.parameters())
    if parameters != 7_162_403:
        raise ValueError(f'expected original 7.2M SFT architecture, got {parameters}')
    del source
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    decoder = LegalWordDecoder(words, next(model.parameters()).device)
    diagnostic_seed = args.seed + 1_000_003
    rng_devices = [torch.cuda.current_device()] if args.device == 'cuda' else []
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(diagnostic_seed)
        probe = sample_games(reference, validation_secrets, decoder, group_size=GROUP_SIZE)
    with torch.no_grad():
        probe_reference_logits = trajectory_logits(reference, probe)
    start_step = 0
    if args.resume_checkpoint:
        resumed = torch.load(args.resume_checkpoint, map_location='cpu', weights_only=True)
        previous = resumed['grpo']
        if previous['algorithm'] != 'token-clipped-equal-trajectory':
            raise ValueError('resume requires a token-level equal-trajectory checkpoint')
        for name in ('lr', 'kl_beta', 'groups_per_update', 'seed', 'state_corpus', 'validation_corpus',
                     'base_checkpoint', 'mode', 'words', 'device'):
            current = getattr(args, name)
            if previous[name] != (str(current) if isinstance(current, Path) else current):
                raise ValueError(f'resume configuration mismatch: {name}')
        if resumed['base_checkpoint_sha256'] != hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest():
            raise ValueError('resume SFT reference differs from source')
        model.load_state_dict(resumed['model_state_dict'])
        optimizer.load_state_dict(resumed['optimizer_state_dict'])
        secret_rng.setstate(resumed['secret_rng_state'])
        torch.set_rng_state(resumed['torch_rng_state'])
        if rng_devices:
            torch.cuda.set_rng_state_all(resumed['cuda_rng_states'])
        start_step = previous['updates']
        if start_step >= args.updates:
            raise ValueError('requested final update must exceed resumed checkpoint')
        del resumed
    output.mkdir(parents=True, exist_ok=True)
    checkpoints = output / 'checkpoints'
    checkpoints.mkdir(exist_ok=True)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    provenance = hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest()
    scheduled = {start_step, args.updates, *(s for s in MILESTONES if start_step < s <= args.updates)}
    if args.eval_every:
        scheduled.update(range(start_step + args.eval_every, args.updates + 1, args.eval_every))
    write_json(output / 'manifest.json', {
        **config, 'algorithm': 'token-clipped equal-trajectory GRPO', 'parameters': parameters,
        'base_checkpoint_sha256': provenance, 'reference': 'frozen original SFT',
        'group_size': GROUP_SIZE, 'max_turns': MAX_TURNS,
        'reward': '7 - attempts if solved within 6 turns, otherwise 0',
        'secret_sampling': ('uniform states with replacement from fixed training corpus'
                            if training_states else 'uniform with replacement from training split'),
        'starting_state': ('sampled reachable history; same secret and state for all eight members'
                           if training_states else 'empty board; same secret and state for all eight group members'),
        'action_sampling': 'independent stochastic dictionary-constrained decoding; duplicates allowed',
        'likelihood': 'per-generated-token constrained importance ratios; supplied history excluded',
        'objective_reduction': 'mean generated tokens within each trajectory, then mean trajectories',
        'kl_regularizer': 'exact raw categorical KL; mean generated tokens per trajectory, then mean trajectories',
        'optimizer_steps_per_fresh_batch': 1, 'scheduled_checkpoints': sorted(scheduled),
        'checkpoint_ranking': ['validation_wins', 'lower_average_attempts',
                               'lower_mean_action_regret', 'lower_fixed_sft_prefix_kl'],
        'diagnostic_seed': diagnostic_seed,
        'diagnostic_groups': len(validation_secrets),
        'diagnostic_panel': 'eight full games for every validation secret; RNG isolated from training',
        'training_secrets': train_secrets, 'validation_secrets': validation_secrets,
        'test_evaluated': False,
        'training_state_count': len(training_states) if training_states else 0,
        'validation_state_count': len(validation_states) if validation_states else 0,
        'state_corpus_sha256': hashlib.sha256(args.state_corpus.read_bytes()).hexdigest() if training_states else None,
        'validation_corpus_sha256': hashlib.sha256(args.validation_corpus.read_bytes()).hexdigest() if validation_states else None,
        'training_panel_source_indices': [state.source_index for state in training_panel] if training_panel else [],
        'training_panel_seed': args.seed + 2_000_003,
        'gradient_diagnostics': 'step 1 and every 100; actual pre-clipping parameter gradients by length',
    })
    write_json(output / 'reference-rollouts.json', rollout_traces(probe))
    started = time.monotonic()
    best_key = None
    ranking = []
    best_continuation_key = None

    def save(step, path):
        temporary = path.with_suffix('.tmp')
        torch.save({
            'format_version': 1, 'model_state_dict': model.state_dict(),
            'model_config': model.config, 'vocabulary_size': VOCABULARY_SIZE,
            'grpo': {**config, 'updates': step, 'algorithm': 'token-clipped-equal-trajectory'},
            'base_checkpoint': str(args.base_checkpoint), 'base_checkpoint_sha256': provenance,
            'optimizer_state_dict': optimizer.state_dict(),
            'secret_rng_state': secret_rng.getstate(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all() if rng_devices else [],
        }, temporary)
        temporary.replace(path)

    def evaluate(step):
        nonlocal best_key, best_continuation_key
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            torch.manual_seed(diagnostic_seed)
            sampled = sample_games(model, validation_secrets, decoder, group_size=GROUP_SIZE)
            diagnostics = rollout_metrics(sampled)
            diagnostics.update(trajectory_policy_metrics(
                trajectory_logits(model, sampled), trajectory_logits(reference, sampled), sampled,
            ))
            fixed = trajectory_policy_metrics(trajectory_logits(model, probe), probe_reference_logits, probe)
            diagnostics['fixed_sft_prefix_kl_from_sft'] = fixed['raw_prefix_kl_from_sft']
        report = evaluate_games(model, validation_secrets, words, step)
        report.update(diagnostics=diagnostics, sampled_rollouts=rollout_traces(sampled),
                      wall_clock_seconds=time.monotonic() - started)
        if validation_states:
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(diagnostic_seed)
                report['continuations'] = evaluate_continuations(model, reference, validation_states, decoder)
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(diagnostic_seed)
                report['training_continuations'] = evaluate_continuations(model, reference, training_panel, decoder)
        write_json(output / f'validation-{step}.json', report)
        save(step, checkpoints / f'update-{step}.pt')
        key = checkpoint_rank(report)
        ranking.append({'updates': step, 'validation_key': key, 'checkpoint': f'checkpoints/update-{step}.pt'})
        ranking.sort(key=lambda entry: entry['validation_key'], reverse=True)
        write_json(output / 'checkpoint-ranking.json', ranking)
        if best_key is None or key > best_key:
            best_key = key
            save(step, checkpoints / 'best.pt')
            write_json(output / 'best.json', {'updates': step, 'validation_key': key})
        if validation_states:
            continuation_key = (report['continuations']['diagnostics']['mean_trajectory_reward'], *key)
            if best_continuation_key is None or continuation_key > best_continuation_key:
                best_continuation_key = continuation_key
                save(step, checkpoints / 'best-continuation.pt')
                write_json(output / 'best-continuation.json', {
                    'updates': step, 'validation_key': continuation_key,
                })
        gameplay = {k: v for k, v in report['gameplay']['constrained'].items() if k != 'results'}
        print(json.dumps({'event': 'evaluation', 'updates': step, 'constrained': gameplay,
                          'diagnostics': diagnostics, 'validation_key': key,
                          'training_continuation_reward': report.get('training_continuations', {}).get('diagnostics', {}).get('mean_trajectory_reward'),
                          'validation_continuation_reward': report.get('continuations', {}).get('diagnostics', {}).get('mean_trajectory_reward'),
                          'elapsed_seconds': time.monotonic() - started}), flush=True)

    evaluate(start_step)
    with (output / 'metrics.jsonl').open('w') as metrics:
        for step in range(start_step + 1, args.updates + 1):
            if training_states:
                states = [secret_rng.choice(training_states) for _ in range(args.groups_per_update)]
                batch = sample_games(
                    model, [state.secret for state in states], decoder,
                    histories=[state.history for state in states],
                )
            else:
                states = None
                secrets = [secret_rng.choice(train_secrets) for _ in range(args.groups_per_update)]
                batch = sample_games(model, secrets, decoder, group_size=GROUP_SIZE)
            with torch.no_grad():
                ref_logits = trajectory_logits(reference, batch)
            optimizer.zero_grad(set_to_none=True)
            policy_logits = trajectory_logits(model, batch)
            loss, stats = trajectory_loss(policy_logits, ref_logits, batch, args.kl_beta)
            length_diagnostics = (
                trajectory_gradient_metrics(model, policy_logits, ref_logits, batch, args.kl_beta)
                if step == 1 or step % 100 == 0 else None
            )
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite trajectory GRPO loss')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                _, after = trajectory_loss(trajectory_logits(model, batch), ref_logits, batch, args.kl_beta)
            record = {
                'updates': step, 'loss': float(loss.detach()), 'gradient_norm': float(norm),
                **rollout_metrics(batch), 'learning_rate': args.lr,
                'pre_update': stats, 'post_update': after,
                'wall_clock_seconds': time.monotonic() - started,
                'source_state_indices': [state.source_index for state in states] if states else None,
                'starting_depths': [len(state.history) for state in states] if states else None,
                'length_diagnostics': length_diagnostics,
            }
            metrics.write(json.dumps(record) + '\n')
            if states and (step == 1 or step in scheduled):
                traces = rollout_traces(batch)
                for trace in traces:
                    state = states[trace['group']]
                    trace.update(source_index=state.source_index, starting_history=state.history)
                write_json(output / f'training-rollouts-{step}.json', traces)
            metrics.flush()
            if step == 1 or step % args.log_every == 0:
                print(json.dumps(record), flush=True)
            if step in scheduled:
                evaluate(step)
    write_json(output / 'training-complete.json', {
        'updates': args.updates, 'elapsed_seconds': time.monotonic() - started,
    })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, default=BASE_CHECKPOINT)
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--mode', type=Path, default=Path('data/wordle-development.json'))
    parser.add_argument('--words', type=Path, default=DEFAULT_WORDS)
    parser.add_argument('--state-corpus', type=Path)
    parser.add_argument('--validation-corpus', type=Path)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/grpo-games-dev'))
    parser.add_argument('--updates', type=int, default=10_000)
    parser.add_argument('--groups-per-update', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--kl-beta', type=float, default=0.10)
    parser.add_argument('--eval-every', type=int, default=0)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    print(run(parser.parse_args()))


if __name__ == '__main__':
    main()
