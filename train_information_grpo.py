"""Balanced-state, exact expected-information GRPO over 64 proposed guesses."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from evaluate_v2 import load_v2_model
from grpo_information_actions import build_guess_batch, sample_information_actions, sample_policy_guesses
from grpo_information_reward import ExpectedInformationReward
from grpo_information_states import BalancedStateSampler, candidate_bucket, load_state_pool, make_state
from grpo_rollouts import LegalWordDecoder, sample_games
from grpo_trajectory_loss import trajectory_logits, trajectory_loss, trajectory_policy_metrics
from train_grpo import BASE_CHECKPOINT, MILESTONES, write_json
from train_grpo_games import evaluate_games
from tokenizer_v2 import VOCABULARY_SIZE
from wordle import DEFAULT_WORDS, load_words, score_guess


def state_record(state):
    return {'secret': state.secret, 'history': state.history, 'behavior': state.behavior,
            'source_index': state.source_index, 'candidate_count': len(state.candidates),
            'candidates': state.candidates}




@torch.no_grad()
def fresh_model_states(model, secrets, words, decoder, rng, groups, generation):
    """Sample real games and retain unsolved prefixes, never imagined histories."""
    chosen = [rng.choice(secrets) for _ in range(groups)]
    states, seen = [], set()
    for start in range(0, groups, 32):
        batch = sample_games(model, chosen[start:start + 32], decoder, group_size=4)
        for row, guesses in enumerate(batch.guesses):
            secret = batch.secrets[row // 4]
            history = []
            for depth, guess in enumerate(guesses[:5], start=1):
                if guess == secret:
                    break
                history.append((guess, score_guess(secret, guess)))
                key = tuple(history)
                if key in seen:
                    continue
                seen.add(key)
                states.append(make_state(secret, key, 'model', f'model-{generation}-{start * 4 + row}-{depth}', words))
    if not states:
        raise ValueError('model rollouts produced no nonterminal histories')
    return states


def score_batch(batch, states, scorer):
    scores = [scorer.score(state.candidates, guesses) for state, guesses in zip(states, batch.guesses)]
    batch.rewards = torch.as_tensor(np.stack([score.rewards for score in scores]),
                                    dtype=torch.float32, device=batch.inputs.device)
    return scores


def batch_trace(batch, states, scores):
    return [{**state_record(state), 'guesses': guesses, 'proposal_sources': sources,
             'rewards': score.rewards.tolist(), 'information_rewards': score.information.tolist(),
             'expected_candidates_after': score.expected_candidates.tolist(),
             'solve_probabilities': score.solve_probability.tolist()}
            for state, guesses, sources, score in zip(states, batch.guesses, batch.proposal_sources, scores)]


@torch.no_grad()
def evaluate_information(model, reference, states, decoder, scorer, oracle_cache):
    """Actual policy samples (with replacement), not the forced proposal mixture."""
    traces = []
    raw_kl = constrained_kl = 0.0
    for start in range(0, len(states), 32):
        chunk = states[start:start + 32]
        stochastic = sample_policy_guesses(model, chunk, decoder, samples=8)
        greedy = sample_policy_guesses(model, chunk, decoder, samples=1, greedy=True)
        batch = build_guess_batch(model, chunk, decoder, stochastic)
        metrics = trajectory_policy_metrics(trajectory_logits(model, batch),
                                            trajectory_logits(reference, batch), batch)
        raw_kl += metrics['raw_prefix_kl_from_sft'] * len(chunk)
        constrained_kl += metrics['constrained_prefix_kl_from_sft'] * len(chunk)
        for state, samples, greedy_words in zip(chunk, stochastic, greedy):
            scores = scorer.score(state.candidates, (*samples, greedy_words[0]))
            if state.candidates not in oracle_cache:
                all_scores = scorer.score(state.candidates, scorer.words)
                index = int(all_scores.rewards.argmax())
                oracle_cache[state.candidates] = (scorer.words[index], float(all_scores.rewards[index]))
            best_guess, best_reward = oracle_cache[state.candidates]
            traces.append({**state_record(state), 'samples': samples,
                           'sample_rewards': scores.rewards[:-1].tolist(),
                           'greedy_guess': greedy_words[0], 'greedy_reward': float(scores.rewards[-1]),
                           'greedy_information_reward': float(scores.information[-1]),
                           'greedy_solve_probability': float(scores.solve_probability[-1]),
                           'greedy_expected_candidates_after': float(scores.expected_candidates[-1]),
                           'oracle_guess': best_guess, 'oracle_reward': best_reward})
    def summarize(rows):
        return {'states': len(rows),
                'stochastic_mean_reward': float(np.mean([np.mean(r['sample_rewards']) for r in rows])),
                'greedy_mean_reward': float(np.mean([r['greedy_reward'] for r in rows])),
                'greedy_mean_reward_regret': float(np.mean([r['oracle_reward'] - r['greedy_reward'] for r in rows])),
                'greedy_mean_information_reward': float(np.mean([r['greedy_information_reward'] for r in rows])),
                'greedy_mean_solve_probability': float(np.mean([r['greedy_solve_probability'] for r in rows]))}
    summary = summarize(traces)
    summary.update(raw_prefix_kl_from_sft=raw_kl / len(states),
                   constrained_prefix_kl_from_sft=constrained_kl / len(states))
    strata = {}
    for dimension, key in (('difficulty', lambda r: candidate_bucket(r['candidate_count'])),
                           ('history_depth', lambda r: str(len(r['history']))),
                           ('behavior', lambda r: r['behavior'])):
        buckets = defaultdict(list)
        for trace in traces:
            buckets[key(trace)].append(trace)
        strata[dimension] = {name: summarize(rows) for name, rows in sorted(buckets.items())}
    return {'summary': summary, 'strata': strata, 'states': traces}


def run(args):
    if min(args.updates, args.groups_per_update, args.microbatch_groups, args.panel_size,
           args.model_refresh_every, args.initial_model_groups, args.refresh_model_groups) < 1:
        raise ValueError('run sizes and refresh interval must be positive')
    if args.lr <= 0 or args.kl_beta < 0:
        raise ValueError('learning rate must be positive and KL beta nonnegative')
    output = args.output_dir
    if (output / 'manifest.json').exists():
        raise ValueError('choose a new experiment output directory')
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    words = tuple(load_words(args.words))
    split = json.loads(args.mode.read_text())['runs'][0]
    if set(split['train']) & (set(split['validation']) | set(split['test'])):
        raise ValueError('source secret splits overlap')
    source = torch.load(args.base_checkpoint, map_location='cpu', weights_only=True)
    if 'grpo' in source or 'dpo' in source:
        raise ValueError('start from original SFT, not an RL/DPO checkpoint')
    del source
    model = load_v2_model(args.base_checkpoint, args.device).eval()
    if sum(p.numel() for p in model.parameters()) != 7_162_403:
        raise ValueError('expected original 7.2M SFT architecture')
    reference = copy.deepcopy(model).requires_grad_(False).eval()
    decoder = LegalWordDecoder(words, next(model.parameters()).device)
    scorer = ExpectedInformationReward(words)
    train_states = load_state_pool(args.corpus / 'train.jsonl', words)
    validation_states = load_state_pool(args.corpus / 'validation.jsonl', words)
    for states, allowed in ((train_states, set(split['train'])), (validation_states, set(split['validation']))):
        if not states or any(state.secret not in allowed for state in states):
            raise ValueError('state pool source secret leakage or empty pool')
    rng_devices = [torch.cuda.current_device()] if args.device == 'cuda' else []
    model_rng, action_rng = random.Random(args.seed + 10), random.Random(args.seed + 20)
    initial_model_states = fresh_model_states(model, split['train'], words, decoder, model_rng,
                                             args.initial_model_groups, 'sft-train')
    sampler = BalancedStateSampler(train_states, words, args.seed, opening_fraction=.075)
    sampler.replace_model_states(initial_model_states)
    train_panel_sampler = BalancedStateSampler(train_states, words, args.seed + 30, opening_fraction=.075)
    train_panel_sampler.replace_model_states(initial_model_states)
    train_panel = train_panel_sampler.sample_batch(args.panel_size)
    with torch.random.fork_rng(devices=rng_devices):
        torch.manual_seed(args.seed + 40)
        validation_model_states = fresh_model_states(reference, split['validation'], words, decoder,
            random.Random(args.seed + 40), args.initial_model_groups, 'sft-validation')
    validation_sampler = BalancedStateSampler(validation_states, words, args.seed + 50, opening_fraction=.075)
    validation_sampler.replace_model_states(validation_model_states)
    validation_panel = validation_sampler.sample_batch(args.panel_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    provenance = hashlib.sha256(args.base_checkpoint.read_bytes()).hexdigest()
    schedule = sorted({0, args.updates, *(step for step in MILESTONES if step <= args.updates)})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'checkpoints').mkdir(exist_ok=True)
    write_json(output / 'manifest.json', {**config, 'algorithm': 'expected-information candidate-ranking GRPO',
        'base_checkpoint_sha256': provenance, 'group_size': 64, 'policy_proposals': 48, 'random_proposals': 16,
        'reward': 'log(N / (sum_feedback_bucket_sizes_squared / N)) + 2 * indicator(guess in candidates) / N',
        'candidate_universe': 'all legal dictionary answers consistent with history, including held-out secret words',
        'source_secret_used_for_reward': False, 'source_secret_restriction': 'train only for training state histories',
        'proposal_objective': 'equal-candidate ranking surrogate; mixed and deduplicated proposals are not unbiased on-policy PPO',
        'ratio': 'token-level current/base-behavior-policy ratio under original full-dictionary prefix masks',
        'refill': 'exclude accepted words during proposal refill only; update policy never excludes other candidates',
        'loss_tokens': 'only five newly proposed guess letters; mean tokens then equal candidate guesses',
        'kl': 'raw categorical KL to frozen original SFT, evaluated at the five action prefixes',
        'opening_fraction': .075, 'scheduled_checkpoints': schedule,
        'train_pool_sha256': hashlib.sha256((args.corpus / 'train.jsonl').read_bytes()).hexdigest(),
        'validation_pool_sha256': hashlib.sha256((args.corpus / 'validation.jsonl').read_bytes()).hexdigest(),
        'initial_training_coverage': sampler.coverage(), 'validation_coverage': validation_sampler.coverage(),
        'initial_model_states': len(initial_model_states), 'validation_model_states': len(validation_model_states),
        'test_gameplay_evaluated': False, 'held_out_caveat': 'held-out source secrets are excluded from training histories, not hypothetical reward answers'})
    write_json(output / 'training-panel.json', [state_record(s) for s in train_panel])
    write_json(output / 'validation-panel.json', [state_record(s) for s in validation_panel])
    write_json(output / 'model-states-0.json', [state_record(s) for s in initial_model_states])
    oracle_cache = {}
    best_information = best_gameplay = None
    started = time.monotonic()

    def save(step, name):
        torch.save({'format_version': 1, 'model_state_dict': model.state_dict(),
                    'model_config': model.config, 'vocabulary_size': VOCABULARY_SIZE,
                    'grpo': {**config, 'updates': step, 'algorithm': 'expected-information'},
                    'base_checkpoint': str(args.base_checkpoint), 'base_checkpoint_sha256': provenance,
                    'optimizer_state_dict': optimizer.state_dict()}, output / 'checkpoints' / name)

    def evaluate(step):
        nonlocal best_information, best_gameplay
        panels = {}
        for name, states in (('training_information', train_panel), ('validation_information', validation_panel)):
            with torch.random.fork_rng(devices=rng_devices):
                torch.manual_seed(args.seed + 1_000_003)
                panels[name] = evaluate_information(model, reference, states, decoder, scorer, oracle_cache)
        report = evaluate_games(model, split['validation'], words, step)
        report.update(panels)
        report['sampling_coverage'] = sampler.coverage()
        report['elapsed_seconds'] = time.monotonic() - started
        write_json(output / f'validation-{step}.json', report)
        save(step, f'update-{step}.pt')
        val = panels['validation_information']['summary']
        gameplay = report['gameplay']['constrained']
        game_key = (gameplay['wins'], -gameplay['average_attempts'],
                    -report['action_regret']['constrained']['summary']['mean_action_regret'],
                    -val['raw_prefix_kl_from_sft'])
        information_key = (val['greedy_mean_reward'], val['stochastic_mean_reward'], *game_key)
        if best_information is None or information_key > best_information:
            best_information = information_key
            save(step, 'best-information.pt')
            write_json(output / 'best-information.json', {'updates': step, 'key': information_key})
        if best_gameplay is None or game_key > best_gameplay:
            best_gameplay = game_key
            save(step, 'best-gameplay.pt')
            write_json(output / 'best-gameplay.json', {'updates': step, 'key': game_key})
        print(json.dumps({'event': 'evaluation', 'updates': step,
            'training': panels['training_information']['summary'], 'validation': val,
            'greedy_wins': report['gameplay']['constrained']['wins'],
            'greedy_attempts': report['gameplay']['constrained']['average_attempts'],
            'elapsed_seconds': time.monotonic() - started}), flush=True)

    evaluate(0)
    with (output / 'metrics.jsonl').open('w') as metrics:
        for step in range(1, args.updates + 1):
            states = sampler.sample_batch(args.groups_per_update)
            batch = sample_information_actions(model, states, decoder, action_rng)
            scores = score_batch(batch, states, scorer)
            optimizer.zero_grad(set_to_none=True)
            loss_value, statistics = 0.0, Counter()
            for start in range(0, len(states), args.microbatch_groups):
                end = min(start + args.microbatch_groups, len(states))
                part = batch.slice_groups(start, end)
                with torch.no_grad():
                    reference_logits = trajectory_logits(reference, part)
                loss, stats = trajectory_loss(trajectory_logits(model, part), reference_logits, part, args.kl_beta)
                if not torch.isfinite(loss):
                    raise FloatingPointError('nonfinite information GRPO loss')
                weight = (end - start) / len(states)
                (loss * weight).backward()
                loss_value += float(loss.detach()) * weight
                for key, value in stats.items():
                    statistics[key] += value * weight
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            rewards = batch.rewards
            record = {'updates': step, 'loss': loss_value, 'gradient_norm': float(norm),
                'proposal_mean_reward': float(rewards.mean()),
                'mean_group_reward_std': float(rewards.std(dim=1, correction=0).mean()),
                'identical_reward_group_rate': float((rewards.amax(1) == rewards.amin(1)).float().mean()),
                'pre_update': dict(statistics), 'proposal_stats': batch.proposal_stats,
                'states': [{'source_index': s.source_index, 'behavior': s.behavior,
                            'history_depth': len(s.history), 'candidate_count': len(s.candidates)} for s in states],
                'elapsed_seconds': time.monotonic() - started}
            metrics.write(json.dumps(record) + '\n')
            metrics.flush()
            if step == 1 or step in schedule:
                write_json(output / f'actions-{step}.json', batch_trace(batch, states, scores))
            if step == 1 or step % 100 == 0:
                print(json.dumps(record), flush=True)
            if step in schedule:
                evaluate(step)
            if step % args.model_refresh_every == 0 and step < args.updates:
                fresh = fresh_model_states(model, split['train'], words, decoder, model_rng,
                                           args.refresh_model_groups, step)
                sampler.replace_model_states(fresh)
                write_json(output / f'model-states-{step}.json', [state_record(s) for s in fresh])
    write_json(output / 'training-complete.json', {'updates': args.updates,
        'elapsed_seconds': time.monotonic() - started, 'sampling_coverage': sampler.coverage()})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-checkpoint', type=Path, default=BASE_CHECKPOINT)
    parser.add_argument('--corpus', type=Path, default=Path('data/grpo-information'))
    parser.add_argument('--mode', type=Path, default=Path('data/wordle-development.json'))
    parser.add_argument('--words', type=Path, default=DEFAULT_WORDS)
    parser.add_argument('--output-dir', type=Path, default=Path('runs/grpo-information-dev'))
    parser.add_argument('--updates', type=int, default=1000)
    parser.add_argument('--groups-per-update', type=int, default=16)
    parser.add_argument('--microbatch-groups', type=int, default=2)
    parser.add_argument('--panel-size', type=int, default=400)
    parser.add_argument('--initial-model-groups', type=int, default=256)
    parser.add_argument('--refresh-model-groups', type=int, default=64)
    parser.add_argument('--model-refresh-every', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-6)
    parser.add_argument('--kl-beta', type=float, default=.10)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    print(run(parser.parse_args()))


if __name__ == '__main__':
    main()
