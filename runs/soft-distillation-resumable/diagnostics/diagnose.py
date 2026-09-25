from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict

sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from evaluate_v2 import evaluate_model
from model import WordleGPT
from soft_policy import candidate_sequence_logps
from soft_teacher import TeacherDataset, teacher_probabilities, exhaustive_scores, feedback_matrix
from tokenizer_v2 import encode, decode
from tokenizer import serialize_trajectory, END_TOKEN, GUESS_TOKEN
from train import generate_constrained_guess

ROOT = Path('runs/soft-distillation-resumable')
OUT = ROOT / 'diagnostics'
DATA = TeacherDataset('data/soft-teacher-1m')
WORDS = DATA.words
WORD_IDS = {word: index for index, word in enumerate(WORDS)}
WORD_TOKENS = np.asarray([encode(word) for word in WORDS])
PANEL = np.asarray(json.loads((ROOT / 'panel-indices.json').read_text()))
SPLIT = json.loads(Path('data/wordle-development.json').read_text())['runs'][0]
assert all(WORDS[int(DATA.source_ids[i])] in SPLIT['validation'] for i in PANEL)
SETS = np.load(DATA.root / 'answer_set_ids.npy', mmap_mode='r')[PANEL]
COUNTS = np.array(DATA.remaining_counts[PANEL])
CANDIDATES = np.array(DATA.candidate_ids[PANEL], dtype=np.int64)
COSTS = np.load(DATA.root / 'answer_set_numerators.npy', mmap_mode='r')[SETS].astype(np.float64) / COUNTS[:, None]
RANKS = np.load(DATA.root / 'answer_set_ranks.npy', mmap_mode='r')[SETS]
OFFSETS = np.load(DATA.root / 'answer_set_offsets.npy', mmap_mode='r')
MEMBERS = np.load(DATA.root / 'answer_set_members.npy', mmap_mode='r')
REMAINING = np.zeros_like(COSTS, dtype=bool)
for row, set_id in enumerate(SETS):
    REMAINING[row, MEMBERS[OFFSETS[set_id]:OFFSETS[set_id + 1]]] = True
assert np.array_equal(REMAINING.sum(axis=1), COUNTS)
TEACHER = teacher_probabilities(torch.tensor(COSTS), torch.tensor(COUNTS.astype(np.int64)), torch.tensor(REMAINING), .25).numpy()
FB = feedback_matrix(WORDS)
BEST_COST_CACHE = {}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prompt(history):
    return encode('<P>' + serialize_trajectory(history)[:-len(END_TOKEN)] + GUESS_TOKEN)


def conditional_greedy(probabilities):
    selected = np.arange(len(WORDS))
    for position in range(5):
        mass = np.bincount(WORD_TOKENS[selected, position], weights=probabilities[selected], minlength=26)
        selected = selected[WORD_TOKENS[selected, position] == mass.argmax()]
    assert len(selected) == 1
    return int(selected[0])


@torch.inference_mode()
def score_prompts(model, prompts):
    result = []
    word_tokens = torch.tensor(WORD_TOKENS, device='cuda')
    for start in range(0, len(prompts), 8):
        values = prompts[start:start + 8]
        lengths = torch.tensor([len(row) for row in values], device='cuda')
        inputs = torch.zeros((len(values), int(lengths.max())), dtype=torch.long, device='cuda')
        for row, tokens in enumerate(values):
            inputs[row, :len(tokens)] = torch.tensor(tokens, device='cuda')
        result.append(candidate_sequence_logps(model, inputs, lengths, word_tokens[None].expand(len(values), -1, -1)).cpu())
    return torch.cat(result).double().numpy()


def normalize(logps):
    maximum = logps.max(axis=-1, keepdims=True)
    values = np.exp(logps - maximum)
    totals = values.sum(axis=-1, keepdims=True)
    return values / totals, (maximum + np.log(totals)).squeeze(-1)


def summarize(logps, greedy_ids, mask):
    rows = np.flatnonzero(mask)
    if not len(rows):
        return {'states': 0}
    scores, counts = logps[rows], COUNTS[rows]
    probabilities, log_legal_mass = normalize(scores)
    candidate_probabilities = np.take_along_axis(probabilities, CANDIDATES[rows], axis=1)
    candidate_mass = candidate_probabilities.sum(axis=1)
    teacher_candidate_mass = np.take_along_axis(TEACHER[rows], CANDIDATES[rows], axis=1).sum(axis=1)
    choices = {'token_greedy': greedy_ids[rows], 'sequence_argmax': scores.argmax(axis=1),
               'conditional_token_greedy': np.asarray([conditional_greedy(p) for p in probabilities]),
               'candidate_argmax': CANDIDATES[rows, candidate_probabilities.argmax(axis=1)]}
    summary = {'states': len(rows), 'mean_raw_legal_mass': float(np.exp(log_legal_mass).mean()),
               'median_raw_legal_mass': float(np.median(np.exp(log_legal_mass))),
               'mean_log_raw_legal_mass': float(log_legal_mass.mean()),
               'mean_mass_outside_candidates_given_legal': float((1 - candidate_mass).mean()),
               'teacher_mass_outside_candidates_full_dictionary': float((1 - teacher_candidate_mass).mean()),
               'mean_probability_on_remaining_answers_given_legal': float((probabilities * REMAINING[rows]).sum(axis=1).mean()),
               'mean_probability_on_remaining_answers_given_candidates': float((candidate_probabilities * DATA.is_remaining[PANEL[rows]]).sum(axis=1).__truediv__(candidate_mass).mean()),
               'token_vs_sequence_disagreement_fraction': float(np.mean(choices['token_greedy'] != choices['sequence_argmax'])),
               'decoders': {}}
    for name, selected in choices.items():
        cost = COSTS[rows, selected]
        ranks = RANKS[rows, selected]
        in_answers = REMAINING[rows, selected]
        in_candidates = (CANDIDATES[rows] == selected[:, None]).any(axis=1)
        singleton = counts == 1
        summary['decoders'][name] = {
            'mean_regret': float((cost - COSTS[rows].min(axis=1)).mean()),
            'mean_relative_quality': float((cost / COSTS[rows].min(axis=1)).mean()),
            'rank1_fraction': float((ranks == 1).mean()), 'top3_fraction': float((ranks <= 3).mean()),
            'top8_fraction': float((ranks <= 8).mean()), 'remaining_answer_fraction': float(in_answers.mean()),
            'outside_candidate_fraction': float((~in_candidates).mean()),
            'singleton_accuracy': float(in_answers[singleton].mean()) if singleton.any() else None,
        }
    return summary


def annotate(gameplay):
    singleton_turns = singleton_misses = repeated_turns = all_turns = 0
    for game in gameplay['results']:
        remaining = np.arange(len(WORDS))
        secret_id = WORD_IDS[game['secret']]
        seen = set()
        details = []
        for guess in game['guesses']:
            guess_id = WORD_IDS[guess]
            n = len(remaining)
            all_turns += 1
            repeated_turns += guess in seen
            singleton_turns += n == 1
            singleton_misses += n == 1 and guess != game['secret']
            key = tuple(map(int, remaining))
            if key not in BEST_COST_CACHE:
                numerators, _ = exhaustive_scores(FB, remaining)
                BEST_COST_CACHE[key] = float(numerators.min() / n)
            buckets = np.bincount(FB[guess_id, remaining].astype(np.int64), minlength=243)
            cost = float((buckets * buckets).sum() / n)
            details.append({'guess': guess, 'remaining_before': n, 'is_remaining': bool(guess_id in remaining),
                            'repeated': guess in seen, 'expected_survivors': cost,
                            'regret': cost - BEST_COST_CACHE[key]})
            seen.add(guess)
            remaining = remaining[FB[guess_id, remaining] == FB[guess_id, secret_id]]
        game['turn_diagnostics'] = details
    gameplay['diagnostics'] = {'turns': all_turns, 'repeated_turns': repeated_turns,
                               'singleton_turns': singleton_turns, 'singleton_misses': singleton_misses}
    return gameplay


@torch.inference_mode()
def sequence_games(model, mode, cache):
    histories = [[] for _ in SPLIT['validation']]
    guesses = [[] for _ in histories]
    won = [False] * len(histories)
    for turn in range(6):
        active = [i for i in range(len(histories)) if not won[i]]
        keys = {i: tuple(prompt(histories[i])) for i in active}
        missing = list(dict.fromkeys(key for key in keys.values() if key not in cache))
        if missing:
            logps = score_prompts(model, missing)
            probs, masses = normalize(logps)
            for key, scores, probabilities in zip(missing, logps, probs):
                cache[key] = {'sequence_argmax': int(scores.argmax()),
                              'conditional_token_greedy': conditional_greedy(probabilities)}
        for i in active:
            word = WORDS[cache[keys[i]][mode]]
            guesses[i].append(word)
            secret = SPLIT['validation'][i]
            if word == secret:
                won[i] = True
            else:
                from wordle import score_guess
                histories[i].append({'guess': word, 'feedback': score_guess(secret, word)})
    wins = sum(won)
    return annotate({'decode': mode, 'games': len(histories), 'wins': wins,
        'average_attempts': sum(map(len, guesses)) / len(histories),
        'average_guesses': sum(len(row) for row, solved in zip(guesses, won) if solved) / wins if wins else 0,
        'invalid_guesses': 0,
        'results': [{'secret': secret, 'guesses': row, 'won': solved} for secret, row, solved in zip(SPLIT['validation'], guesses, won)]})


def run():
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    OUT.mkdir(exist_ok=True)
    checkpoints = {
        'hard_sft': Path('runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt'),
        'soft_best': ROOT / 'B-T0.25/checkpoints/best.pt',
        'soft_stopped': ROOT / 'B-T0.25/resume.pt',
    }
    prompts = [DATA.prompts[i, :DATA.lengths[i]].tolist() for i in PANEL]
    report = {'training_performed': False, 'validation_secrets': 72, 'panel_states': len(PANEL),
              'teacher_temperature': .25, 'models': {},
              'probability_definition': 'raw full-vocabulary five-letter probabilities; legal mass sums all719 words; outside-candidate mass conditioned on legal output',
              'decoding_definitions': {'token_greedy': 'existing per-prefix legal-mask greedy',
                  'sequence_argmax': 'maximum raw five-letter log probability over all719 legal words',
                  'conditional_token_greedy': 'greedy letters using prefix sums of the full719 joint distribution',
                  'candidate_argmax': 'maximum sequence probability among the128 training candidates; panel diagnostic only'}}
    for name, path in checkpoints.items():
        checksum = sha(path)
        payload = torch.load(path, map_location='cpu', weights_only=True)
        config = payload.get('model_config') or payload['best_checkpoint']['model_config']
        model = WordleGPT(**config).to('cuda')
        model.load_state_dict(payload['model_state_dict'])
        model.eval().requires_grad_(False)
        step = payload.get('step', payload.get('soft_distillation', {}).get('step'))
        del payload
        with torch.inference_mode():
            logps = score_prompts(model, prompts)
            greedy = np.asarray([WORD_IDS[decode(generate_constrained_guess(model, tokens, frozenset(WORDS)))] for tokens in prompts])
            strata = {'all': np.ones(len(PANEL), dtype=bool), 'singleton': COUNTS == 1,
                      'N2': COUNTS == 2, 'N3-5': (COUNTS >= 3) & (COUNTS <= 5),
                      'N6-20': (COUNTS >= 6) & (COUNTS <= 20), 'N21-100': (COUNTS >= 21) & (COUNTS <= 100),
                      'N101+': COUNTS >= 101}
            summaries = {key: summarize(logps, greedy, mask) for key, mask in strata.items()}
            gameplay = {'token_greedy': annotate(asdict(evaluate_model(model, SPLIT['validation'], WORDS, decode='constrained')))}
            cache = {}
            for mode in ('sequence_argmax', 'conditional_token_greedy'):
                gameplay[mode] = sequence_games(model, mode, cache)
        probabilities, legal_mass = normalize(logps)
        rows = []
        for row, index in enumerate(PANEL):
            candidate_mass = probabilities[row, CANDIDATES[row]].sum()
            choices = {'token_greedy': int(greedy[row]), 'sequence_argmax': int(logps[row].argmax()),
                       'conditional_token_greedy': conditional_greedy(probabilities[row]),
                       'candidate_argmax': int(CANDIDATES[row, logps[row, CANDIDATES[row]].argmax()])}
            rows.append({'dataset_index': int(index), 'state_id': int(DATA.state_ids[index]),
                         'prompt': decode(prompts[row]), 'remaining_answers': [WORDS[i] for i in np.flatnonzero(REMAINING[row])],
                         'candidate_count': int(COUNTS[row]), 'raw_legal_mass': float(np.exp(legal_mass[row])),
                         'outside_candidate_legal_mass': float(1 - candidate_mass),
                         'choices': {key: {'word': WORDS[i], 'rank': int(RANKS[row, i]), 'cost': float(COSTS[row, i]),
                             'probability_given_legal': float(probabilities[row, i]),
                             'in_candidates': bool(i in CANDIDATES[row]), 'in_remaining': bool(REMAINING[row, i])}
                             for key, i in choices.items()},
                         'remaining_probability_given_legal': float(probabilities[row, REMAINING[row]].sum()),
                         'remaining_probability_given_candidates': float(probabilities[row, CANDIDATES[row]][DATA.is_remaining[index]].sum() / candidate_mass),
                         'top_words': [{'word': WORDS[i], 'probability_given_legal': float(probabilities[row, i]),
                                        'rank': int(RANKS[row, i]), 'in_candidates': bool(i in CANDIDATES[row])}
                                       for i in np.argsort(-probabilities[row])[:8]]})
        result = {'checkpoint': str(path), 'sha256': checksum, 'step': step,
                  'panel': summaries, 'gameplay': gameplay, 'states': rows}
        assert sha(path) == checksum
        write(OUT / f'{name}.json', result)
        np.savez_compressed(OUT / f'{name}-scores.npz', logps=logps, greedy_ids=greedy, panel_indices=PANEL)
        report['models'][name] = {key: value for key, value in result.items() if key != 'states'}
        write(OUT / 'comparison.json', report)
        print(json.dumps({'model': name, 'step': step, 'panel': summaries['all'],
                          'gameplay': {key: {k: v for k, v in value.items() if k not in ('results',)} for key, value in gameplay.items()}}), flush=True)
        del model
        torch.cuda.empty_cache()
    print(str(OUT / 'comparison.json'))


if __name__ == '__main__':
    run()
