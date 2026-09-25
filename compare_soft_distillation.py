from __future__ import annotations

import argparse
import json
from pathlib import Path

from train_soft_distillation import write_json


def gameplay_key(report):
    game = report['gameplay']
    return (game['wins'], -game['average_attempts'], -game['average_guesses'])


def paired_games(baseline, candidate):
    first = {row['secret']: row for row in baseline['results']}
    second = {row['secret']: row for row in candidate['results']}
    if first.keys() != second.keys():
        raise ValueError('paired gameplay requires identical secrets')
    counts = {'sft_loses_soft_wins': 0, 'sft_wins_soft_loses': 0, 'both_win': 0, 'both_lose': 0}
    rows = []
    for secret, old in first.items():
        new = second[secret]
        if old['won'] and new['won']:
            outcome = 'both_win'
        elif old['won']:
            outcome = 'sft_wins_soft_loses'
        elif new['won']:
            outcome = 'sft_loses_soft_wins'
        else:
            outcome = 'both_lose'
        counts[outcome] += 1
        before = len(old['guesses']) if old['won'] else 6
        after = len(new['guesses']) if new['won'] else 6
        rows.append({'secret': secret, 'outcome': outcome, 'sft_attempts': before,
                     'soft_attempts': after, 'attempt_difference_soft_minus_sft': after - before})
    return {'counts': counts, 'secrets': rows,
            'mean_attempt_difference_soft_minus_sft': sum(r['attempt_difference_soft_minus_sft'] for r in rows) / len(rows)}


def compare(root):
    root = Path(root)
    baseline_payload = json.loads((root / 'hard-sft.json').read_text())
    baseline = baseline_payload['evaluations']['0.5']
    baseline_regret = baseline['action_quality']['summary']['mean_action_regret']
    runs = json.loads((root / 'sweep-complete.json').read_text())['results']
    selected = {}
    all_runs = []
    qualifying = []
    for run in runs:
        path = root / run['name']
        manifest = json.loads((path / 'manifest.json').read_text())
        best = run['best']
        report = json.loads((path / best['report_file']).read_text())
        regret = report['action_quality']['summary']['mean_action_regret']
        gameplay_improved = gameplay_key(report) > gameplay_key(baseline)
        gameplay_not_worse = gameplay_key(report) >= gameplay_key(baseline)
        meaningful_quality = baseline_regret > 0 and regret <= .95 * baseline_regret
        eligible = gameplay_improved or (gameplay_not_worse and meaningful_quality)
        row = {'name': run['name'], 'model': manifest['variant'], 'temperature': manifest['temperature'],
               'learning_rate': manifest['learning_rate'], 'checkpoint': run['checkpoint'],
               'best_epoch': best['epoch'], 'best_step': best['step'],
               'gameplay': {k: v for k, v in report['gameplay'].items() if k != 'results'},
               'action_quality': report['action_quality']['summary'], 'policy': report['policy'],
               'mechanics_validation_loss': report['mechanics_validation_loss'],
               'benchmark_eligible': eligible, 'paired_development_games': paired_games(baseline['gameplay'], report['gameplay'])}
        all_runs.append(row)
        key = (manifest['variant'], manifest['temperature'])
        if key not in selected or tuple(best['key']) > selected[key][0]:
            selected[key] = (tuple(best['key']), row)
        if eligible:
            qualifying.append(row['name'])
    table = [{'name': 'Hard SFT', 'model': 'A', 'temperature': None,
              'checkpoint': baseline_payload['checkpoint'],
              'gameplay': {k: v for k, v in baseline['gameplay'].items() if k != 'results'},
              'action_quality': baseline['action_quality']['summary'],
              'mechanics_validation_loss': baseline['mechanics_validation_loss']}]
    table.extend(selected[key][1] for key in sorted(selected))
    comparison = {'table': table, 'all_runs': all_runs, 'qualifying_configurations': qualifying,
                  'benchmark_gate': 'Lexicographically better constrained wins/attempts/guesses-among-wins, or at least 5% lower exhaustive mean regret with no worse gameplay key.',
                  'source_split_caveat': 'Source secrets held out; full dictionary answer inference matches original SFT.',
                  'test_gameplay_evaluated': False}
    write_json(root / 'comparison.json', comparison)
    columns = ['Model', 'Temperature', 'Wins /72', 'Avg attempts', 'Mean regret', 'Rank-1', 'Top-3', 'Top-8', 'Mechanics loss']
    print('| ' + ' | '.join(columns) + ' |')
    print('|' + '|'.join(['---'] * len(columns)) + '|')
    for row in table:
        g, q = row['gameplay'], row['action_quality']
        values = [row['name'], '—' if row['temperature'] is None else f"{row['temperature']:.2f}",
                  f"{g['wins']}/{g['games']}", f"{g['average_attempts']:.4f}", f"{q['mean_action_regret']:.6f}",
                  *(f"{q[key]:.3%}" for key in ('rank_1_fraction', 'top_3_fraction', 'top_8_fraction')),
                  f"{row['mechanics_validation_loss']:.6f}"]
        print('| ' + ' | '.join(values) + ' |')
    print('Qualifying configurations:', ', '.join(qualifying) or 'none')
    return comparison


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path, nargs='?', default=Path('runs/soft-distillation-dev'))
    compare(parser.parse_args().root)
