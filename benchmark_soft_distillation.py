from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from compare_soft_distillation import paired_games
from cross_validation import aggregate_seed_summaries, combine_fold_predictions, load_mode
from evaluate_v2 import load_v2_model
from soft_evaluate import evaluate_soft_model
from soft_teacher import TeacherDataset, build_teacher_dataset, distribution_statistics
from train_soft_distillation import file_hash, split_indices, train_configuration, write_json
from train_v2 import load_v2_split


def run(args):
    development = json.loads((args.development_root / 'comparison.json').read_text())
    names = set(development['qualifying_configurations'])
    configurations = [row for row in development['all_runs'] if row['name'] in names]
    gate = {'development_comparison': str(args.development_root / 'comparison.json'),
            'sha256': file_hash(args.development_root / 'comparison.json'),
            'criterion': development['benchmark_gate'], 'qualifying_configurations': sorted(names)}
    write_json(args.output_dir / 'gate.json', gate)
    if not configurations:
        result = {**gate, 'status': 'not run: no qualifying development configuration',
                  'test_gameplay_evaluated': False}
        write_json(args.output_dir / 'benchmark-complete.json', result)
        print(json.dumps(result))
        return result
    torch.set_num_threads(4)
    mode = load_mode(args.mode)
    if len(mode.runs) != 5 or tuple(mode.model_seeds) != (0, 1, 2):
        raise ValueError('full benchmark requires existing five folds and seeds 0/1/2')
    if not (args.teacher_dir / 'manifest.json').exists():
        build_teacher_dataset(args.teacher_dir, source=args.source, workers=args.workers,
                              seed=20260924, expected_count=1_000_000)
    dataset = TeacherDataset(args.teacher_dir)
    write_json(args.teacher_dir / 'teacher-statistics.json', distribution_statistics(dataset))
    split_payload = json.loads(args.mode.read_text())['runs']
    all_results = {}
    for configuration in configurations:
        name = configuration['name']
        seed_results = []
        for seed in mode.model_seeds:
            baseline_paths, soft_paths, regret_pairs = [], [], []
            for fold in mode.runs:
                output = args.output_dir / name / f'seed-{seed}' / f'fold-{fold.run}'
                original = args.baseline_root / f'seed-{seed}' / f'fold-{fold.run}' / '7.2m'
                hard = original / 'checkpoints/best.pt'
                initialization = original / 'mechanics/checkpoints/best.pt' if configuration['model'] == 'B' else hard
                mechanics = args.data_dir / f'fold-{fold.run}' / 'mechanics'
                mechanics_train = load_v2_split(mechanics, 'train', example_type='mechanics')
                mechanics_validation = load_v2_split(mechanics, 'validation', example_type='mechanics')
                split = split_payload[fold.run - 1]
                val_indices = split_indices(dataset, fold.validation)
                rng = np.random.default_rng(seed + 700_001)
                panel = np.sort(rng.choice(val_indices, min(args.panel_size, len(val_indices)), replace=False))
                training_args = SimpleNamespace(seed=seed, device=args.device, output_dir=output,
                    microbatch_states=args.microbatch_states, eval_batch_states=args.eval_batch_states,
                    teacher_dir=args.teacher_dir, mode=args.mode, max_epochs=args.max_epochs,
                    patience=args.patience, smoke_steps=0, resume=args.resume,
                    save_every=args.save_every, stop_after_updates=args.stop_after_updates)
                original_hash = file_hash(hard)
                result = train_configuration(training_args, dataset, split, panel, mechanics_train,
                    mechanics_validation, configuration['model'], configuration['temperature'],
                    initialization, configuration['learning_rate'])
                heldout_path = output / 'held-out-evaluation.json'
                baseline_path = output / 'hard-held-out-evaluation.json'
                heldout_indices = split_indices(dataset, fold.test)
                test_rng = np.random.default_rng(seed + 900_001)
                test_panel = np.sort(test_rng.choice(heldout_indices, min(args.panel_size, len(heldout_indices)), replace=False))
                for checkpoint, path in ((hard, baseline_path), (Path(result['checkpoint']), heldout_path)):
                    if not path.exists():
                        model = load_v2_model(checkpoint, args.device)
                        report = evaluate_soft_model(model, dataset, test_panel, configuration['temperature'],
                            fold.test, dataset.words, mechanics_validation, torch.device(args.device),
                            batch_size=args.eval_batch_states, distribution_examples=24)
                        report.update(checkpoint=str(checkpoint), phase='held-out cross-validation',
                                      test_gameplay_evaluated=True)
                        write_json(path, report)
                        del model
                if file_hash(hard) != original_hash:
                    raise RuntimeError('full benchmark baseline checkpoint was modified')
                baseline = json.loads(baseline_path.read_text())
                soft = json.loads(heldout_path.read_text())
                old_actions = baseline['action_quality']['actions']
                new_actions = soft['action_quality']['actions']
                if [r['state_id'] for r in old_actions] != [r['state_id'] for r in new_actions]:
                    raise ValueError('action-regret comparison requires matched observable states')
                regret_pairs.extend({'fold': fold.run, 'state_id': old['state_id'],
                    'sft_regret': old['regret'], 'soft_regret': new['regret'],
                    'regret_difference': new['regret'] - old['regret'],
                    'sft_relative_quality': old['relative_quality'], 'soft_relative_quality': new['relative_quality']}
                    for old, new in zip(old_actions, new_actions))
                for report, label, paths in ((baseline, 'hard', baseline_paths), (soft, 'soft', soft_paths)):
                    path = output / f'{label}-held-out-games.json'
                    write_json(path, report['gameplay'])
                    paths.append(path)
                print(json.dumps({'event': 'held-out fold complete', 'configuration': name,
                                  'seed': seed, 'fold': fold.run, 'soft_wins': soft['gameplay']['wins']}), flush=True)
            old = combine_fold_predictions(mode, baseline_paths)
            new = combine_fold_predictions(mode, soft_paths)
            if old['games'] != 719 or new['games'] != 719:
                raise ValueError('each seed must evaluate exactly 719 held-out secrets')
            paired = paired_games(old, new)
            per_seed = {'seed': seed, 'hard_sft': old, 'soft': new, 'paired': paired,
                        'paired_action_regrets': regret_pairs,
                        'regret_difference_mean': statistics.fmean(r['regret_difference'] for r in regret_pairs),
                        'regret_difference_median': statistics.median(r['regret_difference'] for r in regret_pairs)}
            write_json(args.output_dir / name / f'seed-{seed}' / 'combined.json', per_seed)
            seed_results.append(per_seed)
        aggregate = {'hard_sft': aggregate_seed_summaries([r['hard_sft'] for r in seed_results]),
                     'soft': aggregate_seed_summaries([r['soft'] for r in seed_results]),
                     'paired_by_seed': [{'seed': r['seed'], 'counts': r['paired']['counts'],
                         'mean_attempt_difference_soft_minus_sft': r['paired']['mean_attempt_difference_soft_minus_sft'],
                         'regret_difference_mean': r['regret_difference_mean']} for r in seed_results]}
        write_json(args.output_dir / name / 'aggregate.json', aggregate)
        all_results[name] = aggregate
    result = {**gate, 'results': all_results, 'test_gameplay_evaluated': True}
    write_json(args.output_dir / 'benchmark-complete.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Run only development-qualified soft policies on existing five-fold CV.')
    p.add_argument('--development-root', type=Path, default=Path('runs/soft-distillation-dev'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/soft-distillation-cv5'))
    p.add_argument('--teacher-dir', type=Path, default=Path('data/soft-teacher-cv1m'))
    p.add_argument('--source', type=Path, default=Path('data/wordle-v2-diverse-1m-cv/examples.jsonl.gz'))
    p.add_argument('--mode', type=Path, default=Path('data/wordle-cv5.json'))
    p.add_argument('--baseline-root', type=Path, default=Path('runs/scaling-cv5-1m'))
    p.add_argument('--data-dir', type=Path, default=Path('data/wordle-cv5-1m'))
    p.add_argument('--microbatch-states', type=int, default=128)
    p.add_argument('--eval-batch-states', type=int, default=16)
    p.add_argument('--panel-size', type=int, default=512)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--stop-after-updates', type=int)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--device', default='cuda')
    run(p.parse_args())
