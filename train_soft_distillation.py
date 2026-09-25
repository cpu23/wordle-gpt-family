from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

from evaluate_v2 import load_v2_model
from experiments_replay import even_replay_schedule, replay_batch_counts
from soft_evaluate import checkpoint_key, evaluate_soft_model
from soft_policy import candidate_sequence_logps, distillation_loss
from soft_teacher import TeacherDataset, teacher_probabilities
from tokenizer_v2 import encode
from train import IGNORE_INDEX
from train_v2 import load_v2_split

BASE_ROOT = Path('runs/scaling-dev-1m/seed-0/fold-1/7.2m')
TEMPERATURES = (0.25, 0.50, 1.00)
RATIOS = {'expert': .95, 'mechanics': .05, 'consistency': 0.0}
REQUIRED_PARAMETERS = 7_162_403
EARLY_C_EVALUATIONS = (100, 250, 1000)
RESUME_VERSION = 1


@contextmanager
def atomic_file(path, mode):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(descriptor, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path, value):
    with atomic_file(path, 'w') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def save_torch(path, value):
    with atomic_file(path, 'wb') as stream:
        torch.save(value, stream)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def stop_signals():
    stop = {'requested': False}
    previous = {}

    def request_stop(signum, frame):
        stop['requested'] = True

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, request_stop)
        yield stop
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def split_indices(dataset, split):
    ids = [dataset.words.index(word) for word in split]
    return np.flatnonzero(np.isin(dataset.source_ids, ids))


def policy_batch(dataset, indices, word_tokens, device, temperature):
    def tensor(array, dtype=torch.long):
        return torch.as_tensor(np.array(array[indices], copy=True), dtype=dtype, device=device)
    lengths = tensor(dataset.lengths)
    prompts = tensor(dataset.prompts)[:, :int(lengths.max())]
    candidates = word_tokens[tensor(dataset.candidate_ids)]
    ranks = tensor(dataset.ranks)
    teacher = teacher_probabilities(tensor(dataset.costs, torch.float32),
                                    tensor(dataset.remaining_counts),
                                    tensor(dataset.is_remaining, torch.bool), temperature)
    return prompts, lengths, candidates, ranks, teacher


def mechanics_identity(*splits):
    digest = hashlib.sha256()
    for data in splits:
        for field in ('inputs', 'targets'):
            array = getattr(data, field).detach().cpu().contiguous().numpy()
            digest.update(str((array.shape, array.dtype.str)).encode())
            digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()


def configuration(args, dataset, split, panel, mechanics_train, mechanics_validation,
                  variant, temperature, initialization, learning_rate):
    training = split_indices(dataset, split['train'])
    expanded = np.repeat(training, np.asarray(dataset.weights[training], dtype=np.int64))
    policy_steps = args.smoke_steps or math.ceil(len(expanded) / 128)
    if not len(expanded) or policy_steps < 1:
        raise ValueError('training states and policy steps must be positive')
    counts = replay_batch_counts(policy_steps, RATIOS)
    if args.smoke_steps:
        counts['mechanics'] = max(1, counts['mechanics'])
    config = {'resume_version': RESUME_VERSION,
              'variant': variant, 'temperature': temperature, 'initialization': str(initialization),
              'initialization_sha256': file_hash(initialization),
              'parameters': REQUIRED_PARAMETERS, 'seed': args.seed, 'learning_rate': learning_rate,
              'effective_batch_states': 128, 'microbatch_states': args.microbatch_states,
              'optimizer': 'AdamW', 'weight_decay': .01,
              'ratios': RATIOS, 'epoch_batches': counts, 'weighted_training_states': len(expanded),
              'training_source_states': len(training), 'max_epochs': args.max_epochs,
              'patience': args.patience, 'smoke_steps': args.smoke_steps,
              'teacher_dataset': str(args.teacher_dir), 'mode': str(args.mode),
              'teacher_manifest_sha256': file_hash(Path(args.teacher_dir) / 'manifest.json'),
              'split': split, 'panel_indices': np.asarray(panel).tolist(),
              'mechanics_sha256': mechanics_identity(mechanics_train, mechanics_validation),
              'eval_batch_states': args.eval_batch_states, 'device': str(args.device),
              'deterministic_algorithms': True,
              'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG', ':4096:8'),
              'candidate_probability': 'softmax over raw-vocabulary five-letter sequence log probabilities',
              'checkpoint_selection': ['maximum constrained wins', 'minimum constrained average attempts',
                  'minimum average guesses among wins', 'lower exhaustive expected-survivor regret',
                  'lower teacher-to-student KL'],
              'held_out_caveat': 'Source-secret splits are strict; possible answers use the full dictionary, as in original expert SFT.',
              'early_C_evaluations': list(EARLY_C_EVALUATIONS),
              'rapid_drift_rule': 'At step250, at least five fewer constrained wins than initialization; retry that temperature from original SFT at LR3e-6.',
              'test_gameplay_evaluated': False}
    # The serialized representation is the compatibility contract (including tuple/list normalization).
    return json.loads(json.dumps(config)), expanded, policy_steps, counts


def configuration_name(variant, temperature, learning_rate):
    return f'{variant}-T{temperature:.2f}' + (f'-lr{learning_rate:g}' if learning_rate == 3e-6 else '')


def validate_existing(output, config, resume):
    manifest = output / 'manifest.json'
    completion = output / 'training-complete.json'
    checkpoint = output / 'resume.pt'
    if not output.exists() or not any(output.iterdir()):
        return
    if not completion.exists() and not checkpoint.exists():
        raise ValueError(f'legacy incomplete run: optimizer state unavailable; cannot resume or overwrite {output}')
    if not manifest.exists():
        raise ValueError(f'existing run has no manifest; refusing to overwrite {output}')
    existing = json.loads(manifest.read_text())
    mismatches = [key for key, value in config.items() if existing.get(key) != value]
    if mismatches:
        raise ValueError(f'incompatible existing run {output}: {", ".join(mismatches)}')
    if not completion.exists() and not resume:
        raise ValueError(f'incomplete existing run requires explicit --resume: {output}')


def deterministic_execution():
    # CUDA attention backward otherwise introduces tiny gradient differences that
    # Adam can amplify when a resumed process has a different allocation layout.
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in (':4096:8', ':16:8'):
        raise ValueError('resumable training requires a deterministic CUBLAS_WORKSPACE_CONFIG')
    torch.use_deterministic_algorithms(True)


def train_configuration(args, dataset, split, panel, mechanics_train, mechanics_validation,
                        variant, temperature, initialization, learning_rate, *, _stop=None):
    deterministic_execution()
    if _stop is None:
        with stop_signals() as stop:
            return _train_configuration(args, dataset, split, panel, mechanics_train,
                                        mechanics_validation, variant, temperature, initialization,
                                        learning_rate, stop)
    return _train_configuration(args, dataset, split, panel, mechanics_train,
                                mechanics_validation, variant, temperature, initialization,
                                learning_rate, _stop)


def _train_configuration(args, dataset, split, panel, mechanics_train, mechanics_validation,
                         variant, temperature, initialization, learning_rate, stop):
    save_every = getattr(args, 'save_every', 100)
    stop_after = getattr(args, 'stop_after_updates', None)
    if save_every < 1 or (stop_after is not None and stop_after < 0):
        raise ValueError('save_every must be positive and stop_after_updates must be nonnegative')
    name = configuration_name(variant, temperature, learning_rate)
    output = args.output_dir / name
    completion = output / 'training-complete.json'
    config, expanded, policy_steps, counts = configuration(
        args, dataset, split, panel, mechanics_train, mechanics_validation,
        variant, temperature, initialization, learning_rate)
    validate_existing(output, config, getattr(args, 'resume', False))
    if completion.exists():
        if stop['requested']:
            raise SystemExit(130)
        return json.loads(completion.read_text())
    checkpoint_path = output / 'resume.pt'
    restored = None
    if checkpoint_path.exists():
        restored = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if restored.get('version') != RESUME_VERSION or restored.get('config') != config:
            raise ValueError(f'incompatible resume checkpoint: {checkpoint_path}')
        if 'optimizer_state_dict' not in restored:
            raise ValueError(f'optimizer state unavailable in {checkpoint_path}')
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = load_v2_model(initialization, args.device)
    if sum(p.numel() for p in model.parameters()) != REQUIRED_PARAMETERS:
        raise ValueError('this experiment requires the original 7.2M architecture')
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    word_tokens = torch.tensor([encode(word) for word in dataset.words], device=device)
    schedule = even_replay_schedule(counts)
    epoch, schedule_index, cursor, step = 0, 0, 0, 0
    sums, records = {}, []
    best_key, best_checkpoint, best_metadata = None, None, None
    stale, rapid_drift, pending_early, finished = 0, False, False, False
    last_gradient_norm = None
    mechanics_rng = torch.Generator()
    elapsed = 0.0
    if restored is not None:
        model.load_state_dict(restored['model_state_dict'])
        optimizer.load_state_dict(restored['optimizer_state_dict'])
        epoch, schedule_index = restored['epoch'], restored['schedule_index']
        cursor, step = restored['policy_cursor'], restored['step']
        sums, records = restored['sums'], restored['records']
        best_key = restored['best_key']
        best_checkpoint, best_metadata = restored['best_checkpoint'], restored['best_metadata']
        stale, rapid_drift = restored['stale'], restored['rapid_drift']
        pending_early, finished = restored['pending_early_evaluation'], restored['finished']
        last_gradient_norm = restored['last_gradient_norm']
        elapsed = restored['elapsed_seconds']
        mechanics_rng.set_state(restored['mechanics_rng_state'])
        torch.set_rng_state(restored['torch_rng_state'])
        if restored['cuda_rng_states'] is not None:
            torch.cuda.set_rng_state_all(restored['cuda_rng_states'])
    else:
        write_json(output / 'manifest.json', {**config, 'model_config': model.config})
    started = time.monotonic() - elapsed

    def publish_committed_artifacts():
        with atomic_file(output / 'metrics.jsonl', 'w') as stream:
            for record in records:
                stream.write(json.dumps(record) + '\n')
        if best_checkpoint is not None:
            save_torch(output / 'checkpoints' / 'best.pt', best_checkpoint)
            write_json(output / 'best.json', best_metadata)
        else:
            (output / 'checkpoints' / 'best.pt').unlink(missing_ok=True)
            (output / 'best.json').unlink(missing_ok=True)
        committed_reports = {record['report_file'] for record in records}
        for path in output.glob('evaluation-*.json'):
            if path.name not in committed_reports:
                path.unlink()
        # A crash can leave a terminal model written before its completion marker.
        if not finished:
            (output / 'checkpoints' / 'last.pt').unlink(missing_ok=True)

    if restored is not None:
        publish_committed_artifacts()
        del restored

    def model_payload(current_epoch):
        return {'model_state_dict': {key: value.detach().cpu().clone()
                                     for key, value in model.state_dict().items()},
                'model_config': model.config, 'vocabulary_size': 35,
                'soft_distillation': {**config, 'model_config': model.config,
                                      'epoch': current_epoch, 'step': step}}

    def save_resume():
        save_torch(checkpoint_path, {
            'version': RESUME_VERSION, 'config': config,
            'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_states': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
            'epoch': epoch, 'schedule_index': schedule_index, 'policy_cursor': cursor,
            'mechanics_rng_state': mechanics_rng.get_state(), 'sums': sums, 'step': step,
            'best_key': best_key, 'best_checkpoint': best_checkpoint, 'best_metadata': best_metadata,
            'stale': stale, 'rapid_drift': rapid_drift, 'records': records,
            'elapsed_seconds': time.monotonic() - started,
            'pending_early_evaluation': pending_early, 'finished': finished,
            'last_gradient_norm': last_gradient_norm})

    def stop_if_requested():
        if stop['requested'] or (stop_after is not None and step >= stop_after):
            save_resume()
            print(json.dumps({'event': 'stopped', 'run': name, 'step': step,
                              'resume_checkpoint': str(checkpoint_path)}), flush=True)
            raise SystemExit(130)

    def evaluate(current_epoch, training_metrics, *, label=None, patience_check=True):
        nonlocal best_key, best_checkpoint, best_metadata, stale, rapid_drift
        model.eval()
        report = evaluate_soft_model(model, dataset, panel, temperature, split['validation'],
                                     dataset.words, mechanics_validation, device,
                                     batch_size=args.eval_batch_states, distribution_examples=24)
        key = checkpoint_key(report)
        improved = best_key is None or key > best_key
        report_file = f'evaluation-{current_epoch if label is None else label}.json'
        record = {'epoch': current_epoch, 'step': step, 'improved': improved, 'report_file': report_file,
                  'elapsed_seconds': time.monotonic() - started, 'training': training_metrics,
                  'gameplay': {k: v for k, v in report['gameplay'].items() if k != 'results'},
                  'policy': report['policy'], 'action_quality': report['action_quality']['summary'],
                  'mechanics_validation_loss': report['mechanics_validation_loss']}
        if variant == 'C' and step == 250:
            rapid_drift = report['gameplay']['wins'] <= records[0]['gameplay']['wins'] - 5
            record['rapid_destructive_drift'] = rapid_drift
        records.append(record)
        write_json(output / report_file, report)
        if improved:
            best_key, stale = key, 0
            best_checkpoint = model_payload(current_epoch)
            best_metadata = {'key': key, **record}
        elif patience_check:
            stale += 1
        print(json.dumps({'event': 'evaluation', 'run': name, **record}), flush=True)

    # The checkpoint is the commit record. Reports are written first; derived best/metrics
    # files are published afterwards and repaired from that record on every resume.
    if not checkpoint_path.exists():
        save_resume()
    stop_if_requested()
    sampled, sampled_epoch = None, None
    while not finished:
        if pending_early:
            evaluate(epoch - 1 + cursor / policy_steps, {}, label=f'step-{step}', patience_check=False)
            pending_early = False
            save_resume()
            publish_committed_artifacts()
            stop_if_requested()
        if epoch == 0 or schedule_index == len(schedule):
            metrics = {}
            if epoch:
                metrics = {key: value / (counts['mechanics'] if key == 'mechanics_cross_entropy'
                                        else policy_steps * 128) for key, value in sums.items()}
                metrics['last_gradient_norm'] = last_gradient_norm
            evaluate(epoch, metrics)
            finished = epoch >= args.max_epochs or stale >= args.patience
            if not finished:
                epoch += 1
                schedule_index, cursor, sums = 0, 0, {}
                mechanics_rng.manual_seed(args.seed + epoch * 2_000_003)
            save_resume()
            publish_committed_artifacts()
            stop_if_requested()
            continue
        if sampled_epoch != epoch:
            rng = np.random.default_rng(args.seed + epoch * 1_000_003)
            sampled = expanded[rng.integers(len(expanded), size=(policy_steps, 128))]
            sampled_epoch = epoch
        model.train()
        optimizer.zero_grad(set_to_none=True)
        objective = schedule[schedule_index]
        if objective == 'expert':
            selected = sampled[cursor]
            cursor += 1
            # Group lengths inside the effective batch to maximize prefix sharing.
            selected = selected[np.argsort(dataset.lengths[selected], kind='stable')]
            for start in range(0, 128, args.microbatch_states):
                indices = selected[start:start + args.microbatch_states]
                prompts, lengths, tokens, ranks, teacher = policy_batch(
                    dataset, indices, word_tokens, device, temperature)
                sequence = candidate_sequence_logps(model, prompts, lengths, tokens)
                loss, diagnostics = distillation_loss(sequence, teacher, ranks)
                (loss * (len(indices) / 128)).backward()
                for key, value in diagnostics.items():
                    sums[key] = sums.get(key, 0.0) + float(value.detach()) * len(indices)
        else:
            selected = torch.randint(len(mechanics_train.inputs), (128,), generator=mechanics_rng)
            inputs = mechanics_train.inputs[selected].to(device)
            targets = mechanics_train.targets[selected].to(device)
            loss = F.cross_entropy(model(inputs).flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX)
            loss.backward()
            sums['mechanics_cross_entropy'] = sums.get('mechanics_cross_entropy', 0.0) + float(loss.detach())
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'), error_if_nonfinite=True)
        optimizer.step()
        last_gradient_norm = float(gradient_norm)
        step += 1
        schedule_index += 1
        pending_early = variant == 'C' and step in EARLY_C_EVALUATIONS
        if step % save_every == 0:
            save_resume()
        stop_if_requested()
        if step % 1000 == 0:
            print(json.dumps({'event': 'progress', 'run': name, 'epoch': epoch, 'step': step,
                              'elapsed_seconds': time.monotonic() - started}), flush=True)
    result = {'name': name, 'checkpoint': str(output / 'checkpoints' / 'best.pt'),
              'best': best_metadata, 'epochs': epoch,
              'rapid_destructive_drift': rapid_drift, 'elapsed_seconds': time.monotonic() - started}
    save_torch(output / 'checkpoints' / 'last.pt', model_payload(epoch))
    stop_if_requested()
    write_json(completion, result)
    del model, optimizer
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


def run(args):
    deterministic_execution()
    with stop_signals() as stop:
        return _run(args, stop)


def _run(args, stop):
    torch.set_num_threads(4)
    if min(args.microbatch_states, args.eval_batch_states, args.max_epochs, args.patience,
           getattr(args, 'save_every', 100)) < 1:
        raise ValueError('batch sizes and stopping settings must be positive')
    if getattr(args, 'stop_after_updates', None) is not None and args.stop_after_updates < 0:
        raise ValueError('stop_after_updates must be nonnegative')
    dataset = TeacherDataset(args.teacher_dir)
    mode = json.loads(args.mode.read_text())
    split = mode['runs'][args.fold - 1]
    if set(split['train']) & (set(split['validation']) | set(split['test'])):
        raise ValueError('source secret splits overlap')
    val_indices = split_indices(dataset, split['validation'])
    panel_rng = np.random.default_rng(args.seed + 700_001)
    panel = np.sort(panel_rng.choice(val_indices, min(args.panel_size, len(val_indices)), replace=False))
    mechanics_train = load_v2_split(args.mechanics_data, 'train', example_type='mechanics')
    mechanics_validation = load_v2_split(args.mechanics_data, 'validation', example_type='mechanics')
    # Validate every requested configuration before touching shared artifacts or starting any run.
    for variant in args.variants:
        initialization = args.mechanics_checkpoint if variant == 'B' else args.hard_checkpoint
        for temperature in args.temperatures:
            rates = [3e-4 if variant == 'B' else args.c_lr]
            if variant == 'C' and args.c_lr == 1e-5:
                rates.append(3e-6)
            for rate in rates:
                output = args.output_dir / configuration_name(variant, temperature, rate)
                config, _, _, _ = configuration(args, dataset, split, panel, mechanics_train,
                                                mechanics_validation, variant, temperature,
                                                initialization, rate)
                validate_existing(output, config, getattr(args, 'resume', False))
                if (output / 'resume.pt').exists() and not (output / 'training-complete.json').exists():
                    saved = torch.load(output / 'resume.pt', map_location='cpu', weights_only=True)
                    if saved.get('version') != RESUME_VERSION or saved.get('config') != config:
                        raise ValueError(f'incompatible resume checkpoint: {output / "resume.pt"}')
                    if 'optimizer_state_dict' not in saved:
                        raise ValueError(f'optimizer state unavailable in {output / "resume.pt"}')
                    del saved
    panel_path = args.output_dir / 'panel-indices.json'
    if panel_path.exists() and json.loads(panel_path.read_text()) != panel.tolist():
        raise ValueError('incompatible existing validation panel')
    baseline_hash = file_hash(args.hard_checkpoint)
    baseline_path = args.output_dir / 'hard-sft.json'
    baseline_identity = {
        'teacher_manifest_sha256': file_hash(Path(args.teacher_dir) / 'manifest.json'),
        'split': split, 'panel_indices': panel.tolist(),
        'mechanics_sha256': mechanics_identity(mechanics_validation),
        'eval_batch_states': args.eval_batch_states,
    }
    if baseline_path.exists():
        baseline = json.loads(baseline_path.read_text())
        if baseline.get('sha256') != baseline_hash or baseline.get('identity') != baseline_identity:
            raise ValueError('incompatible existing hard SFT baseline')
    if stop['requested']:
        raise SystemExit(130)
    write_json(panel_path, panel.tolist())
    if not baseline_path.exists():
        model = load_v2_model(args.hard_checkpoint, args.device)
        baselines = {}
        for temperature in TEMPERATURES:
            baselines[str(temperature)] = evaluate_soft_model(
                model, dataset, panel, temperature, split['validation'], dataset.words,
                mechanics_validation, torch.device(args.device), batch_size=args.eval_batch_states,
                distribution_examples=24)
            if stop['requested']:
                raise SystemExit(130)
        write_json(baseline_path, {'checkpoint': str(args.hard_checkpoint), 'sha256': baseline_hash,
                                   'identity': baseline_identity, 'evaluations': baselines})
        del model
    results = []
    for variant in args.variants:
        initialization = args.mechanics_checkpoint if variant == 'B' else args.hard_checkpoint
        for temperature in args.temperatures:
            if stop['requested']:
                raise SystemExit(130)
            results.append(train_configuration(args, dataset, split, panel, mechanics_train,
                mechanics_validation, variant, temperature, initialization,
                3e-4 if variant == 'B' else args.c_lr, _stop=stop))
            if variant == 'C' and args.c_lr == 1e-5 and results[-1]['rapid_destructive_drift']:
                results.append(train_configuration(args, dataset, split, panel, mechanics_train,
                    mechanics_validation, variant, temperature, initialization, 3e-6, _stop=stop))
    if stop['requested']:
        raise SystemExit(130)
    if file_hash(args.hard_checkpoint) != baseline_hash:
        raise RuntimeError('hard SFT baseline was modified')
    write_json(args.output_dir / 'sweep-complete.json', {'results': results, 'baseline_unchanged': True})
    return results


def parser():
    p = argparse.ArgumentParser(description='Soft classical-policy distillation without rollout training.')
    p.add_argument('--teacher-dir', type=Path, default=Path('data/soft-teacher-1m'))
    p.add_argument('--output-dir', type=Path, default=Path('runs/soft-distillation-dev'))
    p.add_argument('--mode', type=Path, default=Path('data/wordle-development.json'))
    p.add_argument('--fold', type=int, default=1)
    p.add_argument('--mechanics-data', type=Path, default=Path('data/wordle-dev-1m/fold-1/mechanics'))
    p.add_argument('--hard-checkpoint', type=Path, default=BASE_ROOT / 'checkpoints/best.pt')
    p.add_argument('--mechanics-checkpoint', type=Path, default=BASE_ROOT / 'mechanics/checkpoints/best.pt')
    p.add_argument('--variants', nargs='+', choices=('B', 'C'), default=['B', 'C'])
    p.add_argument('--temperatures', nargs='+', type=float, choices=TEMPERATURES, default=TEMPERATURES)
    p.add_argument('--c-lr', type=float, choices=(1e-5, 3e-6), default=1e-5)
    p.add_argument('--microbatch-states', type=int, default=128)
    p.add_argument('--eval-batch-states', type=int, default=16)
    p.add_argument('--panel-size', type=int, default=512)
    p.add_argument('--max-epochs', type=int, default=100)
    p.add_argument('--patience', type=int, default=4)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--resume', action='store_true', help='Explicitly continue compatible incomplete runs.')
    p.add_argument('--save-every', type=int, default=100, help='Save resume state every N optimizer updates.')
    p.add_argument('--stop-after-updates', type=int, default=None,
                   help='Save and exit 130 at this absolute optimizer step (per configuration).')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
