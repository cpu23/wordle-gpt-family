import copy
import io
import json
import os
import signal
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

import train_soft_distillation as trainer
from evaluate_v2 import load_v2_model
from model import WordleGPT
from soft_teacher import TeacherDataset


class SoftResumeTests(unittest.TestCase):
    """Exercise the real policy objective, Transformer gradients, and AdamW state."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        threads = torch.get_num_threads()
        self.addCleanup(torch.set_num_threads, threads)
        torch.set_num_threads(1)
        rng = torch.get_rng_state()
        self.addCleanup(torch.set_rng_state, rng)
        torch.manual_seed(71)
        model = WordleGPT(vocab_size=35, context_length=8, embedding_size=8,
                          num_layers=1, num_heads=2, mlp_size=16)
        self.parameter_count = sum(parameter.numel() for parameter in model.parameters())
        self.initialization = self.root / 'initial.pt'
        torch.save({'model_state_dict': model.state_dict(), 'model_config': model.config,
                    'vocabulary_size': 35}, self.initialization)
        self.initial_weights = copy.deepcopy(model.state_dict())
        self.teacher_dir = self.root / 'teacher'
        self.teacher_dir.mkdir()
        words = ['crane', 'slate', 'trace', 'stare']
        arrays = {
            'prompts': np.array([[32, 29, 0], [32, 2, 29], [32, 29, 0], [32, 18, 29],
                                 [32, 29, 0], [32, 29, 0]], dtype=np.int64),
            'lengths': np.array([2, 3, 2, 3, 2, 2]),
            'candidate_ids': np.tile(np.array([0, 1, 2]), (6, 1)),
            'costs': np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1], [1, 3, 2],
                               [1, 2, 3], [3, 1, 2]], dtype=np.float32),
            'ranks': np.array([[1, 2, 3], [2, 1, 3], [3, 2, 1], [1, 3, 2],
                               [1, 2, 3], [3, 1, 2]]),
            'remaining_counts': np.full(6, 3),
            'is_remaining': np.ones((6, 3), dtype=np.bool_),
            'source_ids': np.array([0, 0, 1, 1, 2, 3]),
            'state_ids': np.arange(6),
            'weights': np.array([1, 3, 2, 1, 1, 1]),
        }
        (self.teacher_dir / 'manifest.json').write_text(json.dumps({'words': words, 'rows': 6}))
        for name, array in arrays.items():
            np.save(self.teacher_dir / f'{name}.npy', array)
        self.dataset = TeacherDataset(self.teacher_dir)
        self.split = {'train': words[:2], 'validation': words[2:3], 'test': words[3:]}
        self.panel = np.array([4], dtype=np.int64)
        self.mode = self.root / 'mode.json'
        self.mode.write_text(json.dumps({'runs': [self.split]}))
        self.mechanics = SimpleNamespace(
            inputs=torch.tensor([[32, 29, 2, 17], [32, 29, 18, 11], [32, 29, 19, 17],
                                 [32, 29, 18, 19], [32, 29, 0, 1]]),
            targets=torch.tensor([[29, 2, 17, 0], [29, 18, 11, 0], [29, 19, 17, 0],
                                  [29, 18, 19, 0], [29, 0, 1, 2]]),
        )
        self.trace = []
        self.signal_to_send = None
        self.saved_handlers = {}
        self.real_policy_batch = trainer.policy_batch
        for name, value in (
            ('REQUIRED_PARAMETERS', self.parameter_count),
            ('EARLY_C_EVALUATIONS', (2, 4)),
            ('load_v2_model', self.load_model),
            ('evaluate_soft_model', self.evaluate),
            ('load_v2_split', lambda *args, **kwargs: self.mechanics),
            ('policy_batch', self.policy_batch),
        ):
            patcher = patch.object(trainer, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def arguments(self, directory, **overrides):
        args = trainer.parser().parse_args([])
        values = dict(teacher_dir=self.teacher_dir, output_dir=self.root / directory,
                      mode=self.mode, fold=1, mechanics_data=self.root / 'mechanics',
                      hard_checkpoint=self.initialization, mechanics_checkpoint=self.initialization,
                      variants=['C'], temperatures=[0.5], c_lr=1e-5, microbatch_states=64,
                      eval_batch_states=2, panel_size=1, max_epochs=2, patience=10,
                      seed=19, device='cpu', smoke_steps=3, resume=False,
                      save_every=2, stop_after_updates=None)
        values.update(overrides)
        vars(args).update(values)
        return args

    def load_model(self, checkpoint, device):
        model = load_v2_model(checkpoint, device)

        def observe_mechanics(module, inputs):
            if module.training and len(inputs[0]) == 128:
                self.trace.append(('mechanics', inputs[0].detach().cpu().numpy().tobytes()))
                if self.signal_to_send is not None:
                    signum, self.signal_to_send = self.signal_to_send, None
                    # Never deliver to a default or unrelated process handler if installation regresses.
                    self.assertNotEqual(signal.getsignal(signum), self.saved_handlers[signum])
                    os.kill(os.getpid(), signum)

        model.register_forward_pre_hook(observe_mechanics)
        return model

    def policy_batch(self, dataset, indices, word_tokens, device, temperature):
        self.trace.append(('expert', tuple(int(index) for index in indices)))
        return self.real_policy_batch(dataset, indices, word_tokens, device, temperature)

    def evaluate(self, model, *args, **kwargs):
        # A deterministic, genuine held-out loss replaces expensive full-dictionary gameplay.
        # The RNG witness also exposes a missing global-RNG restoration on resume.
        with torch.no_grad():
            loss = float(F.cross_entropy(model(self.mechanics.inputs).flatten(0, 1),
                                         self.mechanics.targets.flatten()))
            witness = float(torch.rand(()))
        return {'gameplay': {'wins': 1, 'average_attempts': loss, 'average_guesses': loss,
                             'results': []},
                'policy': {'teacher_student_kl': loss, 'rng_witness': witness},
                'action_quality': {'summary': {'mean_action_regret': loss}},
                'mechanics_validation_loss': loss}

    def train(self, args, *, split=None, panel=None):
        with redirect_stdout(io.StringIO()):
            return trainer.train_configuration(
                args, self.dataset, self.split if split is None else split,
                self.panel if panel is None else panel, self.mechanics, self.mechanics,
                'C', 0.5, self.initialization, args.c_lr)

    def stop(self, args):
        with self.assertRaises(SystemExit) as stopped:
            self.train(args)
        self.assertEqual(stopped.exception.code, 130)
        self.assertFalse((args.output_dir / 'C-T0.50' / 'training-complete.json').exists())
        return self.checkpoint(args)

    def checkpoint(self, args):
        return torch.load(args.output_dir / 'C-T0.50' / 'resume.pt',
                          map_location='cpu', weights_only=False)

    def artifact_bytes(self, directory):
        return {str(path.relative_to(directory)): path.read_bytes()
                for path in directory.rglob('*') if path.is_file()}

    def assert_state_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            self.assertTrue(torch.equal(left, right), 'tensor state changed across resume')
        elif isinstance(left, dict):
            self.assertEqual(set(left), set(right))
            for key in left:
                if key != 'elapsed_seconds':
                    with self.subTest(state_key=key):
                        self.assert_state_equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            self.assertEqual(len(left), len(right))
            for first, second in zip(left, right):
                self.assert_state_equal(first, second)
        else:
            self.assertEqual(left, right)

    def assert_training_equal(self, expected, actual):
        for key in ('model_state_dict', 'optimizer_state_dict', 'torch_rng_state',
                    'cuda_rng_states', 'epoch', 'schedule_index', 'policy_cursor',
                    'mechanics_rng_state', 'sums', 'step', 'best_key', 'best_checkpoint',
                    'best_metadata', 'stale', 'rapid_drift', 'records',
                    'pending_early_evaluation', 'finished'):
            with self.subTest(checkpoint_key=key):
                self.assert_state_equal(expected[key], actual[key])

    def test_resume_matches_real_updates_across_pending_evaluation_replay_and_epoch(self):
        uninterrupted = self.arguments('uninterrupted')
        self.train(uninterrupted)
        expected = self.checkpoint(uninterrupted)
        expected_trace, self.trace = self.trace, []
        resumed = self.arguments('resumed', stop_after_updates=2)
        state = self.stop(resumed)
        self.assertEqual(state['step'], 2)
        self.assertTrue(state['pending_early_evaluation'])
        self.assertEqual([record['step'] for record in state['records']], [0])
        for stop_at in (3, 4, 5):
            resumed.resume = True
            resumed.stop_after_updates = stop_at
            state = self.stop(resumed)
            self.assertEqual(state['step'], stop_at)
        resumed.stop_after_updates = None
        self.train(resumed)
        actual = self.checkpoint(resumed)
        self.assert_training_equal(expected, actual)
        self.assertEqual(expected_trace, self.trace)
        self.assertEqual(sum(kind == 'mechanics' for kind, _ in self.trace), 2)
        self.assertEqual([(record['step'], record['report_file']) for record in actual['records']],
                         [(0, 'evaluation-0.json'), (2, 'evaluation-step-2.json'),
                          (4, 'evaluation-step-4.json'), (4, 'evaluation-1.json'),
                          (8, 'evaluation-2.json')])
        self.assertTrue(any(not torch.equal(value, self.initial_weights[name])
                            for name, value in actual['model_state_dict'].items()))
        self.assertTrue(all(int(value['step']) == 8
                            for value in actual['optimizer_state_dict']['state'].values()))

    def test_old_committed_checkpoint_rolls_back_future_best_and_metrics(self):
        reference = self.arguments('reference')
        self.train(reference)
        expected = self.checkpoint(reference)
        args = self.arguments('rollback', stop_after_updates=2)
        self.stop(args)
        output = args.output_dir / 'C-T0.50'
        committed = (output / 'resume.pt').read_bytes()
        args.resume = True
        args.stop_after_updates = 5
        self.stop(args)
        future_report = output / 'evaluation-uncommitted.json'
        future_report.write_text(json.dumps({'gameplay': {'wins': 999}}))
        with (output / 'metrics.jsonl').open('a') as stream:
            stream.write(json.dumps({'step': 999, 'report_file': future_report.name}) + '\n')
        (output / 'best.json').write_text(json.dumps({'key': [999], 'step': 999}))
        torch.save({'model_state_dict': {name: torch.full_like(value, 999)
                                        for name, value in self.initial_weights.items()}},
                   output / 'checkpoints' / 'best.pt')
        # Emulate a crash after publishing future evaluation artifacts but before checkpoint commit.
        (output / 'resume.pt').write_bytes(committed)
        args.stop_after_updates = None
        self.train(args)
        actual = self.checkpoint(args)
        self.assert_training_equal(expected, actual)
        self.assertFalse(future_report.exists())
        records = [json.loads(line) for line in (output / 'metrics.jsonl').read_text().splitlines()]
        self.assert_state_equal(expected['records'], records)
        expected_output = reference.output_dir / 'C-T0.50'
        self.assert_state_equal(json.loads((expected_output / 'best.json').read_text()),
                                json.loads((output / 'best.json').read_text()))
        self.assert_state_equal(torch.load(expected_output / 'checkpoints' / 'best.pt', weights_only=True),
                                torch.load(output / 'checkpoints' / 'best.pt', weights_only=True))

    def test_incompatible_resume_refuses_before_any_artifact_mutation(self):
        args = self.arguments('incompatible', stop_after_updates=2)
        self.stop(args)
        args.resume = True
        args.stop_after_updates = None
        before = self.artifact_bytes(args.output_dir)
        teacher_manifest = self.teacher_dir / 'manifest.json'
        original_teacher = teacher_manifest.read_bytes()
        original_initialization = self.initialization.read_bytes()
        changed_split = dict(self.split, train=list(reversed(self.split['train'])))
        cases = ('microbatch', 'teacher', 'split', 'panel', 'initialization', 'mechanics')
        for change in cases:
            with self.subTest(change=change):
                candidate = copy.copy(args)
                split, panel = self.split, self.panel
                original_inputs = self.mechanics.inputs.clone()
                try:
                    if change == 'microbatch':
                        candidate.microbatch_states = 32
                    elif change == 'teacher':
                        teacher_manifest.write_text(original_teacher.decode() + ' ')
                    elif change == 'split':
                        split = changed_split
                    elif change == 'panel':
                        panel = np.array([5], dtype=np.int64)
                    elif change == 'initialization':
                        payload = torch.load(self.initialization, weights_only=True)
                        payload['model_state_dict']['output.bias'].add_(1)
                        torch.save(payload, self.initialization)
                    elif change == 'mechanics':
                        self.mechanics.inputs[0, 0] = 1
                    with self.assertRaises(ValueError):
                        self.train(candidate, split=split, panel=panel)
                    self.assertEqual(before, self.artifact_bytes(args.output_dir))
                finally:
                    teacher_manifest.write_bytes(original_teacher)
                    self.initialization.write_bytes(original_initialization)
                    self.mechanics.inputs.copy_(original_inputs)

    def test_sweep_provenance_preflight_does_not_publish_shared_artifacts(self):
        args = self.arguments('preflight', stop_after_updates=2)
        self.stop(args)
        args.resume = True
        args.stop_after_updates = None
        args.variants = ['B', 'C']
        before = self.artifact_bytes(args.output_dir)
        manifest = self.teacher_dir / 'manifest.json'
        manifest.write_text(manifest.read_text() + ' ')
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            trainer.run(args)
        # B is new, but the incompatible later C run must be checked before any
        # baseline, panel, or earlier configuration can be published.
        self.assertEqual(before, self.artifact_bytes(args.output_dir))

    def test_incomplete_run_requires_explicit_resume(self):
        args = self.arguments('explicit', stop_after_updates=2)
        self.stop(args)
        before = self.artifact_bytes(args.output_dir)
        args.stop_after_updates = None
        with self.assertRaisesRegex(ValueError, 'resume'):
            self.train(args)
        self.assertEqual(before, self.artifact_bytes(args.output_dir))

    def test_legacy_best_checkpoint_without_optimizer_is_not_resumable(self):
        args = self.arguments('legacy', resume=True)
        output = args.output_dir / 'C-T0.50'
        (output / 'checkpoints').mkdir(parents=True)
        (output / 'manifest.json').write_text(json.dumps({'variant': 'C', 'temperature': 0.5}))
        (output / 'checkpoints' / 'best.pt').write_bytes(self.initialization.read_bytes())
        (output / 'best.json').write_text(json.dumps({'epoch': 0, 'step': 0}))
        before = self.artifact_bytes(args.output_dir)
        with self.assertRaisesRegex(ValueError, '(?i)optimizer.*(unavailable|missing|absent|not available)'):
            self.train(args)
        self.assertEqual(before, self.artifact_bytes(args.output_dir))

    def test_completed_run_skips_only_compatible_configuration(self):
        args = self.arguments('completed')
        self.train(args)
        before = self.artifact_bytes(args.output_dir)
        self.train(args)
        self.assertEqual(before, self.artifact_bytes(args.output_dir))
        args.microbatch_states = 32
        with self.assertRaises(ValueError):
            self.train(args)
        self.assertEqual(before, self.artifact_bytes(args.output_dir))

    def test_stop_exits_whole_sweep_without_comparison_or_next_configuration(self):
        args = self.arguments('sweep-stop', variants=['C', 'B'], temperatures=[0.5, 1.0],
                              stop_after_updates=2)
        with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            trainer.run(args)
        self.assertEqual(stopped.exception.code, 130)
        self.assertEqual(self.checkpoint(args)['step'], 2)
        self.assertFalse((args.output_dir / 'C-T1.00').exists())
        self.assertFalse((args.output_dir / 'B-T0.50').exists())
        self.assertFalse((args.output_dir / 'sweep-complete.json').exists())

    def test_sigint_and_sigterm_finish_current_update_and_stop_entire_sweep(self):
        for signum in (signal.SIGINT, signal.SIGTERM):
            with self.subTest(signal=signum):
                args = self.arguments(f'signal-{signum}', variants=['C', 'B'])
                self.saved_handlers = {value: signal.getsignal(value)
                                       for value in (signal.SIGINT, signal.SIGTERM)}
                self.signal_to_send = signum
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as stopped:
                    trainer.run(args)
                self.assertEqual(stopped.exception.code, 130)
                state = self.checkpoint(args)
                self.assertEqual(state['step'], 3)
                self.assertTrue(all(int(value['step']) == 3
                                    for value in state['optimizer_state_dict']['state'].values()))
                self.assertFalse((args.output_dir / 'B-T0.50').exists())
                self.assertFalse((args.output_dir / 'sweep-complete.json').exists())
                for value, handler in self.saved_handlers.items():
                    self.assertEqual(signal.getsignal(value), handler)


if __name__ == '__main__':
    unittest.main()
