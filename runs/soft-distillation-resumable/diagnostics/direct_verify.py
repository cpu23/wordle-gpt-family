import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path.cwd()))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import numpy as np
import torch
from model import WordleGPT
from tokenizer_v2 import encode
from tokenizer import FEEDBACK_TO_SYMBOL
from wordle import score_guess

ROOT = Path('runs/soft-distillation-resumable/diagnostics')
reference = json.loads((ROOT / 'comparison.json').read_text())
words = tuple(json.loads(Path('data/soft-teacher-1m/manifest.json').read_text())['words'])
secrets = json.loads(Path('data/wordle-development.json').read_text())['runs'][0]['validation']
word_tokens = torch.tensor([encode(word) for word in words], device='cuda')
torch.set_num_threads(4)
torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def direct_scores(model, prefix):
    # Independent ordinary model.forward: no soft_policy imports or reused KV.
    inputs = torch.cat((torch.tensor(prefix, device='cuda')[None].expand(len(words), -1), word_tokens[:, :4]), dim=1)
    result = []
    for start in range(0, len(words), 128):
        logits = model(inputs[start:start + 128])[:, len(prefix) - 1:len(prefix) + 4]
        targets = word_tokens[start:start + 128]
        result.append(logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).sum(-1))
    return torch.cat(result).cpu().double().numpy()


verified = {}
for name, description in reference['models'].items():
    payload = torch.load(description['checkpoint'], map_location='cpu', weights_only=True)
    config = payload.get('model_config') or payload['best_checkpoint']['model_config']
    model = WordleGPT(**config).to('cuda').eval().requires_grad_(False)
    model.load_state_dict(payload['model_state_dict'])
    del payload
    states = json.loads((ROOT / f'{name}.json').read_text())['states']
    stored = np.load(ROOT / f'{name}-scores.npz')['logps']
    selected = np.linspace(0, len(states) - 1, 12, dtype=int)
    errors, disagreements = [], 0
    for row in selected:
        direct = direct_scores(model, encode(states[row]['prompt']))
        errors.append(float(np.max(np.abs(direct - stored[row]))))
        disagreements += int(direct.argmax() != stored[row].argmax())
        np.testing.assert_allclose(direct, stored[row], rtol=2e-5, atol=5e-5)
    cache, results = {}, []
    for secret in secrets:
        prefix = encode('<P><G>')
        guesses = []
        for _ in range(6):
            key = tuple(prefix)
            if key not in cache:
                cache[key] = words[int(direct_scores(model, prefix).argmax())]
            guess = cache[key]
            guesses.append(guess)
            if guess == secret:
                break
            feedback = ''.join(FEEDBACK_TO_SYMBOL[c] for c in score_guess(secret, guess))
            prefix += encode(guess + '<F>' + feedback + '<G>')
        results.append({'secret': secret, 'guesses': guesses, 'won': guesses[-1] == secret})
    expected = description['gameplay']['sequence_argmax']['results']
    differing = sum(a['guesses'] != b['guesses'] for a, b in zip(results, expected))
    wins = sum(row['won'] for row in results)
    assert wins == description['gameplay']['sequence_argmax']['wins']
    verified[name] = {'direct_forward_wins': wins, 'games': len(results),
                      'trajectory_disagreements': differing, 'panel_argmax_disagreements': disagreements,
                      'max_logp_error_against_prefix_reuse': max(errors),
                      'direct_full_dictionary_panel_scores_checked': len(selected) * len(words),
                      'results': results}
    print(json.dumps({k: v for k, v in verified[name].items() if k != 'results'}), flush=True)
    del model
    torch.cuda.empty_cache()
(ROOT / 'independent-verification.json').write_text(json.dumps(verified, indent=2) + '\n')
