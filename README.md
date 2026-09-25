# Wordle GPT

A compact decoder-only Transformer exploring whether a neural network can learn both the rules and an effective strategy for Wordle purely from game trajectories as text.

Rather than relying on an external game engine or search algorithm at runtime, Wordle GPT treats the game as a sequence-prediction problem: tracking board state, respecting feedback constraints, and generating informative guesses through autoregressive token generation.

---

## Core Questions

1. **Rule learning:** Can a small transformer learn valid 5-letter dictionary words and exact Wordle feedback mechanics without hardcoded rules?
2. **Strategy acquisition:** Can the model learn to narrow down possible candidate words and pick high-information guesses purely by imitating expert play?
3. **Catastrophic forgetting:** Does learning strategic play erase the model's understanding of basic game rules, and can multi-task experience replay prevent that regression?

---

## Model Architecture

The model is a standard pre-norm decoder-only Transformer with causal multi-head self-attention, GELU feed-forward blocks, and learned positional embeddings.

| Parameter | Base Model | Scaled Model |
| :--- | :--- | :--- |
| **Parameters** | ~815,000 | ~3,202,000 |
| **Embedding Size** | 128 | 256 |
| **Layers** | 4 | 4 |
| **Attention Heads** | 4 | 8 |
| **MLP Hidden Dim** | 512 | 1024 |
| **Context Length** | 96 tokens | 96 tokens |
| **Vocabulary** | 35 tokens | 35 tokens |

The 35-token vocabulary consists of:
- 26 lowercase English letters (`a`–`z`)
- 3 feedback digits: `0` (gray / miss), `1` (yellow / wrong position), `2` (green / exact hit)
- 6 structural control tokens: `<G>` (guess), `<F>` (feedback), `<E>` (end of game), `<M>` (mechanics task), `<S>` (secret word), `<P>` (policy task)

---

## Sequence Representation

Games are serialized as token sequences alternating between guesses and feedback:

```text
<G>could<F>22010<G>colon<F>22222<E>
```

### Multi-Task Objectives

- **Mechanics (`<M>`):** Given a secret word and a guess, predict the 5-digit feedback pattern:
  `<M><S>colon<G>could<F>22010<E>`
- **Policy (`<P>`):** Given the observed game history, predict the next strategic guess:
  `<P><G>could<F>22010<G>colon<E>`

Training with multi-task experience replay (blending mechanics examples during policy fine-tuning) prevents catastrophic forgetting of game rules while policy performance improves.

---

## Key Findings

- **Mechanics are learned rapidly:** The model achieves near-perfect prediction of Wordle feedback patterns within the first few training epochs.
- **Replay stabilizes strategy:** Fine-tuning exclusively on expert moves causes the model to forget game rules. A 5%–10% mechanics replay ratio maintains rule fidelity without degrading gameplay quality.
- **State scaling drives performance:** Scaling training from 10,000 to 1,000,000 unique game states improves held-out 5-fold cross-validation win rates from ~70% to **>98%**.
- **Constrained decoding:** Filtering output logits at generation time to valid 5-letter dictionary words eliminates rare spelling failures and yields consistent wins.

*For full daily training logs, loss curves, gradient norms, and ablation studies, see [EXPERIMENTS.md](EXPERIMENTS.md).*

---

## Quickstart

### Setup

Requires Python 3.11+. Install dependencies using `uv` or standard `pip`:

```bash
uv run --with-requirements requirements.txt python -m unittest discover
# or with standard pip:
pip install -r requirements.txt
python -m unittest discover
```

### Training

Train a model on nested state datasets:

```bash
uv run --with-requirements requirements.txt \
  python train_nested.py \
  --sizes 100000 \
  --steps 10000 \
  --checkpoints 0 100 500 1000 5000 10000
```

### Soft classical-policy distillation

```bash
uv run --with-requirements requirements.txt python soft_teacher.py \
  --output-dir data/soft-teacher-1m --workers 8 \
  --mode data/wordle-development.json
uv run --with-requirements requirements.txt python train_soft_distillation.py
uv run --with-requirements requirements.txt python compare_soft_distillation.py
uv run --with-requirements requirements.txt python benchmark_soft_distillation.py
```

Skip teacher construction when the verified dataset exists. Use `--stats-only`
with the same teacher command to recompute temperature statistics from permanent
scores, without rerunning Wordle simulations. The benchmark command checks the
development gate first and evaluates no test secrets if nothing qualifies.

The teacher builder processes all 1M observable states. Every state stores 128
unique legal actions: the best 32, 32 sampled from ranks 33–256, and 64 random
remaining actions, retaining all stored top-eight guesses and any remaining-answer
expert target. Candidate sampling is seeded by observable prompt bytes, never
the source secret. Raw expected-survivor costs, exhaustive ranks, answer-set
membership, state IDs, prompts, and source IDs are permanent memory-mapped arrays.
An additional answer-set cache preserves exact integer partition numerators.

For `N > 1`, teacher probabilities are
`softmax(-log(expected_survivors / best_expected_survivors) / T)`.
The sweep uses `T = 0.25, 0.50, 1.00`. Exact score ties receive equal probability;
the existing solver's remaining-answer-first, then dictionary-order tie rule
determines ranks and candidate selection only. With one answer left, its target
probability is one. There is no solve bonus or other auxiliary policy objective.

`soft_policy.py` computes raw-vocabulary five-letter sequence log probabilities,
then normalizes across the same 128 candidates. Differentiable shared history
K/V representations avoid processing every history 128 times. The complete
teacher distribution supplies cross-entropy; no history, feedback, or end-token
policy loss is added.

The unchanged original hard-SFT checkpoint is A. B starts from its original
mechanics-pretrained initialization at LR `3e-4`; C starts from hard SFT at
`1e-5`. Both retain the original architecture, AdamW settings, effective batch
of 128 states, corpus sampling weights, and 95% policy / 5% mechanics replay.
Defaults evaluate each full weighted epoch, with patience four and a maximum
of 100 epochs. C additionally evaluates steps 100, 250, and 1,000; losing at
least five constrained validation wins by step 250 triggers a fresh `3e-6` run
for that temperature.

Checkpoint ordering is constrained wins, attempts/game, guesses among wins,
exhaustive action regret, then teacher-to-student KL. Fixed 512-state validation
panels report regret, exhaustive rank fractions, relative cost, and complete
matched teacher/student distributions across candidate-count strata. Mechanics
validation is evaluated on its full existing split. Source-secret splits remain
strict; full-dictionary hypothetical answers match the original SFT teacher.
The development run never automatically evaluates fixed test secrets.

`comparison.json` contains the seven-row comparison, any lower-LR C retries,
and paired development games. Full five-fold/three-seed benchmarking is allowed
only for improved gameplay ordering, or at least 5% lower mean action regret
without worse gameplay. Its outputs include paired results for all 719 held-out
secrets per seed and matched action-regret distributions.

#### Stopping and resuming distillation

New runs save an atomic `resume.pt` inside each configuration directory every
100 optimizer updates, after evaluations, and on graceful stop. It includes
weights, Adam state, CPU/CUDA RNG state, replay position/generator, partial epoch
metrics, pending evaluations, and checkpoint-selection/early-stopping state.
`Ctrl+C` or `SIGTERM` finishes the current update or evaluation, saves, and exits
the entire sweep with status 130; it does not start the next model.

```bash
# Start a new resumable sweep without overwriting the cancelled legacy run:
uv run --with-requirements requirements.txt python train_soft_distillation.py \
  --output-dir runs/soft-distillation-resumable

# Later, repeat the same training arguments and add --resume:
uv run --with-requirements requirements.txt python train_soft_distillation.py \
  --output-dir runs/soft-distillation-resumable --resume
```

Use `--save-every N` to change the periodic save interval.
`--stop-after-updates N` is an absolute per-configuration update boundary for
controlled pauses; remove it or raise it when resuming. A hard kill/power loss
can only recover the last committed checkpoint. Resume restores its authoritative
metrics and best model, discarding evaluation reports newer than that checkpoint.
Training/data/split/panel compatibility is checked before existing run artifacts
are changed. Compatible completed configurations are skipped.

Deterministic PyTorch algorithms and a deterministic cuBLAS workspace are enabled
for reproducible continuation. CPU regression tests and a real 7.2M CUDA test
cover interruptions across policy updates, mechanics replay, and epoch boundaries.
The CUDA test reproduced all weights and Adam state bit-for-bit.

The already-cancelled legacy `runs/soft-distillation-dev/B-T0.25/` has no optimizer
checkpoint and only epoch-zero model weights. Its later evaluation is not a
recoverable model state. It is preserved and cannot be resumed; restart that
configuration in a new output directory. No experiment is automatically restarted
by adding resume support.

### One-guess expected-information GRPO

```bash
uv run --with-requirements requirements.txt python grpo_information_states.py
uv run --with-requirements requirements.txt python train_information_grpo.py
```

Skip corpus creation if `data/grpo-information/` already exists. The defaults
start from the original 7.2M SFT for 1,000 updates, with learning rate `3e-6`,
frozen-SFT KL beta `0.10`, and 16 states per update.

The corpus combines entropy, simple, partly-random, poor, and random histories
from the 1M dataset. Fresh SFT histories and current-policy histories refreshed
every 100 updates supply the model behavior. Nonempty states are sampled
uniformly by available candidate-count bucket (`1–2`, `3–5`, `6–20`, `21–100`,
`101+`), then available history depth (1–5), then available behavior. Coverage
reports unavailable cells rather than inventing states. Empty openings occupy
7.5% of groups.

Each state receives 64 distinct legal guesses: 48 stochastic policy proposals
and 16 random proposals. Duplicates are refilled with accepted words excluded.
For the full set of consistent dictionary answers, reward is exactly:

```text
expected_after = sum(feedback_bucket_size ** 2) / candidate_count
reward = log(candidate_count / expected_after)
         + 2 * (guess in candidates) / candidate_count
```

The 64 rewards are standardized within the state. Only the five new guess
letters receive policy/KL loss. Token-level PPO clipping and likelihoods use
the original dictionary mask, not the refill exclusion mask. Equal weighting
of policy/random candidates is a ranking surrogate, **not unbiased on-policy
PPO**. Cached prefix logits are local to one proposal batch.

Training-source secrets exclude validation/test secrets, but the candidate
universe and exact reward include **every consistent dictionary word**, including
validation/test answers. Held-out-source evaluation therefore measures
generalization to different histories/secrets, not completely unseen answer
supervision. Reports distinguish independent stochastic-policy reward from the
mixed-proposal reward and include greedy gameplay, oracle reward regret, and
difficulty/depth/behavior breakdowns. Artifacts are written under
`runs/grpo-information-dev/`.

### Token-level continuation GRPO and learning-rate comparison

Build a fixed corpus from the 1M development dataset, then train eight independent
continuations from each sampled state rather than always starting at an empty board:

```bash
uv run --with-requirements requirements.txt python grpo_corpus.py
uv run --with-requirements requirements.txt python train_grpo_games.py \
  --state-corpus data/grpo-continuations/train.jsonl \
  --validation-corpus data/grpo-continuations/validation.jsonl \
  --updates 1000 --kl-beta 0.10 --lr 1e-6 \
  --output-dir runs/grpo-token-lr-1e6
```

Repeat the training command with `--lr 3e-6` / `1e-5` and separate output
directories. Keep beta at `0.10` and everything else fixed. Every fresh run starts
from the original SFT. Skip corpus creation if the verified corpus already exists.

The builder scans all 1M records and reservoir-samples **100,000 unique training
states** plus **512 validation states**, using source-secret membership in the
development split. Histories have 1–5 guesses, truthful feedback, and no prior
solve. Empty boards, terminal histories, and test secrets are excluded.
The training corpus covers all 575 training secrets; validation covers all 72
validation secrets. SFT labels and empty-history sampling weights are not used.
The corpus manifest records source hashes, split exclusions, depths, and behaviors.

Each update samples 16 states uniformly with replacement. All eight members of
a group share its supplied history and hidden secret. Reward is
**`7 - total_guesses_used` if solved, otherwise `0`**, where total guesses includes
the supplied history. Only newly generated continuation letters contribute to
policy likelihood and prefix KL; the supplied history is context, not an action.
Each rollout receives one group-relative advantage. Ratios are **per token**:
`exp(new_token_logp - old_token_logp)`, clipped to `[0.8, 1.2]`.
The clipped surrogate is averaged over generated tokens **within each rollout**,
then averaged over rollouts. Raw-vocabulary KL uses the same reduction.
Thus neither the reward objective nor the anchor automatically weights a long
completion more heavily merely because it contains more tokens. Summed
trajectory log-probabilities are diagnostics only, never the PPO ratio.

The short comparison saves checkpoints at **0, 100, 250, 500, 1,000**.
Each run draws 16,000 groups / 128,000 continuations, not a full corpus epoch.
State order and RNG seeds are matched across learning rates; sampling outcomes
can diverge as policies change. Eight samples, temperature 1, and one optimizer
step per fresh batch are fixed. Initial on-policy ratios are approximately one;
clipping does not provide a trust-region guarantee for the ensuing Adam step.

Every checkpoint evaluates both a **fixed 512-state training panel** sampled from
the training corpus and the fixed 512-state held-out panel, with 4,096 continuations
per panel. `training_continuations` and `continuations` in each validation report
contain rewards, wins, total attempts, group variation, equal-rollout prefix KL,
and full history-aware traces. These fixed panels distinguish learning from
changes in the difficulty of freshly sampled training batches.

Reports retain ordinary empty-board raw/constrained gameplay and its existing
checkpoint ranking (`best.pt`). `best-continuation.pt` separately ranks held-out
continuation reward first, with the gameplay ranking as tie-breakers.
Training logs record source-state indices/depths, token-ratio extrema, clipping
fraction, advantage magnitude, and pre/post-update KL. At update 1 and every 100
updates, `length_diagnostics` records counts, mean absolute advantage, mean `1/T`
token weight, and actual reward-only/combined parameter-gradient L2 contributions
by generated guesses and by initially remaining guesses. Bucket losses retain
the global batch denominator; their gradient norms do not add because gradients
can cancel. Diagnostics preserve `.grad` and the subsequent training backward.
Representative training traces are saved at update 1 and scheduled checkpoints.
No test evaluation is performed.

Earlier experiments in `EXPERIMENTS.md` used the superseded whole-trajectory
ratio; their artifacts remain historical evidence, not results of this corrected
objective.

### Full-trajectory GRPO from the original 7.2M SFT

`train_grpo_games.py` trains on **complete games**, starting from the original
`runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt`, never the
250-update one-guess GRPO checkpoint or DPO:

```bash
uv run --with-requirements requirements.txt python train_grpo_games.py \
  --updates 10000 --output-dir runs/grpo-token-games-dev
```

Each update uniformly samples 16 training secrets with replacement. For each
secret, eight independent stochastic constrained rollouts start at the same
empty board. A game ends on a solve or after six guesses. Unlike the one-guess
experiment below, guesses and trajectories are **not forced to be distinct**;
repeated dictionary-legal guesses remain possible. Real Wordle feedback is fed
back into each trajectory, and the model never receives the secret.

The only trajectory reward is:

| Outcome | Solve in 1 | 2 | 3 | 4 | 5 | 6 | Failure |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reward | 6 | 5 | 4 | 3 | 2 | 1 | 0 |

The token-level clipped objective and equal-rollout normalization described
above apply to empty-board games too. All generated turns contribute; supplied
feedback, control tokens, and padding are excluded. Advantages are standardized
within each same-secret group; identical rewards yield zero reward advantage.
Every update uses fresh games.

Defaults: LR `1e-6`, KL coefficient `0.10`, frozen original SFT reference,
gradient clipping `1.0`, six turns, and eight games per group. Raw-vocabulary
categorical KL is averaged over generated-letter prefixes within each rollout,
then across rollouts; constrained KL is also logged. These are not full-game KLs.

Every checkpoint at **0, 100, 250, 500, 1,000, 2,500, 5,000, 10,000** is saved,
plus the final update for shorter runs. `--eval-every` optionally adds checkpoints.
Use a new output directory per run. To continue a corrected-objective checkpoint,
pass `--resume-checkpoint <path>` with unchanged training settings and a larger
`--updates` total. Model, Adam, state-sampling RNG, and CPU/CUDA sampling RNGs are
restored; the reference remains the original SFT. Old trajectory-ratio checkpoints
cannot resume under the new objective.

`metrics.jsonl` tracks sampled training-game win fraction, mean guesses among
winners (null if none), mean trajectory reward, within-group reward std,
identical-reward group rate, and pre/post-update KL. At every checkpoint,
`validation-<update>.json` includes:

- Greedy raw/constrained validation wins, win rate, average attempts, and
  mean terminal trajectory reward.
- Eight stochastic full games for each validation secret, with the same
  outcome/group diagnostics and individual rollout traces.
- Current-policy-prefix KL and fixed SFT-prefix KL for cross-checkpoint comparison.
- Classical-solver action regret, retaining the prior checkpoint ranking.

Validation sampling resets an isolated RNG; evaluation frequency does not change
the training stream. `reference-rollouts.json` records the frozen SFT probe.
All checkpoints are ranked by constrained wins, lower average attempts, lower
mean solver regret, and finally lower fixed-prefix KL; `best.pt` follows that
ranking. No test secrets are evaluated. This experiment has no candidate-reduction
reward or intermediate solve bonus.

### One-step GRPO from the 7.2M SFT model

`train_grpo.py` defaults to the original SFT checkpoint at
`runs/scaling-dev-1m/seed-0/fold-1/7.2m/checkpoints/best.pt`, not a DPO model.
One-step refers to the reward horizon: each action is one guess, not a game rollout.
The default run performs **10,000 optimizer updates** with dense early checkpoints:

```bash
uv run --with-requirements requirements.txt python train_grpo.py \
  --updates 10000 --output-dir runs/grpo-dev-dense
```

Each update samples 16 fresh reachable states from training secrets, then eight
distinct dictionary-legal guesses per state using autoregressive token masking.
Actions are not restricted to remaining answers. Previously drawn group actions
are excluded; the same masks are used when computing update likelihoods.
Rewards use actual hidden-secret feedback:
`ln(candidate_count_before / candidate_count_after) + 5 * solved`.
Advantages are standardized within each group; identical-reward groups have
exactly zero reward advantage. There is one gradient step per fresh batch,
without replaying actions over multiple optimizer epochs.

Defaults: LR `1e-6`, frozen original-SFT reference, KL coefficient `0.02`,
gradient clipping `1.0`. The KL regularizer is exact raw-vocabulary categorical
KL at sampled action prefixes, also anchoring probability outside the dictionary;
constrained-prefix KL is reported separately. Neither metric is full-game KL.

Every scheduled checkpoint is saved: **0, 100, 250, 500, 1,000, 2,500, 5,000,
10,000** updates. The final update is also saved for shorter runs. Set
`--eval-every N` for extra evaluations; the default `0` uses only the schedule.
Use `--updates 2 --output-dir runs/grpo-smoke` for a short check.
Each run starts from SFT; checkpoint resume is not implemented. Use a new output
directory for each run. There is no automatic early stopping.

Artifacts include `metrics.jsonl`, `validation-<update>.json`, a provenance
manifest, `diagnostic-states.json`, and `checkpoints/update-<update>.pt`.
Validation tracks raw/constrained wins, attempts, invalid/repeated guesses,
actual reduction/reward, and expected-survivor regret against the classical solver.

At **every checkpoint, including update 0**, a fixed panel of 128 reachable
validation-secret states (`--diagnostic-states`) is sampled for eight actions each.
Its RNG is reset consistently and isolated from training, so changing evaluation
frequency does not change the training stream. Checkpoint diagnostics include:

- Mean total reward and mean log candidate-reduction reward, excluding solve bonus.
- Fraction of sampled guesses solving immediately.
- Mean within-group reward standard deviation and identical-reward group rate.
- Raw/constrained SFT KL on current sampled prefixes.
- Token-policy entropy, summed conditional action entropy, and first-draw action entropy.
- Unique action counts/fractions and empirical word-frequency entropy across the
  panel, including first-draw-only measures unaffected by forced group uniqueness.

Entropy values are in nats. Summed token entropy estimates action entropy on
sampled prefixes; it is not an exhaustive calculation over every dictionary word.
Diversity is measured across the panel, not by the trivially fixed eight unique
actions per group. With eight distinct guesses, the immediate-solve fraction is
at most 1/8; this is different from whole-game win rate.

`checkpoint-ranking.json` ranks all saved evaluations, lexicographically:
**more constrained validation wins → lower average attempts → lower mean
expected-survivor action regret → lower KL**. The final KL tie-break uses the same
frozen SFT-sampled prefixes at every checkpoint, making it independent of action
sampling drift. `best.pt` follows this ranking; a wins/attempts tie can be broken
by lower regret. Exact four-metric ties retain the earlier checkpoint, but no
scheduled checkpoint is discarded. The held-out test split is not evaluated.

### Evaluation

Run the 5-fold cross-validation benchmark across multiple seeds:

```bash
uv run --with-requirements requirements.txt python benchmark_cv.py --skip-prepare
```

---

## Repository Structure

- `model.py` — Transformer architecture and configuration.
- `wordle.py` — Wordle environment logic, feedback generation, and solver baselines.
- `tokenizer_v2.py` — Tokenizer mappings and sequence formatting.
- `dataset_v2.py` / `dataset_expert.py` — Trajectory generation and training dataset builders.
- `train_v2.py` / `train_nested.py` — Multi-task training and experience replay loaders.
- `benchmark_cv.py` / `cross_validation.py` — Held-out evaluation and benchmark suites.
- `EXPERIMENTS.md` — Complete archive of training runs, ablations, and experiment metrics.
