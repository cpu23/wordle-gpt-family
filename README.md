# Wordle GPT Family

A small decoder-only Transformer trained on serialized Wordle trajectories. The project includes deterministic trajectory generation, nested state datasets, tokenization, training, autoregressive generation, and loss analysis by token type and trajectory strategy.

## Model

`WordleGPT` uses:

| Parameter | Value |
|---|---:|
| Vocabulary size | 32 |
| Context length | 96 |
| Embedding size | 128 |
| Decoder blocks | 4 |
| Attention heads | 4 |
| MLP size | 512 |

Each pre-norm decoder block contains causal multi-head self-attention and a GELU MLP. The final layer normalization feeds a 32-token output projection.

## Sequence format

A trajectory is serialized as alternating guess and feedback fields followed by an end token:

```text
<G>could<F>22010<G>colon<F>22222<E>
```

The vocabulary contains 26 letters, feedback symbols `0`, `1`, and `2`, and control tokens `<G>`, `<F>`, and `<E>`.

## Dataset v2

Build the full v2 artifact with:

```bash
uv run --with-requirements requirements.txt python dataset_v2.py
```

The builder emits `data/wordle-v2/examples.jsonl.gz` and a manifest. Every one of the 100,000 broad source states contributes two examples:

1. **Mechanics:** `<M><S>colon<G>could<F>22010<E>`. `<M>` selects the mechanics task and `<S>` marks the supplied secret. Only the five feedback targets contribute to loss.
2. **Expert:** `<P><G>could<F>22010<G>clever-next-guess<E>`. `<P>` selects policy/action prediction. Only the five expert guess letters contribute to loss.

The source states are balanced across the `clever`, `simple`, `random`, and `partly-random` trajectory strategies. All expert labels are recomputed with `choose_informative_guess`, regardless of the source strategy.

Each record carries half-open `loss_ranges` over shifted next-token targets. `train_v2.load_v2_split` converts every target outside those ranges to `IGNORE_INDEX`. Consequently, training does not reward reconstruction of supplied secrets, historical or random guesses, unobservable environment feedback, structural markers, or end tokens.

The full artifact contains 200,000 examples: 100,000 mechanics and 100,000 expert examples. Its training split has 160,272 examples and exactly 801,360 supervised targets.

V2 appends `<M>`, `<S>`, and `<P>` to the unchanged 32-token v1 vocabulary, producing a 35-token vocabulary. V1 checkpoints remain 32-token models; later transfer from v1 must explicitly copy the shared token rows into a 35-token v2 model.

## Setup

```bash
uv run --with-requirements requirements.txt python -m unittest discover
```

The commands below use CUDA automatically when available. Pass `--device cpu` to force CPU execution.

## Training commands

Overfit a fixed batch of 32 trajectories and greedily continue one memorized example:

```bash
uv run --with-requirements requirements.txt \
  python train.py \
  --steps 1000 \
  --generate-example \
  --example-index 0 \
  --prefix-length 6
```

Train fresh, identically seeded models on every declared nested state prefix:

```bash
uv run --with-requirements requirements.txt \
  python train_nested.py \
  --steps 1000 \
  --checkpoints 0 100 500 1000
```

Run the full 100K-state experiment for 10,000 optimizer steps:

```bash
uv run --with-requirements requirements.txt \
  python train_nested.py \
  --sizes 100000 \
  --steps 10000 \
  --checkpoints 0 100 500 1000 2000 5000 10000
```

Train the full v1 dataset to a token-equivalent epoch target, recording validation metrics and a resumable checkpoint after every epoch:

```bash
uv run --with-requirements requirements.txt \
  python train_epochs.py \
  --target-epochs 20 \
  --save-epochs 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20
```

Run metadata, `metrics.jsonl`, and `latest.json` are written under `runs/v1-full-20epochs/`. Model and AdamW state, optimizer step, tokens seen, training-token count, and deterministic batch-generator state are saved under its `checkpoints/` directory. Resume with `--resume PATH_TO_CHECKPOINT`; the epoch target remains a total target rather than an additional number of epochs.

## Metric definitions

- **Optimizer step:** one AdamW update using a batch of 32 state trajectories.
- **Tokens seen:** non-padding next-token targets processed by optimizer updates.
- **Epochs:** tokens seen divided by the number of non-padding training targets. This token-equivalent definition accounts for variable-length trajectories.
- **Train/validation loss:** exact token-weighted cross-entropy over the complete split at each checkpoint.
- **Guess-letter loss:** letter targets within guesses.
- **Feedback loss:** feedback targets `0`, `1`, and `2`.
- **Structure loss:** `<G>` and `<F>` targets.
- **End loss:** `<E>` targets.
- **Trajectory-strategy loss:** all validation targets grouped by the trajectory's `clever`, `simple`, `random`, or `partly-random` strategy.
- **Next-guess strategy loss:** the five current-action letter targets grouped by trajectory strategy.

The data also records per-action policy. Individual partly-random actions are labeled either `clever` or `random`; therefore the four-way breakdown uses trajectory strategy so that `partly-random` remains a distinct group.

## Experiment log

### 2026-08-14 — Memorizing 32 trajectories

After 1,000 steps, greedy autoregressive decoding from the unique six-token prefix `<G>could` reproduced the remaining 19 tokens exactly:

```text
expected:  <G>could<F>22010<G>colon<F>22222<E>
predicted: <G>could<F>22010<G>colon<F>22222<E>
continuation tokens: 19/19 (100.0%)
```

### 2026-08-14 — Nested datasets at 1,000 steps

| States | Train examples | Validation examples | Train loss | Validation loss |
|---:|---:|---:|---:|---:|
| 1,000 | 814 | 117 | 0.2355 | 2.3186 |
| 10,000 | 8,060 | 968 | 0.8674 | 1.0060 |
| 30,000 | 24,191 | 3,013 | 0.9391 | 0.9622 |
| 100,000 | 80,136 | 10,120 | 0.9518 | 0.9580 |

The 1K model strongly overfits. The 30K and 100K models retain closely matched training and validation losses.

### 2026-08-14 — Full dataset at 10,000 steps

The full training split contains 2,536,680 next-token targets.

| Step | Tokens seen | Epochs | Learning rate | Train loss | Validation loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.000 | 3.00e-4 | 3.5303 | 3.5282 |
| 100 | 100,920 | 0.040 | 3.00e-4 | 1.3109 | 1.3083 |
| 500 | 504,528 | 0.199 | 3.00e-4 | 1.1019 | 1.1039 |
| 1,000 | 1,014,408 | 0.400 | 3.00e-4 | 0.9518 | 0.9580 |
| 2,000 | 2,028,348 | 0.800 | 3.00e-4 | 0.8317 | 0.8397 |
| 5,000 | 5,065,164 | 1.997 | 3.00e-4 | 0.7668 | 0.7910 |
| 10,000 | 10,123,728 | 3.991 | 3.00e-4 | 0.7019 | 0.7724 |

Validation loss by token type:

| Step | Guess letters | Feedback | Structure | End |
|---:|---:|---:|---:|---:|
| 0 | 3.5594 | 3.5455 | 3.4282 | 3.3176 |
| 100 | 2.1281 | 0.8719 | 0.1972 | 0.9961 |
| 500 | 1.7247 | 0.8093 | 0.1526 | 0.8635 |
| 1,000 | 1.4165 | 0.7697 | 0.1265 | 0.9449 |
| 2,000 | 1.1686 | 0.7355 | 0.1522 | 0.8110 |
| 5,000 | 1.0860 | 0.7023 | 0.1459 | 0.8247 |
| 10,000 | 1.0760 | 0.6675 | 0.1299 | 0.8928 |

At 10,000 steps, validation loss by trajectory strategy is:

| Clever | Simple | Random | Partly random |
|---:|---:|---:|---:|
| 0.3699 | 0.9132 | 0.9484 | 0.7570 |

Current next-guess letter loss by trajectory strategy is:

| Clever | Simple | Random | Partly random |
|---:|---:|---:|---:|
| 0.2278 | 1.3417 | 1.4643 | 1.0549 |

Structural syntax is easiest. Clever next guesses are highly learnable, while random next guesses remain the hardest. Feedback is easier than guess letters but substantially harder than structural control tokens.

### 2026-08-14 — Full dataset through 20 epochs

The model completed 50,080 optimizer updates and processed 50,733,816 non-padding targets, or 20.0001 token-equivalent epochs. A resumable checkpoint and exact full-split validation metrics were saved at epoch zero and every integer epoch through 20.

| Epoch | Step | Train loss | Validation loss |
|---:|---:|---:|---:|
| 0 | 0 | 3.5303 | 3.5282 |
| 1 | 2,504 | 0.8145 | 0.8258 |
| 2 | 5,009 | 0.7681 | 0.7934 |
| 3 | 7,518 | 0.7349 | 0.7781 |
| 4 | 10,023 | 0.7048 | **0.7759** |
| 5 | 12,525 | 0.6720 | 0.7854 |
| 6 | 15,033 | 0.6410 | 0.8021 |
| 7 | 17,535 | 0.6122 | 0.8282 |
| 8 | 20,036 | 0.5822 | 0.8633 |
| 9 | 22,541 | 0.5568 | 0.8999 |
| 10 | 25,048 | 0.5369 | 0.9354 |
| 11 | 27,549 | 0.5155 | 0.9735 |
| 12 | 30,050 | 0.5029 | 1.0015 |
| 13 | 32,553 | 0.4842 | 1.0442 |
| 14 | 35,058 | 0.4714 | 1.0726 |
| 15 | 37,566 | 0.4642 | 1.0897 |
| 16 | 40,069 | 0.4530 | 1.1323 |
| 17 | 42,575 | 0.4491 | 1.1533 |
| 18 | 45,079 | 0.4403 | 1.1864 |
| 19 | 47,575 | 0.4358 | 1.1921 |
| 20 | 50,080 | 0.4285 | 1.2170 |

The best saved validation checkpoint is epoch 4 (`epoch-004.0-step-010023.pt`, loss 0.7759). Continued constant-rate training overfits: train loss keeps falling while validation loss rises. At epoch 20, clever next-guess loss is 0.1049, but simple, random, and partly-random next-guess losses have worsened to 2.0807, 2.5255, and 1.6620 respectively. This failure mode directly motivates v2's clever expert targets and masked unobservable targets.

### 2026-08-14 — V2 experiments A and B

Both experiments use the same 814,627-parameter architecture, initialization seed, optimizer settings, data splits, and expert objective. Validation is checked once per token-equivalent epoch. Training stops after four checks without an improvement of at least `1e-4`; every strict validation improvement overwrites that stage's `best.pt`.

| Experiment/stage | Initialization | Best epoch | Train loss | Validation loss | Stopped after |
|---|---|---:|---:|---:|---:|
| A: expert only | Random | 6 | 0.268740 | 0.412606 | Epoch 10 |
| B1: mechanics | Random | 9 | 0.000344 | 0.000515 | Epoch 13 |
| B2: expert SFT | Best B1 mechanics | 8 | 0.237188 | **0.411275** | Epoch 12 |

Mechanics pretraining reduced the best expert validation loss by `0.001332`, or `0.323%`, relative to expert-only training. This is a small advantage from one seed, not yet strong evidence that mechanics pretraining improves policy imitation. Mechanics itself is learned almost perfectly. Its specialized checkpoint initially has very high expert loss (`21.4903`), but expert SFT recovers within the first epoch and reaches its best result two epochs later than experiment A.

Reproduce the runs with:

```bash
uv run --with-requirements requirements.txt \
  python experiments_v2.py --experiment a --patience 4 --max-epochs 100

uv run --with-requirements requirements.txt \
  python experiments_v2.py --experiment b-mechanics --patience 4 --max-epochs 100

uv run --with-requirements requirements.txt \
  python experiments_v2.py \
  --experiment b-expert \
  --initial-checkpoint runs/v2-experiment-b-mechanics/checkpoints/best.pt \
  --patience 4 \
  --max-epochs 100
```

### 2026-08-14 — Mechanics retention, seeded replicates, C, and D

All gameplay measurements use greedy decoding for at most six guesses on all 72 secrets in the fixed test-secret split. A generated token outside `a`–`z` or a five-letter word outside `words.txt` is counted as one invalid guess and ends that game. Average guesses is calculated over wins.

#### B2 catastrophic forgetting

Evaluating the best B2 expert checkpoint on the unchanged mechanics validation split gives:

| Checkpoint | Mechanics validation loss |
|---|---:|
| B1: after mechanics | 0.000515 |
| B2: after expert SFT | **10.392833** |

Expert SFT increased mechanics loss approximately 20,166-fold. The sequential B pipeline therefore does not retain the mechanics behavior it learned.

#### Seed-zero held-out gameplay

| Checkpoint | Wins | Win rate | Average guesses | Invalid guesses |
|---|---:|---:|---:|---:|
| A: expert only | 67/72 | 93.06% | 3.0746 | 3 |
| B: mechanics → expert | 71/72 | **98.61%** | 3.0845 | 1 |

#### A and B across seeds 0, 1, and 2

| Seed | A expert validation | B expert validation | A wins | B wins | A invalid | B invalid |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.412606 | **0.411275** | 67/72 | **71/72** | 3 | **1** |
| 1 | **0.399553** | 0.429818 | 68/72 | **72/72** | 4 | **0** |
| 2 | **0.398009** | 0.456313 | **71/72** | 69/72 | **1** | 2 |
| Mean / aggregate | **0.403389 ± 0.008019** | 0.432468 ± 0.022636 | 206/216 | **212/216** | 8 | **3** |

Across three seeds, A has better and substantially less variable expert validation loss. B nevertheless has the better aggregate gameplay win rate, 98.15% versus 95.37%, and slightly fewer guesses per win, 3.0896 versus 3.1165. These 216 games reuse the same 72-secret test split across three trained checkpoints, so they do not establish statistical significance.

B's post-SFT mechanics validation losses are 10.3928, 12.5295, and 15.1019 for seeds 0, 1, and 2. Catastrophic mechanics forgetting is consistent across all three runs.

#### Experiment C: V1 trajectory pretraining → expert

The 32-token V1 epoch-4 checkpoint is expanded to the 35-token V2 model by copying all shared parameters and the first 32 token/output rows. The three new role-token rows retain their deterministic seed-zero initialization.

| Best expert epoch | Expert validation | Mechanics validation after SFT | Wins | Average guesses | Invalid |
|---:|---:|---:|---:|---:|---:|
| 6 | 0.420855 | 12.237598 | 71/72 (98.61%) | 3.0845 | 0 |

At seed zero, V1 trajectory pretraining does not improve expert validation loss over A (`0.412606`) or B (`0.411275`), although its greedy gameplay result matches B's win count.

#### Experiment D: mechanics → candidate consistency → expert

Candidate consistency uses the existing 35-token model and vocabulary. Each eligible observable state supplies one consistent and one inconsistent candidate:

```text
<M><S>candidate + observable history + <F> + label + <E>
```

Only the binary label contributes to loss: `2` means consistent and `0` means inconsistent. The artifact contains 148,272 balanced examples from 74,136 eligible states. D reuses the seed-zero B1 mechanics checkpoint, trains candidate consistency, then applies the same expert SFT used by A–C.

| Stage/evaluation | Result |
|---|---:|
| Best consistency validation loss, epoch 9 | 0.073051 |
| Best expert validation loss, epoch 11 | 0.481911 |
| Post-SFT mechanics validation loss | 14.352221 |
| Post-SFT consistency validation loss | 8.972896 |
| Test gameplay | 71/72 wins (98.61%) |
| Average guesses / invalid guesses | 3.1268 / 0 |

D has the worst expert validation loss of the seed-zero pipelines and forgets both prior objectives during expert SFT. Sequential prerequisite training alone does not preserve mechanics or consistency; retaining those capabilities would require an explicitly joint or replayed objective.

Reproduce C and D with:

```bash
uv run --with-requirements requirements.txt \
  python experiments_v2.py \
  --experiment c-expert \
  --initial-checkpoint runs/v1-full-20epochs/checkpoints/epoch-004.0-step-010023.pt

uv run --with-requirements requirements.txt python dataset_consistency.py

uv run --with-requirements requirements.txt \
  python experiments_v2.py \
  --experiment d-consistency \
  --data-dir data/wordle-v2-consistency \
  --initial-checkpoint runs/v2-experiment-b-mechanics/checkpoints/best.pt

uv run --with-requirements requirements.txt \
  python experiments_v2.py \
  --experiment d-expert \
  --initial-checkpoint runs/v2-experiment-d-consistency/checkpoints/best.pt
```

Evaluate any final checkpoint on the 72 held-out secrets with:

```bash
uv run --with-requirements requirements.txt \
  python evaluate_v2.py PATH_TO_CHECKPOINT
```
