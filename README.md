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

### 2026-08-14 — Experiment E: multi-task replay

`experiments_replay.py` preserves the original B sampling contract: expert batches are sampled with replacement from the same seed, and epoch boundaries are based on cumulative supervised expert tokens. A reported epoch therefore contains one token-equivalent expert epoch plus replay batches. Separate deterministic random streams sample each objective, and replay batches are spread evenly through the expert updates.

Every epoch records exact expert and mechanics validation loss, optional consistency validation loss, gameplay on all 72 test secrets, average guesses, and invalid guesses. Best checkpoints are still selected only by expert validation loss with four-check early stopping.

All E runs start from the seed-zero B1 mechanics checkpoint at learning rate `3e-4`.

| Run | Expert / mechanics | Best epoch | Expert val | Mechanics val | Wins | Avg guesses | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| E1 | 100% / 0% | 8 | 0.411275 | 10.392833 | 71/72 | 3.0845 | 1 |
| E2 | 95% / 5% | 10 | 0.413113 | 0.007317 | **72/72** | **3.0833** | 0 |
| E3 | 90% / 10% | 8 | **0.409605** | **0.003035** | 71/72 | 3.0986 | 0 |

E1 exactly reproduces the existing B2 checkpoint's expert and mechanics validation losses. Five-percent replay prevents catastrophic forgetting with only a `0.001838` expert-loss regression and produces the only perfect-gameplay best checkpoint. Ten-percent replay both preserves mechanics and slightly improves expert validation by `0.001670` relative to E1. Mechanics replay is therefore effective; 10% is the best validation tradeoff of these three ratios.

#### Mechanics and consistency replay

These runs start from the best D consistency checkpoint, after mechanics and candidate-consistency pretraining.

| Expert / mechanics / consistency | Best epoch | Expert val | Mechanics val | Consistency val | Wins | Avg guesses | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| 80% / 10% / 10% | 12 | **0.463436** | **0.003317** | **0.075620** | **70/72** | **3.0571** | 0 |
| 90% / 5% / 5% | 13 | 0.488350 | 0.009833 | 0.094605 | 68/72 | 3.0588 | 0 |

For comparison, D without replay ended at expert/mechanics/consistency losses `0.481911 / 14.352221 / 8.972896`. The 80/10/10 mixture improves expert loss while retaining both prerequisite objectives near their pre-SFT values. The 90/5/5 allocation is worse on all three validation objectives. However, both consistency-replay runs remain worse on expert validation and gameplay than mechanics-only E3.

#### Expert-only SFT learning rate

Each run starts from the same B1 checkpoint and uses no replay.

| Learning rate | Best epoch | Expert val | Mechanics val | Wins | Avg guesses | Invalid |
|---:|---:|---:|---:|---:|---:|---:|
| `3e-4` | 8 | **0.411275** | 10.392833 | 71/72 | 3.0845 | 1 |
| `1e-4` | 16 | 0.499578 | **9.285607** | **72/72** | **3.0833** | 0 |
| `3e-5` | 46 | 0.544856 | 9.495105 | 70/72 | 3.0857 | 1 |

Lowering the learning rate delays convergence and modestly lowers the final mechanics loss, but mechanics loss remains above `9`. It does not solve forgetting and substantially worsens expert validation. Replay is much more effective than reducing the SFT learning rate.

Example replay commands:

```bash
uv run --with-requirements requirements.txt \
  python experiments_replay.py \
  --output-dir runs/v2-replay/e3-expert90-mechanics10 \
  --initial-checkpoint runs/v2-experiment-b-mechanics/checkpoints/best.pt \
  --expert-ratio 0.9 \
  --mechanics-ratio 0.1 \
  --consistency-ratio 0

uv run --with-requirements requirements.txt \
  python experiments_replay.py \
  --output-dir runs/v2-replay/consistency-expert80-mechanics10-consistency10 \
  --initial-checkpoint runs/v2-experiment-d-consistency/checkpoints/best.pt \
  --expert-ratio 0.8 \
  --mechanics-ratio 0.1 \
  --consistency-ratio 0.1
```

### 2026-08-15 — Reduced consistency replay and 3.2M scaling

The reduced-consistency sweep starts every run from the same seed-zero B1 mechanics checkpoint. This differs from the earlier 80/10/10 and 90/5/5 runs, which started after a separate consistency-pretraining stage. E2 is reused as the 95/5/0 baseline; its consistency loss was measured post hoc on the same validation split.

| Expert / mechanics / consistency | Best epoch | Expert val | Mechanics val | Consistency val | Wins | Avg guesses | Invalid |
|---|---:|---:|---:|---:|---:|---:|---:|
| 95% / 5% / 0% | 10 | 0.413113 | 0.007317 | 4.516440 | **72/72** | 3.0833 | 0 |
| 94% / 5% / 1% | 11 | 0.420792 | 0.010372 | 0.578969 | 71/72 | 3.0845 | 1 |
| 92% / 5% / 3% | 8 | **0.411279** | **0.006531** | 0.415749 | 69/72 | **3.0725** | 0 |
| 90% / 5% / 5% | 8 | 0.415287 | 0.011212 | **0.248635** | 68/72 | 3.0882 | 3 |

One percent consistency replay sharply reduces consistency loss but regresses expert validation. Three percent gives the best expert and mechanics validation results while reducing baseline consistency loss by about 91%. Five percent improves consistency further but loses four games relative to the baseline. The 95/5/0 gameplay baseline and 92/5/3 validation tradeoff were selected for scaling.

#### Exact 3,202,083-parameter architecture

| Setting | Value |
|---|---:|
| Layers | 4 |
| Model width | 256 |
| Attention heads | 8 |
| Head size | 32 |
| MLP width | 1,024 |
| Context | 96 |
| Vocabulary | 35 |
| Parameters | **3,202,083** |

Architecture metadata is now saved in every new checkpoint. Replay and gameplay evaluation reconstruct the stored architecture automatically; older checkpoints retain the original 128-wide default.

The scaled model was first trained from random initialization on mechanics. Its selected epoch-10 checkpoint reached `0.000249` mechanics validation loss. Both replay runs start from that same checkpoint and otherwise retain the small-model optimizer, sampling, epoch, and early-stopping settings.

| Parameters | Expert / mechanics / consistency | Best epoch | Expert val | Mechanics val | Consistency val | Wins | Avg guesses | Invalid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 814,627 | 95% / 5% / 0% | 10 | 0.413113 | 0.007317 | 4.516440 | 72/72 | 3.0833 | 0 |
| 3,202,083 | 95% / 5% / 0% | 7 | 0.518945 | 0.008585 | 5.646697 | 69/72 | 3.1449 | 0 |
| 814,627 | 92% / 5% / 3% | 8 | 0.411279 | 0.006531 | 0.415749 | 69/72 | 3.0725 | 0 |
| 3,202,083 | 92% / 5% / 3% | 6 | 0.527004 | 0.012708 | 0.590974 | 60/72 | 3.0500 | 3 |

The larger model is worse under the unchanged training recipe. This does not show that added capacity is harmful; it shows that the 814K optimizer and stopping configuration does not transfer directly. The 3.2M 95/5/0 run is better than its 92/5/3 counterpart on expert validation and gameplay, while 92/5/3 retains far more consistency.

Build the scaled mechanics checkpoint with:

```bash
uv run --with-requirements requirements.txt \
  python experiments_v2.py \
  --experiment b-mechanics \
  --output-dir runs/v2-3m/b-mechanics \
  --context-length 96 \
  --embedding-size 256 \
  --num-layers 4 \
  --num-heads 8 \
  --mlp-size 1024
```

### 2026-08-15 — Instrumented 3.2M learning-rate sweep

All six runs executed sequentially on CUDA using the same 3.2M mechanics checkpoint. Every validation checkpoint now stores:

- `train_losses`: supervised-token-weighted cross-entropy split into expert, mechanics, and consistency objectives
- `expert_validation_loss`, `mechanics_validation_loss`, and optional `consistency_validation_loss`
- `gradient_norm`: mean global L2 gradient norm over optimizer steps in the preceding token-equivalent epoch
- `gradient_norms`: the same mean split by the objective that produced each update
- the optimizer's actual `learning_rate`

The gradient norm is measured after `backward()` and before `optimizer.step()` without clipping. Epoch zero has no preceding updates, so its gradient fields are null or empty. `run.json` records the selected device as `cuda`.

#### 95% expert / 5% mechanics

| LR | Best epoch | Train expert | Train mechanics | Val expert | Val mechanics | Gradient norm | Expert grad | Mechanics grad | Wins | Invalid |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `3e-4` | 7 | 0.321783 | 0.010585 | **0.518945** | 0.008585 | **1.3225** | 1.3701 | 0.4184 | **69/72** | **0** |
| `2e-4` | 9 | **0.298345** | 0.007579 | 0.560944 | 0.005484 | 1.6681 | 1.7346 | 0.4070 | 68/72 | 1 |
| `1e-4` | 11 | 0.402068 | **0.003420** | 0.598525 | **0.003225** | 2.2580 | 2.3646 | **0.2365** | 64/72 | 3 |

#### 92% expert / 5% mechanics / 3% consistency

| LR | Best epoch | Train expert / mechanics / consistency | Val expert / mechanics / consistency | Gradient norm | Expert / mechanics / consistency grad | Wins | Invalid |
|---:|---:|---:|---:|---:|---:|---:|---:|
| `3e-4` | 6 | 0.397043 / 0.014382 / 0.594879 | **0.527004** / 0.012708 / 0.590974 | **1.3597** | 1.3747 / 0.4769 / 2.3672 | 60/72 | 3 |
| `2e-4` | 10 | **0.279321** / 0.007000 / 0.447983 | 0.568743 / 0.011506 / 0.443192 | 1.6858 | 1.7203 / 0.3310 / 2.8773 | **67/72** | 2 |
| `1e-4` | 12 | 0.407765 / **0.005059** / **0.445089** | 0.599106 / **0.005989** / **0.387397** | 2.2483 | 2.3275 / **0.2244** / 3.1882 | 65/72 | **0** |

`3e-4` remains the best learning rate for expert validation in both mixtures. Lower rates improve mechanics and consistency retention but materially worsen expert validation. Gradient norms remain finite; the larger norms at lower rates accompany slower optimization and later selected epochs rather than an observed gradient explosion. Consistency updates have the largest per-objective gradients, while mechanics updates have the smallest.

Expert-only checkpoint selection also hides useful later tradeoffs. For 92/5/3 at `2e-4`, epoch 11 reaches 71/72 wins, zero invalid guesses, and consistency loss `0.405932`, while expert loss changes only from the selected epoch-10 value `0.568743` to `0.568982`. That epoch is recorded in `metrics.jsonl` but is not the saved `best.pt` because expert validation did not strictly improve.

### 2026-08-16 — Diverse clever targets and strict secret evaluation

Checkpoint selection and early stopping now use the same validation-gameplay ordering:

1. maximize wins on the 72 validation secrets
2. minimize invalid guesses
3. minimize average guesses among wins
4. minimize expert validation loss

`experiments_replay.py` no longer evaluates the fixed test split while training, and `evaluate_v2.py` now defaults to `--split validation`. The fixed 72 test secrets require an explicit `--split test`; they are reserved for final development comparisons.

#### Unique off-policy expert states

`dataset_expert.py` generated 200,000 unique observable histories, each relabeled with the clever minimum-expected-survivors action regardless of the policy that reached it. Nested prefixes contain 10K, 50K, 100K, and 200K states.

| Source behavior | Unique states |
|---|---:|
| Random legal | 39,715 |
| Simple consistent | 17,299 |
| Maximum entropy | 791 |
| Partly random, rates 0.05/0.15/0.30/0.50/0.75/0.90 | 108,508 |
| Deliberately poor sampled legal action | 33,687 |
| **Total** | **200,000** |

The fixed development pool contains 177,642 training and 22,358 validation histories sourced only from the fixed 575/72 development secrets. Full validation recomputed every history, feedback sequence, state fingerprint, answer set, clever target, serialization, and loss mask. It found 200,000 unique histories and 65,779 unique remaining-answer sets.

Deduplication leaves only one empty history, although every game needs the same initial clever action. That canonical answer-independent state therefore carries training-only sampling weight 10,000 while remaining one stored unique state. Validation remains unweighted. This prevents a unique-state corpus from accidentally teaching the opening action only once.

#### Development and benchmark modes

- **Development:** fixed 575 train / 72 validation / 72 test secrets, normally one seed. Validation drives checkpoint selection, early stopping, tuning, and development gameplay.
- **Benchmark:** five deterministic test folds of 144/144/144/144/143 secrets. Each run uses 72 validation secrets and 503/503/503/503/504 training secrets from the remaining folds.

Every one of the 719 secrets is held out exactly once per benchmark seed. Both expert histories and mechanics examples are filtered by source secret before training; fold manifests report zero held-out examples and the exact number omitted. The benchmark compares nested 100K and 200K expert pools using identical folds, mechanics initialization, replay ratio, and model seed.

```bash
# Recreate modes and universal pools
uv run --with-requirements requirements.txt python cross_validation.py \
  --mode benchmark --output data/wordle-cv5.json --model-seeds 0 1 2
uv run --with-requirements requirements.txt python dataset_expert.py \
  --output-dir data/wordle-v2-diverse-cv --include-test
uv run --with-requirements requirements.txt python dataset_mechanics_cv.py

# Materialize strict folds, then run 5 folds × 3 seeds × both nested sizes
uv run --with-requirements requirements.txt python benchmark_cv.py --prepare-only
uv run --with-requirements requirements.txt python benchmark_cv.py --skip-prepare
```

Each seed produces one combined 719-secret artifact per model and a paired per-secret comparison. Reports include A-loses/B-wins, A-wins/B-loses, both-win, both-lose, and `B guesses - A guesses` for every secret. The aggregate reports mean and sample standard deviation across the three initializations.

#### Three-seed five-fold result

Each entry below aggregates five disjoint test folds, so every seed contains exactly 719 held-out games.

| Expert pool | Wins / 719 | Win rate | Avg guesses on wins | Avg attempts, all secrets | Invalid guesses |
|---|---:|---:|---:|---:|---:|
| 100K, seed 0 | 249 | 34.63% | 3.4940 | 4.6592 | 174 |
| 100K, seed 1 | 259 | 36.02% | 3.4054 | 4.4826 | 209 |
| 100K, seed 2 | 238 | 33.10% | 3.5000 | 4.5508 | 213 |
| **100K mean ± SD** | **248.7 ± 10.5** | **34.59% ± 1.46%** | **3.4665 ± 0.0530** | **4.5642 ± 0.0891** | **198.7 ± 21.5** |
| 200K, seed 0 | 397 | 55.22% | 3.4358 | 4.2323 | 147 |
| 200K, seed 1 | 358 | 49.79% | 3.3687 | 4.2782 | 156 |
| 200K, seed 2 | 377 | 52.43% | 3.4244 | 4.2949 | 139 |
| **200K mean ± SD** | **377.3 ± 19.5** | **52.48% ± 2.71%** | **3.4096 ± 0.0359** | **4.2684 ± 0.0324** | **147.3 ± 8.5** |

Paired outcomes compare the 100K model as A and the nested 200K model as B:

| Outcome | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---:|---:|---:|---:|
| A loses, B wins | 212 | 166 | 199 | **192.3 ± 23.7** |
| A wins, B loses | 64 | 67 | 60 | **63.7 ± 3.5** |
| Both win | 185 | 192 | 178 | **185.0 ± 7.0** |
| Both lose | 258 | 294 | 282 | **278.0 ± 18.3** |
| Mean `B attempts - A attempts` | -0.4270 | -0.2045 | -0.2559 | **-0.2958 ± 0.1165** |

Doubling the unique expert pool produces a large, repeatable gain: approximately 129 additional held-out wins per 719-secret seed, 17.90 percentage points of win rate, 51 fewer invalid guesses, and 0.296 fewer attempts per secret. The absolute 52.48% win rate remains poor, however. The CV result rejects any claim that this off-policy corpus and current 3.2M training recipe are near the desired 704/719 regime; many more states help substantially but do not solve policy generalization or invalid generation.

Exact combined predictions and the list of changed secrets are under `runs/cv5-diverse/seed-{0,1,2}/`. Cross-seed metrics are in `runs/cv5-diverse/aggregate.json`.

### 2026-08-20 — 500K logical-state scaling and constrained decoding

This experiment holds the 3,202,083-parameter architecture and successful
replay recipe fixed while extending the unique expert corpus from 200K to
500K observable histories. All expert targets remain labels from the
minimum-expected-survivors solver; source behavior only determines which
reachable state is collected.

#### Corpus construction and validation

`build_500k_pools.py` reproduces the original generation stream through state
200,000, then accepts a state whose remaining-answer-set fingerprint has
already appeared with probability 0.5. Novel answer sets are always accepted.
This leaves the nested 200K prefix field-identical to the old corpus while
biasing only the 200K–500K tail toward logical novelty.

```bash
uv run --with-requirements requirements.txt python build_500k_pools.py dev
uv run --with-requirements requirements.txt python build_500k_pools.py cv
```

The development pool uses only the fixed 575 train and 72 validation source
secrets. None of the 72 fixed development-test secrets contributes an
example. The universal CV pool contains all 719 source secrets, but fold
materialization removes every held-out test source before training.

| Development-pool prefix | Histories | Unique remaining-answer sets | New sets |
|---|---:|---:|---:|
| 100K | 100,000 | 37,989 | — |
| 200K | 200,000 | 65,779 | +27,790 |
| 500K | 500,000 | 167,621 | +101,842 |

The universal CV pool has 37,992, 65,859, and 168,667 unique answer sets at
the same prefixes. Thus 200K → 500K adds 300,000 histories and 101,842 logical
states in the development pool: about 0.339 new answer sets per added history.

| Source behavior, development pool | States |
|---|---:|
| Random legal | 101,114 |
| Simple consistent | 43,105 |
| Maximum entropy | 791 |
| Partly random, rates 0.05/0.15/0.30/0.50/0.75/0.90 | 264,746 |
| Deliberately poor sampled legal action | 90,244 |
| **Total** | **500,000** |

The resulting development pool contains 133,082 answer sets used once,
28,795 used 2–5 times, 2,868 used 6–10 times, 1,939 used 11–50 times,
197 used 51–100 times, and 740 used more than 100 times. Candidate-set sizes
remain broad:

| Remaining candidates | Stored states | Distinct answer sets |
|---|---:|---:|
| 1 | 203,698 | 647 |
| 2–5 | 145,663 | 47,280 |
| 6–10 | 55,832 | 41,967 |
| 11–50 | 76,852 | 64,999 |
| 51–100 | 12,146 | 9,766 |
| 101–250 | 5,774 | 2,933 |
| 251–500 | 34 | 28 |
| 501–719 | 1 | 1 |

`validation.json` records a complete recomputation over all 500,000 records:
unique history, reachable feedback, source secret still in the candidate set,
clever target, serialization, token IDs, loss mask, source metadata, answer-set
fingerprint, and manifest distributions. The compressed artifacts are pinned
in their manifests:

- development `examples.jsonl.gz` SHA-256:
  `96dd7a0620701fe9db2cb2437dc3c2746a5e7972a2428984952e4d8ec6737ecf`
- CV `examples.jsonl.gz` SHA-256:
  `f8b9022973318305d7d4e0589f91e8d1d8860e5472795be5c94eb0387d14b9ad`

The canonical empty-history state remains one stored record with training-only
sampling weight 10,000. Validation, development gameplay, and benchmark
gameplay remain unweighted.

#### Fixed training and decoding configuration

The model remains four decoder blocks, width 256, eight heads, MLP width 1024,
context length 96: 3,202,083 parameters. Training uses batch size 32,
evaluation batch size 256, learning rate `3e-4`, at most 100 expert epochs,
patience 4, and 95% expert / 5% mechanics replay from the same seeded mechanics
initialization. Checkpoints are selected by maximum validation wins, minimum
invalid guesses, minimum average guesses among wins, then minimum expert
validation loss.

Every `metrics.jsonl` record includes optimizer step, epoch/effective pass,
learning rate, expert and mechanics train/validation losses, validation wins,
validation invalid guesses, and average guesses among wins. New 500K records
include exact supervised tokens seen; reused 200K records are backfilled with
exact sampled examples seen (`optimizer step × batch size`). Test secrets do
not participate in selection.

Raw decoding is unchanged greedy autoregressive generation. Constrained
decoding masks each next-letter distribution to letters that continue at least
one allowed-word prefix; after five positions the generated sequence is
therefore an allowed guess. It does not repair text or otherwise rescore the
model policy.

#### Fixed development split

These results use the selected seed-0 checkpoints and the 72 fixed development
test secrets, after all selection on the separate 72 validation secrets:

| Expert pool / decoding | Wins / 72 | Win rate | Avg guesses on wins | Avg attempts | Invalid |
|---|---:|---:|---:|---:|---:|
| 200K raw | 53 | 73.61% | 3.4906 | 3.8472 | 10 |
| 200K constrained | 59 | 81.94% | 3.5593 | 4.0000 | 0 |
| 500K raw | 60 | 83.33% | 3.4500 | 3.7917 | 4 |
| 500K constrained | 62 | 86.11% | 3.5323 | 3.8750 | 0 |

For 200K vs 500K, raw decoding has 10 A-loses/B-wins, 3
A-wins/B-loses, 50 both-win, and 9 both-lose; mean `B attempts - A
attempts` is -0.0556. Constrained decoding has 5, 2, 57, and 8,
respectively; mean attempt delta is -0.1250. Constraining 200K converts six
raw losses into wins with no reversals; constraining 500K converts two with no
reversals. Exact changed secrets and guess sequences are in
`runs/dev-500k/seed-0/paired-*.json`.

#### Three-seed, five-fold raw and constrained benchmark

Each row combines five disjoint folds, covering all 719 secrets exactly once.
The 200K raw rows reuse the prior selected checkpoints and predictions from the
field-identical nested prefix.

| Expert pool / decoding | Seed | Wins / 719 | Win rate | Avg guesses on wins | Avg attempts | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| 200K raw | 0 | 397 | 55.22% | 3.4358 | 4.2323 | 147 |
| 200K raw | 1 | 358 | 49.79% | 3.3687 | 4.2782 | 156 |
| 200K raw | 2 | 377 | 52.43% | 3.4244 | 4.2949 | 139 |
| **200K raw mean ± SD** | — | **377.3 ± 19.5** | **52.48% ± 2.71%** | **3.4096 ± 0.0359** | **4.2684 ± 0.0324** | **147.3 ± 8.5** |
| 200K constrained | 0 | 438 | 60.92% | 3.5457 | 4.5049 | 0 |
| 200K constrained | 1 | 393 | 54.66% | 3.4784 | 4.6217 | 0 |
| 200K constrained | 2 | 415 | 57.72% | 3.5446 | 4.5828 | 0 |
| **200K constrained mean ± SD** | — | **415.3 ± 22.5** | **57.77% ± 3.13%** | **3.5229 ± 0.0385** | **4.5698 ± 0.0595** | **0.0 ± 0.0** |
| 500K raw | 0 | 454 | 63.14% | 3.4163 | 4.0987 | 112 |
| 500K raw | 1 | 487 | 67.73% | 3.3860 | 3.9875 | 92 |
| 500K raw | 2 | 488 | 67.87% | 3.4426 | 4.0654 | 81 |
| **500K raw mean ± SD** | — | **476.3 ± 19.3** | **66.25% ± 2.69%** | **3.4150 ± 0.0283** | **4.0505 ± 0.0571** | **95.0 ± 15.7** |
| 500K constrained | 0 | 494 | 68.71% | 3.5243 | 4.2990 | 0 |
| 500K constrained | 1 | 524 | 72.88% | 3.4676 | 4.1544 | 0 |
| 500K constrained | 2 | 514 | 71.49% | 3.5058 | 4.2170 | 0 |
| **500K constrained mean ± SD** | — | **510.7 ± 15.3** | **71.02% ± 2.12%** | **3.4992 ± 0.0289** | **4.2235 ± 0.0725** | **0.0 ± 0.0** |

Paired 200K (A) vs 500K (B):

| Decoding / seed | A loses, B wins | A wins, B loses | Both win | Both lose | Mean `B attempts - A attempts` |
|---|---:|---:|---:|---:|---:|
| Raw / 0 | 140 | 83 | 314 | 182 | -0.1335 |
| Raw / 1 | 183 | 54 | 304 | 178 | -0.2907 |
| Raw / 2 | 165 | 54 | 323 | 177 | -0.2295 |
| Constrained / 0 | 131 | 75 | 363 | 150 | -0.2058 |
| Constrained / 1 | 179 | 48 | 345 | 147 | -0.4673 |
| Constrained / 2 | 150 | 51 | 364 | 154 | -0.3658 |

Paired 500K raw (A) vs constrained (B):

| Seed | Raw loses, constrained wins | Raw wins, constrained loses | Both win | Both lose |
|---|---:|---:|---:|---:|
| 0 | 40 | 0 | 454 | 225 |
| 1 | 37 | 0 | 487 | 195 |
| 2 | 26 | 0 | 488 | 205 |

Exact predictions and changed-secret records are under
`runs/cv5-500k/seed-{0,1,2}/`; cross-seed metrics are in
`runs/cv5-500k/aggregate.json`.

#### Interpretation

The 500K corpus materially improves generalization in every seed and both
decoding modes. Relative to 200K, raw wins rise by 99.0 per seed on average
(+13.77 percentage points), invalid failures fall by 52.3, and attempts fall
by 0.218 per secret. Constrained wins rise by 95.3 (+13.26 points), with 500K
winning substantially more paired disagreements in every seed.

The gain is not proportional to logical-state growth. Unique remaining-answer
sets rise from 65,779 to 167,621 (2.55×), while constrained win rate rises from
57.77% to 71.02%. Compared with the earlier 100K → 200K gain of 17.90 raw
points, 200K → 500K gains 13.77 raw points despite adding many more logical
states. Data scaling is still effective, but returns are diminishing.

Illegal generation is a meaningful, not dominant, bottleneck. At 500K,
constraints remove 95 invalid failures per seed and convert 34.3 losses into
wins on average with no reverse flips, a +4.78-point gap. The constrained
71.02% result also shows that most remaining failures are strategic rather
than lexical.

The evidence supports retaining the 500K corpus and testing a larger model as
the next controlled axis. More data is not exhausted—the paired 500K gain is
large and consistent—but the smaller gain per added logical state and modest
raw/constrained gap now provide a concrete reason to test whether 3.2M
parameters, rather than legal-word generation, is becoming the tighter limit.

### 2026-08-21 — 1M logical-state scaling with solver top actions

#### Corpus construction

The 500K stream is continued to 1,000,000 unique observable states with the
same generator settings (seed 20260815, identical behaviors, answer-set bias
`repeat_probability=0.5` from state 200,000). Every record additionally
stores `top_guesses`: the eight legal guesses ranked by the
minimum-expected-survivors solver together with their expected-survivor
scores. This field is metadata only — generation, the expert target (always
`top_guesses[0]`, identical to the classic clever target), training, and
evaluation are unchanged.

The first 500,000 records of each 1M pool are field-identical to the
corresponding 500K pool; `build_1m_pools.py` verifies the prefix, and
`seed_1m_runs.py` re-verifies every materialized fold on the exact fields the
training loader reads before reusing the 200K/500K checkpoints. Only the 1M
variant is retrained (15 new checkpoints: 5 folds × 3 seeds).

| Metric | 100K | 200K | 500K | 1M |
|---|---:|---:|---:|---:|
| Unique observable histories | 100,000 | 200,000 | 500,000 | 1,000,000 |
| Unique remaining-answer sets (dev pool) | 37,992 | 65,779 | 167,621 | 303,212 |
| New answer sets added by the step | — | 27,787 | 101,842 | 135,591 |

The development pool splits 888,231 train / 111,769 validation; the CV pool
799,754 train / 101,076 validation / 99,170 test. Answer-set reuse stays
broad: 238,445 sets used once, 53,847 used 2–5 times, 5,569 used 6–10 times,
3,932 used 11–50 times, 488 used 51–100 times, and 931 used more than 100
times. Candidate-set sizes remain varied:

| Remaining candidates | Stored states | Distinct answer sets |
|---|---:|---:|
| 1 | 411,937 | 647 |
| 2–5 | 290,939 | 47,280 |
| 6–10 | 112,298 | 41,967 |
| 11–50 | 152,483 | 64,999 |
| 51–100 | 22,990 | 9,766 |
| 101–250 | 9,318 | 2,933 |
| 251–500 | 34 | 28 |
| 501–719 | 1 | 1 |

Source behaviors: random legal 206,420; simple consistent 85,004; maximum
entropy 791; partly random (rates 0.05/0.15/0.30/0.50/0.75/0.90) 9,197 +
27,403 + 56,329 + 98,187 + 151,926 + 184,924; deliberately poor 179,819.
`validation.json` recomputes every record (unique history, reachable
feedback, source secret still candidate, clever target, stored top actions,
serialization, token IDs, loss mask, source metadata). Artifacts pinned in
their manifests:

- development `examples.jsonl.gz` SHA-256:
  `4fff02cb7c8b2a4488a9f9b68763e5d625b909201c17ad4e8be9f3bef1f483a4`
- CV `examples.jsonl.gz` SHA-256:
  `ee693c17f1ac4c37b28abd02347ba466c325e719471cb76e5a182ad6879a5dd8`

Training and decoding configuration is unchanged (3,202,083 parameters, 95/5
expert/mechanics replay, same checkpoint-selection order). Materialized fold
data is regenerable from pool + mode file and is gitignored; pools, run
artifacts, and checkpoints are committed. Each 1M pool archive exceeds
GitHub's 100 MB per-file limit, so it is committed as
`examples.jsonl.gz.part-0` + `examples.jsonl.gz.part-1`; run
`python join_pools.py` to reassemble and verify `examples.jsonl.gz` against
the manifest SHA-256.

#### Fixed development split

Seed-0 checkpoints, 72 fixed development test secrets:

| Expert pool / decoding | Wins / 72 | Win rate | Avg guesses on wins | Avg attempts | Invalid |
|---|---:|---:|---:|---:|---:|
| 200K raw | 53 | 73.61% | 3.4906 | 3.8472 | 10 |
| 200K constrained | 59 | 81.94% | 3.5593 | 4.0000 | 0 |
| 500K raw | 60 | 83.33% | 3.4500 | 3.7917 | 4 |
| 500K constrained | 62 | 86.11% | 3.5323 | 3.8750 | 0 |
| 1M raw | 57 | 79.17% | 3.1754 | 3.6528 | 6 |
| 1M constrained | 59 | 81.94% | 3.2203 | 3.7222 | 0 |

On this 72-secret sample the 1M model wins slightly fewer games than 500K
(500K-vs-1M raw: 7 A-loses/B-wins vs 4 A-wins/B-loses) but is faster on wins
(3.1754 vs 3.4500 average guesses). The sample is too small to be conclusive;
the five-fold benchmark below is the reliable measurement.

#### Three-seed, five-fold raw and constrained benchmark

Each row combines five disjoint folds covering all 719 secrets exactly once.
The 200K and 500K rows reuse the prior selected checkpoints and predictions
(training-identical data, verified per fold); only the 1M rows are new.

| Expert pool / decoding | Seed | Wins / 719 | Win rate | Avg guesses on wins | Avg attempts | Invalid |
|---|---:|---:|---:|---:|---:|---:|
| 200K raw | 0 | 397 | 55.22% | 3.4358 | 4.2323 | 147 |
| 200K raw | 1 | 358 | 49.79% | 3.3687 | 4.2782 | 156 |
| 200K raw | 2 | 377 | 52.43% | 3.4244 | 4.2949 | 139 |
| **200K raw mean ± SD** | — | **377.3 ± 19.5** | **52.48% ± 2.71%** | **3.4096 ± 0.0359** | **4.2684 ± 0.0324** | **147.3 ± 8.5** |
| 200K constrained | 0 | 438 | 60.92% | 3.5457 | 4.5049 | 0 |
| 200K constrained | 1 | 393 | 54.66% | 3.4784 | 4.6217 | 0 |
| 200K constrained | 2 | 415 | 57.72% | 3.5446 | 4.5828 | 0 |
| **200K constrained mean ± SD** | — | **415.3 ± 22.5** | **57.77% ± 3.13%** | **3.5229 ± 0.0385** | **4.5698 ± 0.0595** | **0.0 ± 0.0** |
| 500K raw | 0 | 454 | 63.14% | 3.4163 | 4.0987 | 112 |
| 500K raw | 1 | 487 | 67.73% | 3.3860 | 3.9875 | 92 |
| 500K raw | 2 | 488 | 67.87% | 3.4426 | 4.0654 | 81 |
| **500K raw mean ± SD** | — | **476.3 ± 19.3** | **66.25% ± 2.69%** | **3.4150 ± 0.0283** | **4.0505 ± 0.0571** | **95.0 ± 15.7** |
| 500K constrained | 0 | 494 | 68.71% | 3.5243 | 4.2990 | 0 |
| 500K constrained | 1 | 524 | 72.88% | 3.4676 | 4.1544 | 0 |
| 500K constrained | 2 | 514 | 71.49% | 3.5058 | 4.2170 | 0 |
| **500K constrained mean ± SD** | — | **510.7 ± 15.3** | **71.02% ± 2.12%** | **3.4992 ± 0.0289** | **4.2235 ± 0.0725** | **0.0 ± 0.0** |
| 1M raw | 0 | 510 | 70.93% | 3.3745 | 3.9235 | 80 |
| 1M raw | 1 | 548 | 76.22% | 3.3704 | 3.8456 | 57 |
| 1M raw | 2 | 533 | 74.13% | 3.3996 | 3.9152 | 64 |
| **1M raw mean ± SD** | — | **530.3 ± 19.1** | **73.76% ± 2.66%** | **3.3815 ± 0.0158** | **3.8948 ± 0.0428** | **67.0 ± 11.8** |
| 1M constrained | 0 | 535 | 74.41% | 3.4168 | 4.0779 | 0 |
| 1M constrained | 1 | 572 | 79.55% | 3.4266 | 3.9527 | 0 |
| 1M constrained | 2 | 555 | 77.19% | 3.4577 | 4.0376 | 0 |
| **1M constrained mean ± SD** | — | **554.0 ± 18.5** | **77.05% ± 2.58%** | **3.4337 ± 0.0213** | **4.0227 ± 0.0639** | **0.0 ± 0.0** |

Paired 500K (A) vs 1M (B):

| Decoding / seed | A loses, B wins | A wins, B loses | Both win | Both lose | Mean `B attempts - A attempts` |
|---|---:|---:|---:|---:|---:|
| Raw / 0 | 124 | 68 | 386 | 141 | -0.1752 |
| Raw / 1 | 114 | 53 | 434 | 118 | -0.1419 |
| Raw / 2 | 112 | 67 | 421 | 119 | -0.1502 |
| Constrained / 0 | 106 | 65 | 429 | 119 | -0.2211 |
| Constrained / 1 | 100 | 52 | 472 | 95 | -0.2017 |
| Constrained / 2 | 99 | 58 | 456 | 106 | -0.1794 |

Paired 1M raw (A) vs constrained (B):

| Seed | Raw loses, constrained wins | Raw wins, constrained loses | Both win | Both lose |
|---|---:|---:|---:|---:|
| 0 | 25 | 0 | 510 | 184 |
| 1 | 24 | 0 | 548 | 147 |
| 2 | 22 | 0 | 533 | 164 |

Constrained decoding uses slightly more attempts than raw (mean delta
+0.1544/+0.1071/+0.1224) because it replaces a raw failure with a longer won
game; it never loses a game that raw wins. Exact predictions and changed
secrets are under `runs/cv5-1m/seed-{0,1,2}/`; cross-seed metrics in
`runs/cv5-1m/aggregate.json`.

#### Interpretation

1M continues to improve generalization in every seed and both decoding modes.
Relative to 500K, raw wins rise by 54.0 per seed (+7.51 points), invalid
failures fall by 28.0, and attempts fall by 0.156 per secret; constrained
wins rise by 43.3 (+6.03 points). The 500K→1M step is roughly half as strong
as 200K→500K (+99.0 raw wins, +13.77 points) and the 100K→200K step (+17.90
points). Gains remain consistent: the paired 500K-vs-1M outcome is a win for
1M by a wide margin in every seed (114–124 net converted secrets raw,
99–106 constrained).

The gain is not proportional to logical-state growth. Unique answer sets
rise from 167,621 to 303,212 (+81%), but raw wins rise only +11.3%: about
0.40 wins per 1,000 added answer sets versus 0.97 for 200K→500K. Coverage
still helps, but each additional doubling of logical states is worth less.

Illegal generation is a shrinking, minor bottleneck. Raw invalid guesses fall
147 → 95 → 67 across the three pool sizes; the raw/constrained win-rate gap
shrinks 5.29 → 4.77 → 3.29 points. At 1M, constrained decoding converts 23.7
raw losses into wins per seed with no reversals (+3.29 points), and the
constrained model still loses 165.0 games per seed — most failures are now
strategic, not lexical.

Data scaling has not plateaued but is clearly diminishing: raw gains per step
are +17.90, +13.77, and +7.51 points. The constrained 1M win rate (77.05%) is
still far below the classical solver's near-perfect play, so the corpus is
not saturated, but the 3.2M-parameter model is now the more plausible binding
constraint: per-logical-state returns have roughly halved, and enforcing
lexical validity recovers only a small fraction of the remaining failures.
The next controlled axis should be a larger model trained on the 1M corpus,
holding data and decoding fixed.
