# Teaching a Transformer to Play Wordle: From Catastrophic Forgetting to Anchored Preferences

*What happened when I trained small language models on 1,000,000 game states, why DPO collapsed into gibberish, and how I rescued it.*

**Author:** Harris Oldroyd  
**Published on:** [harrisoldroyd.com](https://harrisoldroyd.com)  

---

Can a small, decoder-only Transformer learn to play Wordle well purely from text sequences?

I don't mean giving an LLM an external Python solver, a board simulator, or a scratchpad to write code. I mean treating Wordle strictly as an autoregressive next-token prediction problem:

$$\mathcal{P}(w_t \mid w_1, w_2, \dots, w_{t-1})$$

The model sees prior guesses and ternary color feedback as text, tracks the remaining candidate words in its hidden activations, and outputs the next 5-letter guess token-by-token.

Over the past few weeks, I built this system from scratch, scaling it from an 814,000-parameter prototype to 12.7 million parameters, and from 1,000 toy trajectories to 1,000,000 unique off-policy logical states. 

Along the way, the model encountered nearly every classic pathology in modern post-training:
1. **Catastrophic forgetting:** Learning game strategy completely erased the model's understanding of the basic rules (a >20,000-fold error spike), which I solved with multi-task experience replay.
2. **The SFT capacity ceiling:** Supervised behavioural cloning on 1M states hit a hard plateau at ~80.4% win rate regardless of whether I used 7.2M or 12.7M parameters.
3. **The DPO illusion:** Direct Preference Optimization drove preference accuracy to 90.7% while crashing actual gameplay to **0% wins**, requiring an anchored loss formulation to rescue.

Here is the experiment story, the empirical telemetry, what broke, and what I'm building next.

---

## 1. The Setup: A 35-Token Language for Wordle

Wordle has 719 valid five-letter words in my dictionary. To keep the model focused entirely on core logic, I encoded the entire game into a minimal **35-token vocabulary**:
- **26 letters:** `a`–`z`
- **3 feedback digits:** `0` (gray / miss), `1` (yellow / wrong position), `2` (green / exact hit)
- **6 control tokens:** `<G>` (guess), `<F>` (feedback), `<E>` (end), `<M>` (mechanics), `<S>` (secret), `<P>` (policy)

A game is simply serialized as an alternating string of guesses and feedback:

```text
Policy prompt & rollout:
<P><G>could<F>22010<G>colon<F>22222<E>

Mechanics rule prompt:
<M><S>colon<G>could<F>22010<E>
```

I used standard pre-norm decoder-only Transformers with causal self-attention, GELU feed-forwards, and learned 1D positional embeddings.

| Model Size | Parameters | Layers | Width ($d_{\text{model}}$) | Heads | MLP Dim | Context |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base** | 814,627 | 4 | 128 | 4 | 512 | 96 |
| **3.2M** | 3,202,083 | 4 | 256 | 8 | 1024 | 96 |
| **7.2M** | 7,162,403 | 4 | 384 | 12 | 1536 | 96 |
| **12.7M** | 12,695,587 | 4 | 512 | 16 | 2048 | 96 |

---

## 2. Memorization, Overfitting, and The 20,000x Forgetting Spike

My first sanity check was simple: memorizing 32 trajectories. At 1,000 steps, greedy autoregressive decoding reproduced 19-token continuations with 100% exact match. 

Next, I trained the base model on mixed game trajectories (combining optimal moves, simple consistent moves, and random legal guesses). Validation loss dropped quickly for syntax (`<G>`, `<F>` fell to 0.12) and feedback digits (0.66), but hit a hard floor on guess letters (1.07). 

When I pushed training through 20 full epochs, the model hit its best validation checkpoint at **Epoch 4 (0.7759)** and then overfitted severely:

| Target Type | Epoch 4 Loss | Epoch 20 Loss | Status |
| :--- | :---: | :---: | :---: |
| **Train Loss (overall)** | 0.7048 | 0.4285 | Continues dropping |
| **Validation Loss (overall)** | **0.7759** | 1.2170 | Heavy overfitting |
| **Clever next-guess letters** | 0.2278 | 0.1049 | Over-memorized |
| **Random next-guess letters** | 1.4643 | 2.5255 | Severe blow-up |

Because the model was not conditioned on player intent, forcing it to predict random moves caused it to hallucinate. This prompted me to split the architecture into explicit roles: `<M>` for mechanics and `<P>` for policy.

### The Catastrophic Forgetting Crisis

I tested a classic two-stage curriculum:
- **Stage 1 (Mechanics Pretraining):** Train on `<M><S>secret<G>guess<F>feedback<E>`.
- **Stage 2 (Expert SFT):** Fine-tune that checkpoint on expert moves (`<P>...`).

Mechanics pretraining worked brilliantly. Validation loss dropped to **0.000515** by epoch 9. The Transformer had learned the exact simulator internally.

Then I ran Stage 2 fine-tuning on expert moves, and evaluated the resulting policy checkpoint on the mechanics test set:

```text
Mechanics validation loss (after Stage 1):        0.000515
Mechanics validation loss (after Stage 2 SFT):   10.392833  (>20,000x explosion)
```

Policy training had completely overwritten the attention heads and MLP layers that computed letter-matching and duplicate-letter counts. Across three different random seeds, mechanics loss consistently exploded to 10.39, 12.53, and 15.10.

Lowering the fine-tuning learning rate by an order of magnitude (from $3\times 10^{-4}$ down to $3\times 10^{-5}$) did not fix it; mechanics loss still stalled above 9.49.

### The Fix: Multi-Task Experience Replay

Instead of sequential stages, I introduced an interleaved **multi-task experience replay buffer**. During policy training, mechanics batches were drawn from a separate stream and mixed evenly into optimizer steps:

| Run Configuration | Expert / Mechanics Replay Ratio | Expert Val Loss | Mechanics Val Loss | Held-Out Wins (72 Secrets) |
| :--- | :---: | :---: | :---: | :---: |
| **No Replay (Sequential)** | 100% / 0% | 0.4112 | 10.3928 | 71 / 72 |
| **5% Replay** | 95% / 5% | 0.4131 | **0.0073** | **72 / 72 (100%)** |
| **10% Replay** | 90% / 10% | **0.4096** | **0.0030** | 71 / 72 |

A tiny **5% mechanics replay stream** reduced mechanics error by three orders of magnitude, preserved rule comprehension, and achieved a perfect 72/72 score on the test set.

---

## 3. Scaling State Coverage: 100K to 1M States

A 72-secret test set is noisy: one lucky guess shifts the win rate by 1.39%. To get clean ground-truth numbers, I built a **5-fold cross-validation benchmark** covering all **719 secrets** across three random seeds (2,157 held-out games per benchmark). Any state originating from a held-out test secret was purged prior to training.

I then generated off-policy training trajectories using diverse policies (random moves, greedy consistent moves, deliberately bad moves), but relabeled every single state with the optimal action from an information-theoretic minimax solver.

For the 500K and 1M pools, I added **logical novelty sampling**: deduplicating states based on their remaining-answer candidate sets:

| Dataset Size | Unique Remaining-Answer Sets | Raw Win Rate (719 Secrets) | Raw Invalid Guesses | Avg Attempts / Game |
| :--- | :---: | :---: | :---: | :---: |
| **100K States** | 37,992 | 34.59% ± 1.46% | 198.7 ± 21.5 | 4.5642 |
| **200K States** | 65,779 | 52.48% ± 2.71% | 147.3 ± 8.5 | 4.2684 |
| **500K States** | 167,621 | 66.25% ± 2.69% | 95.0 ± 15.7 | 4.0505 |
| **1M States** | 303,212 | **73.76% ± 2.66%** | **67.0 ± 11.8** | **3.8948** |

Data scaling followed a clear logarithmic curve:
- $100\text{K} \to 200\text{K}$: **+17.89%** win rate
- $200\text{K} \to 500\text{K}$: **+13.77%** win rate
- $500\text{K} \to 1\text{M}$: **+7.51%** win rate

---

## 4. Lexical vs. Strategic Failures: Constrained Decoding

When the model lost a game, why did it lose? Did it hallucinate an illegal 5-letter string, or did it pick valid words that failed to narrow down the answer in 6 turns?

To isolate these errors, I built **prefix-trie constrained decoding**: dynamically masking the model's logits at generation time so it could only output letters that formed valid words in `words.txt`.

| Dataset Size | Raw Win Rate | Constrained Win Rate | Trie Gap | Raw Invalid Words | Constrained Invalid |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **200K States** | 52.48% | 57.77% | **+5.29%** | 147.3 | **0.0** |
| **500K States** | 66.25% | 71.02% | **+4.77%** | 95.0 | **0.0** |
| **1M States** | 73.76% | **77.05%** | **+3.29%** | 67.0 | **0.0** |

Two critical lessons emerged:
1. **The model was learning the English lexicon:** Raw invalid guesses dropped from 147.3 down to 67.0, and the gap between raw and constrained play shrank from 5.29% to 3.29%.
2. **Most remaining errors were strategic:** At 1M states, even with zero invalid guesses, the model still lost ~165 games per seed. More than **85% of remaining losses were purely strategic deductions running out of turns**.

---

## 5. Model Scaling & The 80% SFT Ceiling

With data returns slowing down, I held the 1,000,000-state corpus fixed and scaled the Transformer's parameters across the 5-fold benchmark:

| Model Architecture | Parameters | Raw Win Rate | Constrained Win Rate | Raw Invalid Words |
| :--- | :---: | :---: | :---: | :---: |
| **3.2M Model** | 3,202,083 | 72.55% ± 1.46% | 76.59% ± 2.23% | 88.3 |
| **7.2M Model** | 7,162,403 | 75.75% ± 1.39% | **80.44% ± 0.63%** | 81.7 |
| **12.7M Model** | 12,695,587 | **76.59% ± 1.26%** | **80.39% ± 1.50%** | **61.7** |

Scaling from 3.2M to 7.2M parameters produced a reliable leap: **+3.85% constrained win rate**, crossing the 80% milestone ($578.3$ wins / 719) with very tight variance across seeds ($\text{SD} = 0.63\%$).

However, scaling further to 12.7M parameters hit a ceiling:
- Constrained win rate remained flat at $80.39\%$.
- In head-to-head games, 12.7M won 34 games that 7.2M lost, but lost 34.3 games that 7.2M won ($\Delta = -0.3$ net wins).
- While 12.7M cleaned up raw spelling errors (down to 61.7), its strategic decision-making hit an asymptote.

Supervised imitation had hit its limit.

---

## 6. The DPO Trap & The Anchored SFT Rescue

In supervised fine-tuning, the solver's top-1 guess is ground truth ($P=1$) and all other words are treated as equally wrong ($P=0$). But Wordle isn't binary: a 2nd-best move might partition 90% of words, while a 10th-best move partitions 10%.

I extracted $439,483$ preference pairs using the minimax solver's expected survivor scores:
- **Clear pairs:** Rank 1/2 vs. Rank 5–8 (expected survivor ratio $\ge 1.25$)
- **Hard pairs:** Rank 1 vs. Rank 2

### The Standard DPO Disaster

I applied standard DPO to the 7.2M SFT model ($\text{lr} = 1\times 10^{-5}$, $\beta = 0.20$).

On paper, the training metrics looked incredible: DPO validation loss dropped from 0.69 to 0.32, and preference accuracy jumped from 75.6% to 87.0%.

Then I tested the checkpoint on Wordle gameplay:

```text
Pass 0 (SFT Baseline): Constrained Wins = 59/72 | Raw Wins = 57/72 | Invalid =  5
Pass 1 (DPO):          Constrained Wins = 25/72 | Raw Wins =  0/72 | Invalid = 72 (CRASH)
Pass 3 (DPO):          Constrained Wins = 12/72 | Raw Wins =  0/72 | Invalid = 72 (COLLAPSE)
```

Actual gameplay had collapsed completely. Raw win rate was **0%**. Every single guess generated by the model was an invalid string of gibberish.

### The Autopsy: The Negative Log-Probability Sinkhole

Why did standard DPO destroy the model? 

DPO optimizes the relative margin between chosen ($y_w$) and rejected ($y_l$) sequences:

$$\log \sigma \left( \beta \left[ \left(\log \frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)}\right) - \left(\log \frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}\right) \right] \right)$$

The optimizer can maximize this margin without increasing the probability of $y_w$. It can simply drive $\log \pi_\theta(y_l)$ toward $-\infty$ while **simultaneously driving down $\log \pi_\theta(y_w)$**, as long as $y_l$ falls faster.

Tracking the log probabilities exposed this exact failure:

| Metric | Pass 0 (SFT) | Pass 1 | Pass 2 | Pass 3 |
| :--- | :---: | :---: | :---: | :---: |
| **Policy Chosen $\log \pi(y_w)$** | **-3.71** | -18.91 | -24.20 | **-28.89** |
| **Policy Rejected $\log \pi(y_l)$** | -7.51 | -43.99 | -58.88 | -70.97 |
| **Reference Deviation** | 0.00 | -25.84 | -35.93 | **-44.32** |
| **Raw Invalid Guesses** | 5 | **72** | **72** | **72** |

The model pushed probability mass completely off the English language manifold into garbage tokens.

### The Rescue: Anchored SFT-DPO

To anchor the policy to valid language, I added an **SFT auxiliary loss on the chosen tokens** and lowered the learning rate to $1\times 10^{-6}$:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) + \lambda_{\text{SFT}} \cdot \mathcal{L}_{\text{NLL}}(y_w)$$

With $\lambda_{\text{SFT}} = 1.0$, `Chosen Δlogp` stayed positive ($+0.7530$), preference accuracy rose to 80.3%, and the policy remained stable.

Benchmarking this Anchored DPO model against the 7.2M SFT baseline across the full 5-fold CV suite broke through the ceiling:

| Metric | 7.2M SFT Baseline | 7.2M Anchored DPO | Delta |
| :--- | :---: | :---: | :---: |
| **Constrained Win Rate** | 80.44% ± 0.63% | **81.46% ± 1.49%** | **+1.02%** |
| **Constrained Wins / 719** | 578.3 | **585.7** | **+7.4 net wins / seed** |
| **Raw Win Rate** | 75.75% ± 1.39% | **77.33% ± 1.84%** | **+1.58%** |
| **Raw Invalid Guesses** | 81.7 | **69.7** | **-12.0 errors** |
| **Mean Action Regret** | 0.02918 | **0.02633** | **-9.8% regret** |
| **Rank 1 Action Match** | 65.46% | **65.90%** | **+0.44%** |

In paired head-to-head games, Anchored DPO won $17.7$ games per seed that SFT lost, while losing only $10.3$ games that SFT won ($+7.4$ net wins). On Seed 0, it reached **83.17% constrained win rate** (598 / 719).

---

## 7. What the Model Learned (and What It Couldn't)

### What Worked
- **Opening theory:** The model consistently opens with `irate`—an information-theoretic powerhouse covering high-value vowels and common consonants.
- **Consonant elimination:** On turn 2, the model frequently plays sacrifice words (`clump`, `bendy`) to partition consonants when the initial hint is ambiguous.
- **Lexical memory:** Over 96% of generated guesses in raw mode are real words from the 719-word vocabulary without using any external dictionary.

### The Remaining Blindspot: The Anagram Trap
The model's most persistent failure is what I call the **rhyming trap**. When faced with a four-green pattern like `_ight`, the model often plays "hard mode": guessing `light`, then `might`, then `night`, then `tight`. When five candidates remain and only two turns are left, this guarantees a loss.

A human player (or an optimal solver) intentionally burns a turn on an unrelated word containing those consonants (`lemon`, `melon`, `clump`) to identify the exact letter in a single move. Because SFT and DPO evaluate actions against expected survivor proxies rather than full trajectory outcomes, the model struggles to realize that burning a turn now saves the game later.

---

## 8. What's Next: GRPO on the 7.2M Checkpoint

The limitation of both SFT and DPO here is that they rely on a proxy: the classical solver's expected survivor count. But Wordle is won or lost on **actual game outcomes**.

My next step is to apply **Group Relative Policy Optimization (GRPO)** directly to the 7.2M SFT checkpoint:

1. For a given board state $s_t$, sample a group of $G$ candidate guesses $\{g_1, g_2, \dots, g_G\}$ from the model.
2. Roll each branch forward to terminal game states using the internal mechanics model or environment.
3. Compute trajectory rewards:
   - $+1.0$ for winning in $\le 6$ guesses
   - Turn-budget bonus: $(6 - \text{attempts}) \times 0.1$
   - Penalty for invalid guesses or timeouts
4. Compute relative advantages within the group ($A_i = \frac{R_i - \mu}{\sigma}$) and update the policy via clipped PPO objectives without requiring a separate critic network.

By rolling the game forward to actual wins and losses, GRPO allows the model to discover that sacrificing a turn to break an anagram trap produces a higher expected win rate than greedily hoping for a 1-in-5 lucky hit.

I'll be logging the GRPO training runs and comparative benchmarks in the next post.
