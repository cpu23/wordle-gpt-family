"""Differentiable five-letter scoring without replaying history per candidate.

This reuses the existing WordleGPT parameters, not a separate model/cache state.
Attention dropout must be zero (the checkpoint's existing configuration). Prefix
attention uses SDPA; suffix attention shares prefix K/V without expanding them
across candidates. Different floating-point reduction orders need not be bitwise
identical to full forward: CPU float64 comparisons use rtol=1e-8, atol=1e-9;
float32 uses rtol=2e-5, atol=2e-5. Mixed precision needs dtype-appropriate tolerances.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

from model import WordleGPT


def _log_softmax(logits: Tensor) -> Tensor:
    # Keep the objective in float32 under autocast, without degrading float64.
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return F.log_softmax(logits, dim=-1)


def _score_length_group(
    model: WordleGPT, prompts: Tensor, candidate_tokens: Tensor
) -> Tensor:
    batch, length = prompts.shape
    candidates = candidate_tokens.shape[1]
    positions = torch.arange(length + 4, device=prompts.device)
    prefix = model.token_embedding(prompts) + model.position_embedding(positions[:length])
    suffix = model.token_embedding(candidate_tokens[..., :4]) + model.position_embedding(
        positions[length:]
    )
    suffix_allowed = torch.ones((4, 4), dtype=torch.bool, device=prompts.device).tril()

    for block in model.blocks:
        attention = block.attention
        heads = attention.num_heads
        width = attention.head_dim
        embedding = attention.embed_dim
        prefix_qkv = F.linear(
            block.attention_norm(prefix),
            attention.in_proj_weight,
            attention.in_proj_bias,
        )
        pq, pk, pv = (
            item.reshape(batch, length, heads, width).transpose(1, 2)
            for item in prefix_qkv.chunk(3, dim=-1)
        )
        prefix_attended = F.scaled_dot_product_attention(pq, pk, pv, is_causal=True)
        prefix_attended = prefix_attended.transpose(1, 2).reshape(batch, length, embedding)
        prefix = prefix + attention.out_proj(prefix_attended)
        prefix = prefix + block.mlp(block.mlp_norm(prefix))

        suffix_qkv = F.linear(
            block.attention_norm(suffix),
            attention.in_proj_weight,
            attention.in_proj_bias,
        )
        sq, sk, sv = (
            item.reshape(batch, candidates, 4, heads, width).permute(0, 3, 1, 2, 4)
            for item in suffix_qkv.chunk(3, dim=-1)
        )
        # Flatten candidate/query axes, never expand history K/V to [B,K,H,L,D].
        # Causality makes the history representations independent of each suffix.
        scale = 1.0 / math.sqrt(width)
        history_scores = torch.matmul(
            sq.flatten(2, 3), pk.transpose(-2, -1)
        ).reshape(batch, heads, candidates, 4, length) * scale
        suffix_scores = torch.matmul(sq, sk.transpose(-2, -1)) * scale
        suffix_scores = suffix_scores.masked_fill(~suffix_allowed, float("-inf"))
        scores = torch.cat((history_scores, suffix_scores), dim=-1)
        probability_dtype = torch.float32 if scores.dtype in (torch.float16, torch.bfloat16) else scores.dtype
        probability = F.softmax(scores, dim=-1, dtype=probability_dtype).to(sq.dtype)
        history_attended = torch.matmul(
            probability[..., :length].flatten(2, 3), pv
        ).reshape(batch, heads, candidates, 4, width)
        suffix_attended = torch.matmul(probability[..., length:], sv)
        attended = (history_attended + suffix_attended).permute(0, 2, 3, 1, 4)
        attended = attended.reshape(batch, candidates, 4, embedding)
        suffix = suffix + attention.out_proj(attended)
        suffix = suffix + block.mlp(block.mlp_norm(suffix))

    first_logps = _log_softmax(model.output(model.norm(prefix[:, -1])))
    first_logps = first_logps.gather(-1, candidate_tokens[..., 0])
    suffix_logps = _log_softmax(model.output(model.norm(suffix)))
    suffix_logps = suffix_logps.gather(-1, candidate_tokens[..., 1:, None]).squeeze(-1)
    return first_logps + suffix_logps.sum(dim=-1)


def candidate_sequence_logps(
    model: WordleGPT,
    prompts: Tensor,
    lengths: Tensor,
    candidate_tokens: Tensor,
) -> Tensor:
    """Return raw-vocabulary log P(five letters | prompt), shape [B,K].

    Prompts are right-padded [B,L]; lengths count their nonpadding tokens. Only
    the five candidate letters are scored, never history, feedback, or EOS.
    Candidates [B,K,5] share one differentiable prefix computation per state.
    Callers can microbatch states (typically 2--8); K need not equal 128.
    """
    if prompts.ndim != 2 or lengths.shape != (prompts.shape[0],):
        raise ValueError("prompts must be [B,L] and lengths must be [B]")
    if (
        candidate_tokens.ndim != 3
        or candidate_tokens.shape[0] != prompts.shape[0]
        or candidate_tokens.shape[2] != 5
        or candidate_tokens.shape[1] == 0
        or prompts.shape[0] == 0
    ):
        raise ValueError("candidate_tokens must be nonempty [B,K,5]")
    for block in model.blocks:
        attention = block.attention
        if (
            attention.dropout != 0
            or not attention._qkv_same_embed_dim
            or attention.bias_k is not None
            or attention.bias_v is not None
            or attention.add_zero_attn
        ):
            raise ValueError("prefix reuse requires existing WordleGPT dropout=0 self-attention")

    # A single small CPU transfer avoids a GPU synchronization for each length.
    length_values = lengths.detach().cpu().tolist()
    groups: dict[int, list[int]] = {}
    for index, length in enumerate(length_values):
        if not isinstance(length, int) or length < 1 or length > prompts.shape[1]:
            raise ValueError("prompt lengths must be positive integers within the padded width")
        if length + 4 > model.context_length:
            raise ValueError("prompt plus four input letters exceeds context length")
        groups.setdefault(length, []).append(index)
    scores = []
    order = []
    for length, indices in groups.items():
        selected = torch.tensor(indices, dtype=torch.long, device=prompts.device)
        scores.append(
            _score_length_group(
                model,
                prompts.index_select(0, selected)[:, :length],
                candidate_tokens.index_select(0, selected),
            )
        )
        order.extend(indices)
    inverse = torch.argsort(torch.tensor(order, dtype=torch.long, device=prompts.device))
    return torch.cat(scores, dim=0).index_select(0, inverse)


def distillation_loss(
    sequence_logps: Tensor, teacher_probs: Tensor, ranks: Tensor
) -> tuple[Tensor, dict[str, Tensor]]:
    """Full teacher cross-entropy over the same candidate-normalized policy.

    Ranks are exhaustive solver ranks, not ranks within the sampled candidates.
    Returned metrics are detached scalar batch means; only loss retains a graph.
    """
    if (
        sequence_logps.ndim != 2
        or 0 in sequence_logps.shape
        or teacher_probs.shape != sequence_logps.shape
        or ranks.shape != sequence_logps.shape
    ):
        raise ValueError("sequence_logps, teacher_probs and ranks must share nonempty [B,K]")
    student_logps = _log_softmax(sequence_logps)
    targets = teacher_probs.detach().to(dtype=student_logps.dtype)
    cross_entropy = -(targets * student_logps).sum(dim=-1).mean()
    with torch.no_grad():
        student_probs = student_logps.exp()
        teacher_entropy = -torch.special.xlogy(targets, targets).sum(dim=-1).mean()
        metrics = {
            "cross_entropy": cross_entropy.detach(),
            "teacher_student_kl": cross_entropy.detach() - teacher_entropy,
            "teacher_entropy": teacher_entropy,
            "student_entropy": -(student_probs * student_logps).sum(dim=-1).mean(),
            "rank1_probability": (student_probs * (ranks == 1)).sum(dim=-1).mean(),
            "top3_probability": (student_probs * (ranks <= 3)).sum(dim=-1).mean(),
            "top8_probability": (student_probs * (ranks <= 8)).sum(dim=-1).mean(),
        }
    return cross_entropy, metrics
