"""Full-game GRPO objective and diagnostics for sampled Wordle trajectories."""
from __future__ import annotations

from typing import Any

import torch
from torch import Tensor
from torch.nn import functional as F

from train_grpo import relative_advantages


def _valid(batch: Any) -> Tensor:
    return batch.valid.to(dtype=torch.bool)


def _trajectory_shape(batch: Any) -> tuple[int, int]:
    groups, group_size = batch.rewards.shape
    if batch.inputs.shape[0] != groups * group_size:
        raise ValueError("trajectory rows must be flattened in group-major order")
    return groups, group_size


def trajectory_logits(model: Any, batch: Any) -> Tensor:
    """Return next-token logits at every stored letter prediction position."""
    logits = model(batch.inputs)
    positions = batch.positions.unsqueeze(-1).expand(-1, -1, logits.shape[-1])
    return logits.gather(1, positions)


def _constrained_log_probs(logits: Tensor, batch: Any) -> Tensor:
    """Masked token log-probabilities, with invalid rows made numerically benign."""
    valid = _valid(batch)
    active = valid.unsqueeze(-1)
    # Invalid positions may contain padding/control positions or arbitrary logits.
    # Replace them before log_softmax so even non-finite padding cannot leak NaNs.
    safe_logits = torch.where(active, logits, torch.zeros_like(logits))
    masks = batch.masks.to(dtype=torch.bool) | ~active
    return F.log_softmax(safe_logits.masked_fill(~masks, -torch.inf), dim=-1)


def token_logps(logits: Tensor, batch: Any) -> Tensor:
    """Return constrained sampled-letter log-probabilities; padding is zero."""
    valid = _valid(batch)
    log_probs = _constrained_log_probs(logits, batch)
    actions = torch.where(valid, batch.actions, torch.zeros_like(batch.actions))
    selected = log_probs.gather(-1, actions.unsqueeze(-1)).squeeze(-1)
    return torch.where(valid, selected, torch.zeros_like(selected))


def trajectory_logps(logits: Tensor, batch: Any) -> Tensor:
    """Diagnostic sum of generated-letter log-probabilities, not a PPO action."""
    groups, group_size = _trajectory_shape(batch)
    return token_logps(logits, batch).sum(dim=1).reshape(groups, group_size)


def _rollout_token_means(values: Tensor, valid: Tensor) -> Tensor:
    selected = torch.where(valid, values, torch.zeros_like(values))
    counts = valid.sum(dim=1).to(dtype=values.dtype).clamp_min(1)
    return selected.sum(dim=1) / counts


def _prefix_kls(policy_logits: Tensor, reference_logits: Tensor, batch: Any) -> tuple[Tensor, Tensor]:
    """Exact raw/legal KL, averaged over generated letters within each rollout."""
    valid = _valid(batch)
    active = valid.unsqueeze(-1)
    policy_safe = torch.where(active, policy_logits, torch.zeros_like(policy_logits))
    reference_safe = torch.where(
        active, reference_logits.detach(), torch.zeros_like(reference_logits)
    )

    policy_raw_logp = F.log_softmax(policy_safe, dim=-1)
    reference_raw_logp = F.log_softmax(reference_safe, dim=-1)
    raw_per_position = (
        policy_raw_logp.exp() * (policy_raw_logp - reference_raw_logp)
    ).sum(dim=-1)
    raw_kl = _rollout_token_means(raw_per_position, valid)

    masks = batch.masks.to(dtype=torch.bool) | ~active
    policy_legal_logp = F.log_softmax(
        policy_safe.masked_fill(~masks, -torch.inf), dim=-1
    )
    reference_legal_logp = F.log_softmax(
        reference_safe.masked_fill(~masks, -torch.inf), dim=-1
    )
    # Do not form (-inf) - (-inf) at excluded vocabulary entries.
    policy_finite = policy_legal_logp.masked_fill(~masks, 0)
    reference_finite = reference_legal_logp.masked_fill(~masks, 0)
    legal_per_position = (
        policy_legal_logp.exp() * (policy_finite - reference_finite)
    ).sum(dim=-1)
    constrained_kl = _rollout_token_means(legal_per_position, valid)
    return raw_kl, constrained_kl


def _reward_terms(
    policy_logits: Tensor, batch: Any, clip: float
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return per-rollout loss and token diagnostics with a single shared advantage."""
    _trajectory_shape(batch)
    valid = _valid(batch)
    logps = token_logps(policy_logits, batch)
    old_logps = batch.old_token_logps.to(device=logps.device, dtype=logps.dtype)
    log_ratios = torch.where(valid, logps - old_logps, torch.zeros_like(logps))
    ratios = log_ratios.exp()
    advantages = relative_advantages(
        batch.rewards.to(device=logps.device, dtype=logps.dtype)
    ).reshape(-1)
    unclipped = ratios * advantages[:, None]
    clipped = ratios.clamp(1 - clip, 1 + clip) * advantages[:, None]
    reward_losses = -_rollout_token_means(torch.minimum(unclipped, clipped), valid)
    return reward_losses, ratios, advantages, logps


def trajectory_loss(
    policy_logits: Tensor,
    reference_logits: Tensor,
    batch: Any,
    kl_beta: float = 0.02,
    clip: float = 0.2,
) -> tuple[Tensor, dict[str, float]]:
    """Token-clipped GRPO with equal rollout weight, independent of game length."""
    reward_losses, ratios, advantages, logps = _reward_terms(policy_logits, batch, clip)
    raw_kls, constrained_kls = _prefix_kls(policy_logits, reference_logits, batch)
    raw_kl, constrained_kl = raw_kls.mean(), constrained_kls.mean()
    loss = reward_losses.mean() + kl_beta * raw_kl
    valid = _valid(batch)
    valid_ratios = ratios.detach()[valid]
    clipped = ((ratios.detach() < 1 - clip) | (ratios.detach() > 1 + clip)).to(ratios.dtype)
    stats = {
        "raw_prefix_kl_from_sft": float(raw_kl.detach()),
        "constrained_prefix_kl_from_sft": float(constrained_kl.detach()),
        "mean_importance_ratio": float(_rollout_token_means(ratios.detach(), valid).mean()),
        "token_ratio_min": float(valid_ratios.min()) if valid_ratios.numel() else 1.0,
        "token_ratio_max": float(valid_ratios.max()) if valid_ratios.numel() else 1.0,
        "token_clip_fraction": float(_rollout_token_means(clipped, valid).mean()),
        "mean_trajectory_logp": float(logps.detach().sum(dim=1).mean()),
        "mean_group_advantage": float(advantages.detach().mean()),
        "mean_abs_advantage": float(advantages.detach().abs().mean()),
    }
    return loss, stats


def trajectory_gradient_metrics(
    model: Any,
    policy_logits: Tensor,
    reference_logits: Tensor,
    batch: Any,
    kl_beta: float,
    clip: float = 0.2,
) -> dict[str, Any]:
    """Measure actual gradient contributions to the globally normalized batch loss.

    Bucket losses retain the global rollout denominator rather than re-averaging
    within a bucket. Norms need not add: parameter gradients can cancel. Autograd
    retains the training graph and never writes parameter ``.grad`` buffers.
    """
    reward_losses, _, advantages, _ = _reward_terms(policy_logits, batch, clip)
    raw_kls, _ = _prefix_kls(policy_logits, reference_logits, batch)
    combined_losses = reward_losses + kl_beta * raw_kls
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    count = reward_losses.numel()
    token_counts = _valid(batch).sum(dim=1)
    generated = token_counts // 5
    remaining = 6 - (batch.attempts.reshape(-1).to(generated.device) - generated)
    token_weights = token_counts.to(reward_losses.dtype).clamp_min(1).reciprocal()
    abs_advantages = advantages.detach().abs()

    def gradient_l2(loss: Tensor) -> float:
        if not parameters or not loss.requires_grad:
            return 0.0
        gradients = torch.autograd.grad(
            loss, parameters, retain_graph=True, allow_unused=True
        )
        squared_norm = torch.zeros((), device=loss.device, dtype=torch.float32)
        for gradient in gradients:
            if gradient is not None:
                squared_norm += gradient.detach().float().square().sum()
        return float(squared_norm.sqrt())

    def norms(selected: Tensor) -> tuple[float, float]:
        reward_norm = gradient_l2(reward_losses[selected].sum() / count)
        combined_norm = (
            gradient_l2(combined_losses[selected].sum() / count)
            if kl_beta else reward_norm
        )
        return reward_norm, combined_norm

    # Identical row selections (common for fixed-depth prompts) share backward work.
    cached_norms: dict[tuple[int, ...], tuple[float, float]] = {}

    def buckets(lengths: Tensor) -> dict[str, Any]:
        result = {}
        for length in range(1, 7):
            selected = lengths == length
            indices = tuple(selected.nonzero(as_tuple=True)[0].tolist())
            if indices and indices not in cached_norms:
                cached_norms[indices] = norms(selected)
            reward_norm, combined_norm = cached_norms.get(indices, (0.0, 0.0))
            result[str(length)] = {
                "count": len(indices),
                "mean_abs_advantage": float(abs_advantages[selected].mean()) if indices else None,
                "mean_token_normalization_weight": float(token_weights[selected].mean()) if indices else None,
                "reward_gradient_l2": reward_norm,
                "combined_gradient_l2": combined_norm,
            }
        return result

    all_rows = torch.ones(count, dtype=torch.bool, device=reward_losses.device)
    reward_norm, combined_norm = norms(all_rows)
    cached_norms[tuple(range(count))] = reward_norm, combined_norm
    return {
        "reward_gradient_l2": reward_norm,
        "combined_gradient_l2": combined_norm,
        "by_generated_guesses": buckets(generated),
        "by_remaining_guesses": buckets(remaining),
    }


def rollout_metrics(batch: Any) -> dict[str, int | float | None]:
    """Summarize sampled outcomes using JSON-compatible scalar values."""
    rewards = batch.rewards.detach()
    won = batch.won.detach().to(dtype=torch.bool)
    attempts = batch.attempts.detach()
    group_std = rewards.std(dim=1, correction=0)
    identical = rewards.amax(dim=1) == rewards.amin(dim=1)
    win_count = int(won.sum())
    rollout_count = int(won.numel())
    mean_winning_attempts = (
        float(attempts[won].to(dtype=torch.float32).mean()) if win_count else None
    )
    return {
        "sampled_rollout_win_fraction": float(won.to(dtype=torch.float32).mean()),
        "mean_guesses_for_winning_rollouts": mean_winning_attempts,
        "mean_trajectory_reward": float(rewards.to(dtype=torch.float32).mean()),
        "mean_group_reward_std": float(group_std.to(dtype=torch.float32).mean()),
        "identical_reward_group_rate": float(identical.to(dtype=torch.float32).mean()),
        "winning_rollouts": win_count,
        "sampled_rollouts": rollout_count,
    }


@torch.no_grad()
def trajectory_policy_metrics(
    policy_logits: Tensor, reference_logits: Tensor, batch: Any
) -> dict[str, float]:
    """Report equal-rollout mean raw/legal KL at generated-letter prefixes."""
    raw_kl, constrained_kl = _prefix_kls(policy_logits, reference_logits, batch)
    return {
        "raw_prefix_kl_from_sft": float(raw_kl.mean()),
        "constrained_prefix_kl_from_sft": float(constrained_kl.mean()),
    }
