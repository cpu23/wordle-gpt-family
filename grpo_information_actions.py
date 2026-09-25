"""Unique mixed proposals for one-guess information-reward GRPO.

This is an equal-candidate proposal-mixture ranking surrogate, not unbiased
on-policy PPO. Exclusion changes proposal sampling only; all stored likelihoods
and update masks use the original dictionary-constrained policy.
"""
from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from torch import Tensor
from torch.nn import functional as F

from grpo_rollouts import LegalWordDecoder, WORD_LENGTH
from tokenizer_v2 import VOCABULARY_SIZE, decode, encode

if TYPE_CHECKING:
    from grpo_information_states import InformationState

POLICY_COUNT = 48
RANDOM_COUNT = 16
GROUP_SIZE = POLICY_COUNT + RANDOM_COUNT


def _proposal_stats(per_group: Sequence[dict[str, int]]) -> dict:
    return {
        "group_count": len(per_group),
        "initial_policy_draws": sum(row["initial_policy_draws"] for row in per_group),
        "initial_duplicates": sum(row["initial_duplicates"] for row in per_group),
        "refill_rounds": max((row["refill_rounds"] for row in per_group), default=0),
        "refill_policy_draws": sum(row["refill_draws"] for row in per_group),
        "policy_proposals": sum(row["policy_proposals"] for row in per_group),
        "random_proposals": sum(row["random_proposals"] for row in per_group),
        "per_group": list(per_group),
    }


@dataclass
class GuessBatch:
    """State-major groups; exactly five generated letters per candidate."""

    inputs: Tensor
    positions: Tensor
    actions: Tensor
    masks: Tensor
    valid: Tensor
    old_token_logps: Tensor
    rewards: Tensor
    guesses: tuple[tuple[str, ...], ...]
    proposal_sources: tuple[tuple[str, ...], ...]
    proposal_stats: dict

    def slice_groups(self, start: int, end: int) -> GuessBatch:
        """Return tensor views over complete groups for bounded gradient batches."""
        if not 0 <= start <= end <= len(self.guesses):
            raise ValueError("group slice must satisfy 0 <= start <= end <= group count")
        group_size = self.rewards.shape[1]
        rows = slice(start * group_size, end * group_size)
        return GuessBatch(
            inputs=self.inputs[rows],
            positions=self.positions[rows],
            actions=self.actions[rows],
            masks=self.masks[rows],
            valid=self.valid[rows],
            old_token_logps=self.old_token_logps[rows],
            rewards=self.rewards[start:end],
            guesses=self.guesses[start:end],
            proposal_sources=self.proposal_sources[start:end],
            proposal_stats=_proposal_stats(self.proposal_stats["per_group"][start:end]),
        )


def _policy_draws(
    model,
    prompts: Tensor,
    lengths: Tensor,
    group_ids: Sequence[int],
    decoder: LegalWordDecoder,
    generator: torch.Generator | None,
    forward_batch_size: int,
    remaining_counts: Tensor | None = None,
    greedy: bool = False,
    logit_cache: dict[tuple[int, int], Tensor] | None = None,
) -> list[str]:
    """Draw independently within a round, optionally excluding accepted words."""
    sampled: list[str] = []
    # The model is fixed for this call. A trie node identifies the entire guess
    # prefix; repeated draws/refills need not rerun the transformer on that context.
    if logit_cache is None:
        logit_cache = {}
    for start in range(0, len(group_ids), forward_batch_size):
        groups = torch.tensor(
            group_ids[start : start + forward_batch_size],
            dtype=torch.long, device=decoder.device,
        )
        inputs = prompts[groups]
        local_lengths = lengths[groups]
        max_length = int(local_lengths.max())
        rows = torch.arange(len(groups), device=decoder.device)
        nodes = torch.zeros_like(groups)
        letters = []
        for offset in range(WORD_LENGTH):
            keys = list(zip(group_ids[start:start + len(groups)], nodes.tolist()))
            missing_keys, missing_rows = [], []
            seen = set()
            for row, key in enumerate(keys):
                if key not in logit_cache and key not in seen:
                    seen.add(key)
                    missing_keys.append(key)
                    missing_rows.append(row)
            if missing_rows:
                indices = torch.tensor(missing_rows, device=decoder.device)
                missing_logits = model(inputs[indices, :max_length + offset])[
                    torch.arange(len(indices), device=decoder.device),
                    local_lengths[indices] + offset - 1,
                ]
                logit_cache.update(zip(missing_keys, missing_logits.unbind(0)))
            logits = torch.stack([logit_cache[key] for key in keys])
            if logits.shape[-1] != VOCABULARY_SIZE:
                raise ValueError("model vocabulary must match the v2 tokenizer")
            legal = decoder.legal_mask(nodes)
            if remaining_counts is not None:
                children = decoder.child_nodes[nodes]
                legal = legal & (remaining_counts[groups[:, None], children.clamp_min(0)] > 0)
            constrained = logits[:, :26].masked_fill(~legal, -torch.inf)
            if greedy:
                token = constrained.argmax(dim=-1)
            else:
                token = torch.multinomial(
                    F.softmax(constrained, dim=-1), 1, generator=generator
                ).squeeze(1)
            inputs[rows, local_lengths + offset] = token
            nodes = decoder.advance(nodes, token)
            letters.append(token)
        sampled.extend(decode(row) for row in torch.stack(letters, dim=1).tolist())
    return sampled


@lru_cache(maxsize=4)
def _proposal_geometry(decoder):
    word_ids = {word: index for index, word in enumerate(decoder.words)}
    word_tokens = torch.tensor([encode(word) for word in decoder.words], device=decoder.device)
    nodes = torch.zeros(len(decoder.words), dtype=torch.long, device=decoder.device)
    path_columns = []
    for offset in range(WORD_LENGTH):
        nodes = decoder.advance(nodes, word_tokens[:, offset])
        path_columns.append(nodes)
    word_paths = torch.stack(path_columns, dim=1)
    counts = torch.bincount(word_paths.flatten(), minlength=len(decoder.child_nodes))
    return word_ids, word_tokens, word_paths, counts


@torch.no_grad()
def sample_information_actions(
    model,
    states: Sequence[InformationState],
    decoder: LegalWordDecoder,
    rng: random.Random,
    forward_batch_size: int = 256,
) -> GuessBatch:
    """Draw 48 unique policy-origin and 16 unique uniform-random legal guesses.

    The initial 48 policy draws are independent. After deduplication, random
    words are selected without replacement from the rest of the full dictionary.
    Each refill round draws all missing policy slots with accepted words excluded
    at trie branches. Thus every active group gains at least one new word per
    round, even when all draws choose the same overwhelmingly likely word.
    """
    if len(decoder.words) < GROUP_SIZE:
        raise ValueError("64 distinct legal words are required")
    prompts, lengths = _prepare_prompts(model, states, decoder, forward_batch_size)
    device = decoder.device
    group_count = len(states)
    generator = torch.Generator(device=device).manual_seed(rng.getrandbits(63))
    logit_cache = {}
    initial = _policy_draws(
        model, prompts, lengths,
        [group for group in range(group_count) for _ in range(POLICY_COUNT)],
        decoder, generator, forward_batch_size, logit_cache=logit_cache,
    )
    policy_words = [
        list(dict.fromkeys(initial[group * POLICY_COUNT : (group + 1) * POLICY_COUNT]))
        for group in range(group_count)
    ]
    accepted = [set(words) for words in policy_words]
    random_words = []
    for group in range(group_count):
        selected = rng.sample(
            [word for word in decoder.words if word not in accepted[group]], RANDOM_COUNT
        )
        random_words.append(selected)
        accepted[group].update(selected)
    per_group = [
        {"initial_duplicates": POLICY_COUNT - len(words), "refill_rounds": 0, "refill_draws": 0}
        for words in policy_words
    ]

    # Count the remaining dictionary leaves beneath every trie node. Decrement
    # only accepted paths; no per-proposal dictionary scans or rejection loops.
    word_ids, word_tokens, word_paths, counts = _proposal_geometry(decoder)
    remaining_counts = counts[None, :].expand(group_count, -1).clone()

    def exclude(entries: Sequence[tuple[int, str]]) -> None:
        if not entries:
            return
        groups = torch.tensor([group for group, _ in entries], device=device)
        indices = torch.tensor([word_ids[word] for _, word in entries], device=device)
        paths = word_paths[indices].flatten()
        remaining_counts.index_put_(
            (groups.repeat_interleave(WORD_LENGTH), paths),
            -torch.ones_like(paths), accumulate=True,
        )

    exclude([(group, word) for group, words in enumerate(accepted) for word in sorted(words)])
    while any(len(words) < POLICY_COUNT for words in policy_words):
        refill_groups = []
        for group, words in enumerate(policy_words):
            missing = POLICY_COUNT - len(words)
            if missing:
                per_group[group]["refill_rounds"] += 1
                per_group[group]["refill_draws"] += missing
                refill_groups.extend([group] * missing)
        refills = _policy_draws(
            model, prompts, lengths, refill_groups, decoder, generator,
            forward_batch_size, remaining_counts, logit_cache=logit_cache,
        )
        newly_accepted = []
        for group, word in zip(refill_groups, refills):
            if word not in accepted[group]:
                accepted[group].add(word)
                policy_words[group].append(word)
                newly_accepted.append((group, word))
        exclude(newly_accepted)

    guesses = tuple(tuple(policy + random) for policy, random in zip(policy_words, random_words))
    sources = (("policy",) * POLICY_COUNT + ("random",) * RANDOM_COUNT,) * group_count
    for row in per_group:
        row.update(initial_policy_draws=POLICY_COUNT, policy_proposals=POLICY_COUNT,
                   random_proposals=RANDOM_COUNT)
    actions = word_tokens[torch.tensor(
        [word_ids[word] for group in guesses for word in group], device=device,
    )]
    return _assemble_guess_batch(
        model, prompts, lengths, decoder, guesses, sources, actions,
        _proposal_stats(per_group), forward_batch_size,
    )


def _prepare_prompts(model, states, decoder, forward_batch_size):
    if not states:
        raise ValueError("at least one information state is required")
    if forward_batch_size < 1:
        raise ValueError("forward_batch_size must be positive")
    try:
        model_device = next(model.parameters()).device
    except (AttributeError, StopIteration):
        model_device = decoder.device
    if torch.device(model_device) != decoder.device:
        raise ValueError("model and legal-word decoder must use the same device")
    if any(not state.prompt for state in states):
        raise ValueError("every state must contain a policy prompt")
    lengths = torch.tensor([len(state.prompt) for state in states], device=decoder.device)
    width = max(len(state.prompt) for state in states) + WORD_LENGTH
    prompts = torch.zeros((len(states), width), dtype=torch.long, device=decoder.device)
    for group, state in enumerate(states):
        prompts[group, : len(state.prompt)] = torch.tensor(state.prompt, device=decoder.device)
    return prompts, lengths


@torch.no_grad()
def sample_policy_guesses(
    model,
    states: Sequence[InformationState],
    decoder: LegalWordDecoder,
    samples: int = 8,
    greedy: bool = False,
    forward_batch_size: int = 256,
) -> tuple[tuple[str, ...], ...]:
    """Sample the original legal policy with replacement for fixed-panel evaluation."""
    if samples < 1:
        raise ValueError("samples must be positive")
    prompts, lengths = _prepare_prompts(model, states, decoder, forward_batch_size)
    draws = _policy_draws(
        model, prompts, lengths,
        [group for group in range(len(states)) for _ in range(samples)],
        decoder, None, forward_batch_size, greedy=greedy,
    )
    return tuple(
        tuple(draws[start : start + samples]) for start in range(0, len(draws), samples)
    )


@torch.no_grad()
def build_guess_batch(
    model,
    states: Sequence[InformationState],
    decoder: LegalWordDecoder,
    guesses: Sequence[Sequence[str]],
    proposal_sources: Sequence[Sequence[str]] | None = None,
    forward_batch_size: int = 256,
) -> GuessBatch:
    """Teacher-force equally sized groups under the original legal policy."""
    prompts, lengths = _prepare_prompts(model, states, decoder, forward_batch_size)
    guesses = tuple(tuple(group) for group in guesses)
    if len(guesses) != len(states) or not guesses[0]:
        raise ValueError("guesses must contain one nonempty group per state")
    group_size = len(guesses[0])
    if any(len(group) != group_size for group in guesses):
        raise ValueError("guess groups must have the same size")
    allowed = set(decoder.words)
    if any(word not in allowed for group in guesses for word in group):
        raise ValueError("every guess must be a legal dictionary word")
    if proposal_sources is None:
        sources = (("policy",) * group_size,) * len(states)
    else:
        sources = tuple(tuple(group) for group in proposal_sources)
        if len(sources) != len(states) or any(len(group) != group_size for group in sources):
            raise ValueError("proposal sources must match guess groups")
        if any(source not in ("policy", "random") for group in sources for source in group):
            raise ValueError("proposal sources must be policy or random")
    stats = _proposal_stats([
        {
            "initial_policy_draws": group.count("policy"),
            "initial_duplicates": 0,
            "refill_rounds": 0,
            "refill_draws": 0,
            "policy_proposals": group.count("policy"),
            "random_proposals": group.count("random"),
        }
        for group in sources
    ])
    actions = torch.tensor(
        [encode(word) for group in guesses for word in group], device=decoder.device,
    )
    return _assemble_guess_batch(
        model, prompts, lengths, decoder, guesses, sources, actions, stats, forward_batch_size,
    )


def _assemble_guess_batch(
    model, prompts, lengths, decoder, guesses, sources, actions, stats, forward_batch_size,
) -> GuessBatch:
    group_count = len(guesses)
    group_size = len(guesses[0])
    device = decoder.device
    inputs = prompts.repeat_interleave(group_size, dim=0)
    positions = (
        lengths[:, None] + torch.arange(WORD_LENGTH, device=device) - 1
    ).repeat_interleave(group_size, dim=0)
    inputs.scatter_(1, positions + 1, actions)
    row_count = group_count * group_size
    masks = torch.zeros(
        (row_count, WORD_LENGTH, VOCABULARY_SIZE), dtype=torch.bool, device=device,
    )
    nodes = torch.zeros(row_count, dtype=torch.long, device=device)
    for offset in range(WORD_LENGTH):
        masks[:, offset, :26] = decoder.legal_mask(nodes)
        nodes = decoder.advance(nodes, actions[:, offset])
    old_token_logps = torch.empty((row_count, WORD_LENGTH), device=device)
    for start in range(0, row_count, forward_batch_size):
        stop = min(start + forward_batch_size, row_count)
        logits = model(inputs[start:stop])
        if logits.shape[-1] != VOCABULARY_SIZE:
            raise ValueError("model vocabulary must match the v2 tokenizer")
        selected = logits.gather(
            1, positions[start:stop, :, None].expand(-1, -1, logits.shape[-1])
        )
        logps = F.log_softmax(selected.masked_fill(~masks[start:stop], -torch.inf), dim=-1)
        old_token_logps[start:stop] = logps.gather(
            -1, actions[start:stop, :, None]
        ).squeeze(-1)
    return GuessBatch(
        inputs=inputs,
        positions=positions,
        actions=actions,
        masks=masks,
        valid=torch.ones_like(actions, dtype=torch.bool),
        old_token_logps=old_token_logps,
        rewards=torch.zeros((group_count, group_size), device=device),
        guesses=guesses,
        proposal_sources=sources,
        proposal_stats=stats,
    )
