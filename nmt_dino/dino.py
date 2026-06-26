"""Shared DINO components used by diagnostics and training experiments."""

from __future__ import annotations

import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOProjectionHead(nn.Module):
    """Projection head used for token-wise DINO self-distillation."""

    def __init__(self, d_model: int = 1536, hidden: int = 3072, output: int = 256):
        super().__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(hidden)
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(hidden, output, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear1(x)
        x = self.act(x)
        x = self.ln(x)
        x = F.normalize(x, dim=-1)
        return self.last_layer(x)


def byte_span_corruption(
    input_ids: torch.Tensor,
    noise_density: float = 0.15,
    mean_span_len: int = 3,
    sentinel_start: int = 259,
    eos_token_id: int = 1,
    pad_token_id: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """T5-style byte span corruption used by the diagnostic script."""
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    all_corrupted, all_targets = [], []

    for b in range(batch_size):
        tokens = input_ids[b].tolist()
        content_end = seq_len
        while content_end > 0 and tokens[content_end - 1] == pad_token_id:
            content_end -= 1
        if content_end > 0 and tokens[content_end - 1] == eos_token_id:
            content_end -= 1
        content = tokens[:content_end]
        content_len = len(content)

        if content_len < 4:
            all_corrupted.append(content + [eos_token_id])
            all_targets.append([sentinel_start, eos_token_id])
            continue

        num_noise_tokens = max(1, round(content_len * noise_density))
        num_spans = max(1, round(num_noise_tokens / mean_span_len))
        noise_mask = [False] * content_len
        spans_placed, attempts = 0, 0

        while spans_placed < num_spans and attempts < 100:
            span_len = max(1, int(np.random.geometric(1.0 / mean_span_len)))
            span_len = min(span_len, content_len - 1)
            start = random.randint(0, content_len - span_len)
            overlap = any(
                noise_mask[i]
                for i in range(max(0, start - 1), min(content_len, start + span_len + 1))
            )
            if not overlap:
                for i in range(start, start + span_len):
                    noise_mask[i] = True
                spans_placed += 1
            attempts += 1

        if spans_placed == 0:
            noise_mask[random.randint(0, content_len - 1)] = True

        corrupted, target = [], []
        sentinel_id = sentinel_start
        in_span = False
        for i, token in enumerate(content):
            if noise_mask[i]:
                if not in_span:
                    corrupted.append(sentinel_id)
                    target.append(sentinel_id)
                    sentinel_id += 1
                    in_span = True
                target.append(token)
            else:
                in_span = False
                corrupted.append(token)
        corrupted.append(eos_token_id)
        target.append(eos_token_id)
        all_corrupted.append(corrupted)
        all_targets.append(target)

    max_corrupted = max(len(c) for c in all_corrupted)
    max_target = max(len(t) for t in all_targets)
    corrupted_batch = torch.full(
        (batch_size, max_corrupted), pad_token_id, dtype=torch.long, device=device
    )
    target_batch = torch.full(
        (batch_size, max_target), pad_token_id, dtype=torch.long, device=device
    )
    for b in range(batch_size):
        corrupted_batch[b, : len(all_corrupted[b])] = torch.tensor(
            all_corrupted[b], dtype=torch.long, device=device
        )
        target_batch[b, : len(all_targets[b])] = torch.tensor(
            all_targets[b], dtype=torch.long, device=device
        )
    return corrupted_batch, target_batch
