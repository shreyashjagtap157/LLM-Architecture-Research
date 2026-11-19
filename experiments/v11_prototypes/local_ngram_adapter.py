#!/usr/bin/env python3
"""Experimental causal hashed n-gram residual adapter prototype.

This is a new, deliberately small research prototype. It is not recovered V11
source, an Engram implementation, a pretrained component, or evidence of a
novel architecture. It exists to make the local-memory hypothesis testable
with synthetic tensors before any dataset-backed training is considered.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


_HASH_MODULUS = 2_147_483_647
_HASH_BASE = 1_000_003


def causal_ngram_indices(
    token_ids: Tensor,
    *,
    vocab_size: int,
    table_size: int,
    orders: Sequence[int] = (2, 3),
) -> Tensor:
    """Return per-position hashed n-gram buckets with strict prefix causality.

    Args:
        token_ids: Integer tensor shaped [batch, sequence]. Values must be
            in [0, vocab_size). Position t uses only tokens <= t.
        vocab_size: Size of the token-ID domain. vocab_size itself is used
            internally as a left-padding sentinel, so it is not a legal token.
        table_size: Number of buckets in each order-specific embedding table.
        orders: Unique n-gram orders (each at least 2). Returned order axis
            follows this sequence exactly.

    Returns:
        Long tensor shaped [batch, sequence, len(orders)]. Each value is
        in [0, table_size). Hash collisions are expected and must be
        measured for a real corpus; this function does not imply collision-free
        memory.

    The polynomial hash uses bounded int64 intermediates. The prime modulus
    and base are constants of this prototype, not tuned hyperparameters.
    """
    if token_ids.ndim != 2:
        raise ValueError("token_ids must have shape [batch, sequence]")
    if token_ids.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise TypeError("token_ids must use an integer dtype")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if vocab_size >= _HASH_MODULUS - 1:
        raise ValueError("vocab_size is too large for the prototype hash domain")
    if not isinstance(table_size, int) or isinstance(table_size, bool) or table_size <= 0:
        raise ValueError("table_size must be a positive integer")
    if not isinstance(orders, Sequence) or not orders:
        raise ValueError("orders must be a non-empty sequence")
    normalized_orders = tuple(orders)
    if any(not isinstance(order, int) or isinstance(order, bool) or order < 2 for order in normalized_orders):
        raise ValueError("each n-gram order must be an integer >= 2")
    if len(set(normalized_orders)) != len(normalized_orders):
        raise ValueError("orders must be unique")

    ids = token_ids.to(dtype=torch.int64)
    if ids.numel() and (bool(torch.any(ids < 0)) or bool(torch.any(ids >= vocab_size))):
        raise ValueError("token_ids contain values outside [0, vocab_size)")

    batch_size, sequence_length = ids.shape
    if sequence_length == 0:
        return torch.empty((batch_size, 0, len(normalized_orders)), dtype=torch.int64, device=ids.device)

    buckets: list[Tensor] = []
    for order in normalized_orders:
        # The sentinel makes initial prefixes well-defined without looking
        # ahead. unfold produces one trailing window per current token.
        padded = F.pad(ids, (order - 1, 0), value=vocab_size)
        windows = padded.unfold(dimension=1, size=order, step=1)
        hashed = torch.zeros((batch_size, sequence_length), dtype=torch.int64, device=ids.device)
        for offset in range(order):
            hashed = torch.remainder(hashed * _HASH_BASE + windows[:, :, offset] + 1, _HASH_MODULUS)
        buckets.append(torch.remainder(hashed, table_size))

    if not buckets:
        raise RuntimeError("no n-gram orders were supplied")
    return torch.stack(buckets, dim=-1)


class HashedNGramResidualAdapter(nn.Module):
    """Context-gated n-gram lookup with additive residual injection.

    hidden_states must already be causal representations supplied by the
    caller. The adapter guarantees that its token-ID lookup path is causal;
    it cannot repair future leakage already present in hidden states.
    """

    def __init__(
        self,
        *,
        vocab_size: int,
        d_model: int,
        table_size: int,
        orders: Sequence[int] = (2, 3),
        initial_gate_bias: float = -2.0,
    ) -> None:
        super().__init__()
        if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not math.isfinite(initial_gate_bias):
            raise ValueError("initial_gate_bias must be finite")
        if not isinstance(orders, Sequence) or not orders:
            raise ValueError("orders must be a non-empty sequence")
        normalized_orders = tuple(orders)
        if any(not isinstance(order, int) or isinstance(order, bool) or order < 2 for order in normalized_orders):
            raise ValueError("each n-gram order must be an integer >= 2")
        if len(set(normalized_orders)) != len(normalized_orders):
            raise ValueError("orders must be unique")
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
            raise ValueError("vocab_size must be a positive integer")
        if vocab_size >= _HASH_MODULUS - 1:
            raise ValueError("vocab_size is too large for the prototype hash domain")
        if not isinstance(table_size, int) or isinstance(table_size, bool) or table_size <= 0:
            raise ValueError("table_size must be a positive integer")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.table_size = table_size
        self.orders = normalized_orders

        self.tables = nn.ModuleList(
            nn.Embedding(table_size, d_model) for _ in self.orders
        )
        self.query_norm = nn.LayerNorm(d_model)
        self.key_norm = nn.LayerNorm(d_model)
        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.gate_bias = nn.Parameter(torch.tensor(float(initial_gate_bias)))

        for table in self.tables:
            nn.init.normal_(table.weight, mean=0.0, std=0.02)

    def forward(self, token_ids: Tensor, hidden_states: Tensor) -> Tensor:
        """Inject context-gated local memory; inputs and output are [B,T,D]."""
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, d_model]")
        if hidden_states.shape[-1] != self.d_model:
            raise ValueError(f"hidden_states must have shape [batch, sequence, {self.d_model}]")
        if token_ids.shape != hidden_states.shape[:2]:
            raise ValueError("token_ids shape must match hidden_states [batch, sequence]")
        if token_ids.device != hidden_states.device:
            raise ValueError("token_ids and hidden_states must be on the same device")
        if not hidden_states.is_floating_point():
            raise TypeError("hidden_states must be floating point")

        indices = causal_ngram_indices(
            token_ids,
            vocab_size=self.vocab_size,
            table_size=self.table_size,
            orders=self.orders,
        )
        memory_parts = [table(indices[..., i]) for i, table in enumerate(self.tables)]
        memory = torch.stack(memory_parts, dim=0).sum(dim=0) / math.sqrt(len(memory_parts))

        query = self.query_norm(self.query_projection(hidden_states))
        key = self.key_norm(self.key_projection(memory))
        gate_logits = (query * key).sum(dim=-1, keepdim=True) / math.sqrt(self.d_model)
        gate = torch.sigmoid(gate_logits + self.gate_bias)
        residual = self.value_projection(memory)
        return hidden_states + gate * residual
