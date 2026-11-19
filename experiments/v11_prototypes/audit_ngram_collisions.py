#!/usr/bin/env python3
"""Audit hashed n-gram table collisions and storage from token-ID sequences.

The input must be a locally prepared, rights-cleared whitespace-separated
sequence of integer token IDs. The tool never emits token IDs or source text.
It is an exact, O(number of distinct n-grams) in-memory audit intended for
small/medium pilot corpora, not a streaming large-corpus estimator.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

_HASH_MODULUS = 2_147_483_647
_HASH_BASE = 1_000_003


def _hash_complete_ngram(ngram: Sequence[int], table_size: int) -> int:
    value = 0
    for token_id in ngram:
        value = (value * _HASH_BASE + token_id + 1) % _HASH_MODULUS
    return value % table_size


def read_token_ids(path: Path, *, vocab_size: int, max_tokens: int | None = None) -> list[int]:
    """Load and validate whitespace-separated integer token IDs."""
    if not path.is_file():
        raise FileNotFoundError(f"token-ID file not found: {path}")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool) or vocab_size <= 0:
        raise ValueError("vocab_size must be a positive integer")
    if vocab_size >= _HASH_MODULUS - 1:
        raise ValueError("vocab_size is too large for the prototype hash domain")
    if max_tokens is not None and (
        not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens <= 0
    ):
        raise ValueError("max_tokens must be a positive integer when supplied")

    tokens: list[int] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            for field in line.split():
                try:
                    token_id = int(field, 10)
                except ValueError as exc:
                    raise ValueError(f"invalid integer token ID on line {line_number}") from exc
                if token_id < 0 or token_id >= vocab_size:
                    raise ValueError(f"token ID outside [0, vocab_size) on line {line_number}")
                tokens.append(token_id)
                if max_tokens is not None and len(tokens) >= max_tokens:
                    return tokens
    if not tokens:
        raise ValueError("token-ID file contains no tokens")
    return tokens


def audit_ngram_collisions(
    token_ids: Sequence[int],
    *,
    table_size: int,
    orders: Sequence[int] = (2, 3),
    d_model: int,
    dtype_bytes: int = 2,
    optimizer_state_bytes_per_param: int = 4,
) -> dict:
    """Return exact n-gram collision statistics and approximate table storage.

    Only complete n-grams are counted: for order n, the first n-1 positions
    are excluded rather than padded with a synthetic BOS token. Each table is
    order-specific, matching HashedNGramResidualAdapter.
    """
    if not isinstance(table_size, int) or isinstance(table_size, bool) or table_size <= 0:
        raise ValueError("table_size must be a positive integer")
    if not isinstance(d_model, int) or isinstance(d_model, bool) or d_model <= 0:
        raise ValueError("d_model must be a positive integer")
    for name, value in (
        ("dtype_bytes", dtype_bytes),
        ("optimizer_state_bytes_per_param", optimizer_state_bytes_per_param),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    normalized_orders = tuple(orders)
    if not normalized_orders or any(
        not isinstance(order, int) or isinstance(order, bool) or order < 2
        for order in normalized_orders
    ):
        raise ValueError("orders must contain positive integers >= 2")
    if len(set(normalized_orders)) != len(normalized_orders):
        raise ValueError("orders must be unique")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in token_ids):
        raise ValueError("token_ids must contain non-negative integers")

    per_order: list[dict] = []
    for order in normalized_orders:
        token_type_counts: Counter[tuple[int, ...]] = Counter()
        bucket_to_types: dict[int, set[tuple[int, ...]]] = defaultdict(set)
        for end in range(order, len(token_ids) + 1):
            ngram = tuple(token_ids[end - order : end])
            bucket = _hash_complete_ngram(ngram, table_size)
            token_type_counts[ngram] += 1
            bucket_to_types[bucket].add(ngram)

        observations = sum(token_type_counts.values())
        unique_types = len(token_type_counts)
        occupied_buckets = len(bucket_to_types)
        collision_buckets = {
            bucket: types for bucket, types in bucket_to_types.items() if len(types) > 1
        }
        excess_types = sum(len(types) - 1 for types in collision_buckets.values())
        types_in_collision_buckets = sum(len(types) for types in collision_buckets.values())
        occurrences_in_collision_buckets = sum(
            token_type_counts[ngram]
            for types in collision_buckets.values()
            for ngram in types
        )
        per_order.append(
            {
                "order": order,
                "complete_ngram_observations": observations,
                "unique_ngram_types": unique_types,
                "occupied_buckets": occupied_buckets,
                "unique_type_collision_excess": excess_types,
                "unique_type_collision_excess_fraction": (
                    excess_types / unique_types if unique_types else 0.0
                ),
                "unique_types_in_collided_buckets": types_in_collision_buckets,
                "unique_type_collision_bucket_share": (
                    types_in_collision_buckets / unique_types if unique_types else 0.0
                ),
                "occurrences_in_collided_buckets": occurrences_in_collision_buckets,
                "occurrence_collision_exposure": (
                    occurrences_in_collision_buckets / observations if observations else 0.0
                ),
                "max_unique_types_per_bucket": max(
                    (len(types) for types in bucket_to_types.values()), default=0
                ),
                "unique_type_load_factor": unique_types / table_size,
            }
        )

    table_count = len(normalized_orders)
    parameter_count = table_count * table_size * d_model
    weight_bytes = parameter_count * dtype_bytes
    gradient_bytes = weight_bytes
    optimizer_state_bytes = parameter_count * 2 * optimizer_state_bytes_per_param
    return {
        "schema_version": "1.0",
        "method": "exact_hash_bucket_collision_audit_complete_ngrams_only",
        "token_count": len(token_ids),
        "table_size_per_order": table_size,
        "orders": list(normalized_orders),
        "table_count": table_count,
        "d_model": d_model,
        "dtype_bytes": dtype_bytes,
        "optimizer_state_bytes_per_param": optimizer_state_bytes_per_param,
        "embedding_table_parameter_count": parameter_count,
        "estimated_weight_bytes": weight_bytes,
        "estimated_gradient_bytes": gradient_bytes,
        "estimated_adam_moment_bytes": optimizer_state_bytes,
        "estimated_training_storage_bytes": weight_bytes + gradient_bytes + optimizer_state_bytes,
        "memory_estimate_scope": (
            "Embedding tables only; excludes projections, activations, allocator overhead, "
            "temporary hash-audit structures and model/runtime state. Adam moments use the "
            "configured optimizer_state_bytes_per_param estimate."
        ),
        "per_order": per_order,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--token-ids", type=Path, required=True, help="whitespace-separated integer token IDs")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--table-size", type=int, required=True, help="bucket count per n-gram order")
    parser.add_argument("--orders", type=int, nargs="+", default=[2, 3])
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--dtype-bytes", type=int, default=2)
    parser.add_argument("--optimizer-state-bytes-per-param", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=None, help="optional leading-token cap for bounded pilot audits")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON output path; stdout otherwise")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        tokens = read_token_ids(args.token_ids, vocab_size=args.vocab_size, max_tokens=args.max_tokens)
        report = audit_ngram_collisions(
            tokens,
            table_size=args.table_size,
            orders=args.orders,
            d_model=args.d_model,
            dtype_bytes=args.dtype_bytes,
            optimizer_state_bytes_per_param=args.optimizer_state_bytes_per_param,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
