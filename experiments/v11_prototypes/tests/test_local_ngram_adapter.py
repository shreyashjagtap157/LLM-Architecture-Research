from __future__ import annotations

import pytest
import torch

from experiments.v11_prototypes.local_ngram_adapter import (
    HashedNGramResidualAdapter,
    causal_ngram_indices,
)


def test_hash_shape_range_and_determinism() -> None:
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    first = causal_ngram_indices(tokens, vocab_size=8, table_size=31, orders=(2, 3, 4))
    second = causal_ngram_indices(tokens, vocab_size=8, table_size=31, orders=(2, 3, 4))

    assert first.shape == (2, 4, 3)
    assert torch.equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) < 31


def test_lookup_is_prefix_causal_for_future_token_changes() -> None:
    original = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    changed_future = original.clone()
    changed_future[:, 4:] = torch.tensor([[17, 18]])

    a = causal_ngram_indices(original, vocab_size=32, table_size=257, orders=(2, 3, 4))
    b = causal_ngram_indices(changed_future, vocab_size=32, table_size=257, orders=(2, 3, 4))

    assert torch.equal(a[:, :4], b[:, :4])
    assert not torch.equal(a[:, 4:], b[:, 4:])


def test_adapter_prefix_output_is_invariant_to_future_token_ids() -> None:
    torch.manual_seed(7)
    adapter = HashedNGramResidualAdapter(
        vocab_size=32, d_model=16, table_size=127, orders=(2, 3)
    ).eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    future_changed = tokens.clone()
    future_changed[:, 4:] = torch.tensor([[20, 21]])
    hidden = torch.randn(1, 6, 16)

    with torch.no_grad():
        output_a = adapter(tokens, hidden)
        output_b = adapter(future_changed, hidden)

    assert torch.equal(output_a[:, :4], output_b[:, :4])


def test_adapter_shape_finiteness_and_gradient_flow() -> None:
    torch.manual_seed(11)
    adapter = HashedNGramResidualAdapter(
        vocab_size=64, d_model=24, table_size=97, orders=(2, 3, 4)
    )
    tokens = torch.randint(0, 64, (3, 9))
    hidden = torch.randn(3, 9, 24, requires_grad=True)

    output = adapter(tokens, hidden)
    assert output.shape == hidden.shape
    assert torch.isfinite(output).all()

    output.square().mean().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert hidden.grad.abs().sum() > 0
    for table in adapter.tables:
        assert table.weight.grad is not None
        assert torch.isfinite(table.weight.grad).all()
        assert table.weight.grad.abs().sum() > 0
    for parameter in (
        adapter.query_projection.weight,
        adapter.key_projection.weight,
        adapter.value_projection.weight,
        adapter.gate_bias,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_empty_sequence_is_supported() -> None:
    tokens = torch.empty((2, 0), dtype=torch.long)
    indices = causal_ngram_indices(tokens, vocab_size=10, table_size=13, orders=(2, 3))
    assert indices.shape == (2, 0, 2)

    adapter = HashedNGramResidualAdapter(vocab_size=10, d_model=8, table_size=13)
    hidden = torch.empty((2, 0, 8))
    assert adapter(tokens, hidden).shape == (2, 0, 8)


@pytest.mark.parametrize(
    ("tokens", "kwargs", "error"),
    [
        (torch.tensor([1, 2]), {"vocab_size": 4, "table_size": 7}, ValueError),
        (torch.tensor([[1.0, 2.0]]), {"vocab_size": 4, "table_size": 7}, TypeError),
        (torch.tensor([[-1, 1]]), {"vocab_size": 4, "table_size": 7}, ValueError),
        (torch.tensor([[1, 4]]), {"vocab_size": 4, "table_size": 7}, ValueError),
        (torch.tensor([[1, 2]]), {"vocab_size": 4, "table_size": 0}, ValueError),
        (torch.tensor([[1, 2]]), {"vocab_size": 4, "table_size": 7, "orders": ()}, ValueError),
        (torch.tensor([[1, 2]]), {"vocab_size": 4, "table_size": 7, "orders": (2, 2)}, ValueError),
    ],
)
def test_invalid_hash_inputs(tokens: torch.Tensor, kwargs: dict, error: type[Exception]) -> None:
    with pytest.raises(error):
        causal_ngram_indices(tokens, **kwargs)


def test_adapter_rejects_shape_and_device_mismatch() -> None:
    adapter = HashedNGramResidualAdapter(vocab_size=10, d_model=8, table_size=13)
    with pytest.raises(ValueError, match="shape"):
        adapter(torch.ones((1, 3), dtype=torch.long), torch.ones((1, 3, 7)))
    with pytest.raises(ValueError, match="match"):
        adapter(torch.ones((1, 2), dtype=torch.long), torch.ones((1, 3, 8)))
    with pytest.raises(TypeError, match="floating"):
        adapter(torch.ones((1, 2), dtype=torch.long), torch.ones((1, 2, 8), dtype=torch.long))
