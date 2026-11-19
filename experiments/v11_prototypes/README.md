# V11 research prototypes (new, not recovered)

This directory contains small, newly authored prototypes reconstructed from the research questions in the session log. It does **not** contain the original V11 source code, checkpoints, or raw experiment manifests.

## `local_ngram_adapter.py`

A compact testable baseline for causal hashed local n-gram retrieval and context-gated residual injection. It uses one embedding table per configured n-gram order, an independently projected query/key gate, and a projected residual value.

The prototype is intentionally limited:

- It is untrained and has only been tested with synthetic tensors.
- It is not an implementation of Engram, SCONE, or the prior-art papers cited in `docs/research/PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md`.
- It is not a reproduction of the missing V11 experiment.
- It makes no novelty, quality, speed, or memory-efficiency claim.
- Hash collisions are expected; collision rates and table-size trade-offs have not yet been measured on a corpus.
- The token-ID lookup path is causal. The caller remains responsible for ensuring that `hidden_states` are themselves causal.
- No user dataset or public text corpus is loaded, and no training is performed by these tests.

Run the isolated unit tests from the repository root:

```bash
python -m pytest -q experiments/v11_prototypes/tests/test_local_ngram_adapter.py
```

The next gate is to measure distinct n-gram collisions and table footprint on a rights-cleared dataset, then compare against statistical n-gram and parameter-matched neural controls under validation-only selection. Do not promote this prototype based on unit tests alone.
