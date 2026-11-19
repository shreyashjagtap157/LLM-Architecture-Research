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


## `audit_ngram_collisions.py`

This utility accepts a local whitespace-separated file of integer token IDs and reports exact complete-n-gram type collisions per order, bucket occupancy, collision exposure and approximate embedding-table storage. It does not print token IDs or source text. Only run it on a dataset you have already reviewed and cleared for the intended use.

Example:

```bash
python experiments/v11_prototypes/audit_ngram_collisions.py \
  --token-ids token_ids.txt --vocab-size 32000 --table-size 65536 \
  --orders 2 3 4 --d-model 512 --dtype-bytes 2 \
  --optimizer-state-bytes-per-param 4 --output reports/ngram-audit.json
```

The report counts complete n-grams only and uses the same polynomial hash constants and modulo-bucket mapping as the prototype adapter. Memory estimates cover embedding-table weights, gradients and two Adam moments only; they exclude projections, activations, temporary audit structures, allocator overhead and runtime state. The audit stores distinct n-gram types in memory, so its memory use is O(number of unique n-gram types); it is a pilot-scale exact audit, not a scalable streaming estimator. A supplied `--max-tokens` audits only the leading prefix and must be recorded as such.

The audit is diagnostic, not a training or quality benchmark. It has only synthetic tests at this stage; no real-corpus collision rate has been measured.


## Dataset-rights preflight

Use `dataset_rights_manifest.template.json` to create a **pending** manifest for one exact immutable dataset revision and one declared intended use. The template is intentionally incomplete and must fail closed until the rights and privacy reviews are actually completed.

Run the preflight for the exact use being considered:

```bash
python experiments/v11_prototypes/dataset_rights_gate.py \
  --manifest dataset-rights-review.json \
  --use research_training
```

The accepted use values are `research_training`, `commercial_training`, `evaluation`, `model_distribution`, `hosted_service`, and `public_data_redistribution`. The requested use must be approved both by the overall human review and for every included upstream source.

The checker requires a full immutable dataset commit SHA, hashes of the dataset-card and terms snapshots, a source inventory with immutable identifiers and per-source licence-evidence hashes, an identified reviewer/jurisdiction/use decision, and a separate privacy review report hash plus removal plan. It does not download data or independently check licence/terms URLs.

A pass is named `ATTESTATIONS_PRESENT_FOR_REQUESTED_USE`. It means only that the manifest is structurally complete and contains the requested human attestations. It does **not** verify the correctness or authority of those attestations, interpret licence scope, certify privacy compliance, or constitute legal clearance. The tool must never be used to convert a pending/uncertain source into an approved one by filling fields mechanically.
