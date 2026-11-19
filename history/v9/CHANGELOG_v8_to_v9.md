# Changelog — v8 to v9

## Added
- A Draft 2020-12 experiment-manifest JSON Schema with conditional requirements for training, serving, P7, P8 and P9.
- Candidate-profile, benchmark-contract and KV-policy schemas.
- An intentionally incomplete run-manifest template; it is designed to fail until real environment/run values are supplied.
- An offline validator for references, numbering, Markdown/TXT equality, YAML/JSON schemas, candidate IDs, KV policies, checksums and archive integrity.
- CPU-only, accelerator, heterogeneous and managed hardware identities; bare-metal/container/managed environment digests; dataset role-specific allowed uses and rights-review attestations.
- Explicit metric definitions with units, numerators, denominators and comparison boundaries.
- Canonical benchmark-contract field paths aligned to the normative run-manifest schema, plus validator checks for field-path drift and role/use coverage.
- Stronger semantic gates: evaluation runs require approved test/benchmark data; RL requires a separately approved reward dataset; P8 bounded approximations must demonstrate a named metric within a declared bound; P9 must report five edit-quality metrics; P8/P9 are restricted to speculative-decoding runs.
- Run metrics declare a value kind and support raw numerator/denominator counts, replicate values and confidence intervals; ratio values are checked against counts and zero denominators are represented as undefined. Failed P8 fidelity tests can be retained as `unverified` without making an exactness claim.
- 27 automated regression tests, including positive and negative fixtures for these semantic requirements.
- A dependency-ordered implementation roadmap from execution-envelope declaration through P0, matched sequence-path baselines, cache/capacity/residual/memory ablations and independent generation tracks.
- Metric definitions for speculative acceptance, state realization, edit damage, correction precision/recall, latency and energy.

## Corrected
- Removed a duplicated copy of Section 31.1–31.4 discovered at the end of the v8 main document.
- Updated the active decision/status sections and revision metadata to v9.
- Expanded required run provenance to include immutable code, data, hardware inventory, software environment, tokenizer/configuration, kernel, measurement and raw-result identities.

## Not claimed
- No model was trained; no GPU kernel, end-to-end inference path or speculative sampler was executed.
- No hardware, data rights, model scale or compute budget was invented.
- Schema validity is preflight structure, not proof of scientific validity or quality.
- Concrete run-manifest hashes are externalized in a checksum sidecar to avoid a circular self-hash.
