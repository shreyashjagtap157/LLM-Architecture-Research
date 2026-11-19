# Changelog

All notable research/specification transitions are recorded here at a high level. The version-specific detailed changelogs and audits are preserved under `history/` and `experiments/v9_character_pilot/`.

## 2026-11-19 — GitHub synchronization audit and validation

- Verified GitHub `main` at `d54ba5bd1e980fb40b4e80ae3758fe6b41548799`; its root tree exactly matches the uploaded bundle tree (`610f38e65b8e9042b39f7ca548c79a6197c12f6f`, 145 tracked files).
- Confirmed the original five bundle commits remain in the remote history. An unrelated AetherOS prompt was added and subsequently removed in intermediate commits; it is absent from the current tree. No history rewrite was performed.
- Corrected the README, publishing guide, current status and handoff to reflect the actual remote state and the currently public repository visibility.
- Fixed bundle-clone synchronization instructions: a bundle clone already has an `origin`, so the GitHub URL must be configured with `git remote set-url`, not `git remote add`.
- Revalidated the repository: 145 SHA-256 entries passed, 44 v9 tests passed, 137 pilot artifact checks passed, and dataset exporter help completed safely.

## 2026-11-19 — Dataset rights preflight utility

- GitHub Actions run 44542481592 passed on the rights-preflight commit: 29 V11 prototype tests, 44 v9 regression tests, 137 v9 artifact checks, manifest regeneration/check and safe exporter CLI validation.
- Checksum automation advanced main to 5b00e36aba6c69d7ca2ad1403952412a0ceabad0 with 158 manifest entries.

- Added a fail-closed manifest checker for immutable dataset revisions, dataset-card/terms hashes, every included upstream source and its own licence-evidence hash, specifically approved intended uses, reviewer identity and privacy-review evidence.
- Added a default-pending JSON manifest template; it cannot pass until a human-reviewed record is complete.
- Added 10 tests for pending/approved states, use restrictions, per-source evidence, duplicate source IDs, immutable revision format and privacy review.
- The checker validates manifest structure and recorded attestations only; it does not interpret licences or provide legal clearance.
- Total isolated V11 prototype tests: 29 passed locally.

## 2026-11-19 — Dataset rights and provenance review

- Reviewed the current dataset cards and relevant licence text for TinyStories, WikiText, Open License Corpus/PubText, CuratedCorpus-5B and Kyoto-Corpus.
- Recorded source-level, ShareAlike, PII, validation/test and aggregate-metadata risks in `docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md`.
- No candidate was approved; no corpus was downloaded or used for training. The project remains synthetic-test-only pending immutable revision and rights approval.

## 2026-11-19 — N-gram collision and storage audit utility

- Added `audit_ngram_collisions.py` to measure complete n-gram type collisions, bucket occupancy and approximate table/gradient/Adam storage from a caller-provided token-ID file.
- Added six synthetic tests for collision-free and forced-collision cases, complete-ngram handling, storage arithmetic, token-ID validation and JSON output privacy.
- Expanded the isolated prototype suite to 19 passing tests; existing v9 regression tests remain 44/44 and the artifact verifier remains 137/137.
- Documented that the audit is exact but uses O(unique n-gram types) memory and has not yet been run on a rights-cleared corpus.
- Updated CI to run the full prototype test directory.

## 2026-11-19 — Prior-art review and revised experiment gates

- Added a sourced comparison of 2025–2026 token-adaptive computation and n-gram-memory prior art.
- Recorded the naming collision between the session's “TSA” label and the published Token-Selective Attention paper; future experiment artefacts distinguish `session_tsa` from `tsa_2605`.
- Reframed pair/N-gram work as a reproduction/efficiency hypothesis, not a broad novelty claim; added hashed lookup, statistical n-gram, matched parameter/compute and memory controls.
- Added held-out-complexity, route-stability, budget-reliability and safety evaluations separated by prompt-prefill and answer-generation phases.
- Preserved all earlier session results as session-reported because V11 source and raw run artefacts remain absent.

## 2026-11-19 — Research repository snapshot

- Consolidated v3–v9 architecture specifications, audits, changelogs, contracts, validation tooling and original inputs.
- Added the complete v10 architecture specification recovered from the dynamic-computation research session.
- Preserved the v9 runnable example-model pilot, checkpoints, results, online-data exporter and regression tests.
- Added a conversation-provenance dynamic-computation log for ACT, causal token-selective attention, V.O.C. and pair/N-gram adapter experiments.
- Distinguished artifact-verified v9 results from V11 session-reported results whose raw source/checkpoints/results were not recovered into this bundle.
- Added current status, handoff, evidence limitations, data notice, next-experiment gates and GitHub publishing instructions.
- Initialized a new local Git repository on `main`; no remote configured. No pre-existing Git history was available in the recovered workspace.
- Ran the v9 test suite: 44 passed. Ran the v9 artifact verifier: 137 checks passed.

## v9 and earlier

See `experiments/v9_character_pilot/CHANGELOG_v8_to_v9.md` and `history/v3` through `history/v8` for the detailed revision history, technical audits and validation reports.
