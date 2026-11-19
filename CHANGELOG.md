# Changelog

All notable research/specification transitions are recorded here at a high level. The version-specific detailed changelogs and audits are preserved under `history/` and `experiments/v9_character_pilot/`.

## 2026-11-19 — GitHub synchronization audit and validation

- Verified GitHub `main` at `d54ba5bd1e980fb40b4e80ae3758fe6b41548799`; its root tree exactly matches the uploaded bundle tree (`610f38e65b8e9042b39f7ca548c79a6197c12f6f`, 145 tracked files).
- Confirmed the original five bundle commits remain in the remote history. An unrelated AetherOS prompt was added and subsequently removed in intermediate commits; it is absent from the current tree. No history rewrite was performed.
- Corrected the README, publishing guide, current status and handoff to reflect the actual remote state and the currently public repository visibility.
- Fixed bundle-clone synchronization instructions: a bundle clone already has an `origin`, so the GitHub URL must be configured with `git remote set-url`, not `git remote add`.
- Revalidated the repository: 145 SHA-256 entries passed, 44 v9 tests passed, 137 pilot artifact checks passed, and dataset exporter help completed safely.

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
