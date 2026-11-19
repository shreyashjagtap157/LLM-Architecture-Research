# Continuation handoff — LLM Architecture Research

**Snapshot date:** 19 November 2026  
**Current complete architecture specification:** v10  
**Runnable implementation baseline:** v9 character-level pilot  
- **Repository branch:** `main`
- **Canonical remote:** `https://github.com/shreyashjagtap157/LLM-Architecture-Research.git`
- **Branch policy:** continue on `main`; inspect its live head before each change and use fast-forward-only updates.
- **Import integrity:** the five-commit uploaded-bundle history is retained; the imported 145-file tree was verified identical before subsequent documentation, CI and prototype additions.

## Read first

1. `README.md`
2. `docs/status/CURRENT_STATUS_2026-11-19.md`
3. `docs/architecture/SOTA_LLM_Architecture_v10.md`
4. `docs/research/DYNAMIC_COMPUTE_SESSION_LOG.md`
5. `docs/research/EXPERIMENTAL_REVIEW.md`
6. `docs/NEXT_EXPERIMENTS.md`
7. `NOTICE.md`

## Verified in this package

- The architecture lineage and supporting docs v3–v9 are preserved under `history/` and the v9 pilot folder.
- `docs/architecture/SOTA_LLM_Architecture_v10.md` is the latest complete architecture specification recovered from the session.
- The v9 miniature pilot source, training/evaluation scripts, dataset exporter, configs, reports, checks, checkpoints and original inputs are present under `experiments/v9_character_pilot/` and `inputs/`.
- The v9 tests were run during packaging: `44 passed`.
- `verify_pilot_artifacts.py` ran during packaging and reported `137` checks passed.
- New isolated prototype: `experiments/v11_prototypes/local_ngram_adapter.py`, with `19` synthetic-tensor tests passing locally. This is newly authored research code, not recovered V11 source and not trained.
- A fresh, source-identified v9 repeat completed on 2026-11-19 at 07:52:46 UTC: two seeds × three profiles, 500 steps/run. It wrote six checkpoints and recorded trainer, exporter and dataset-manifest SHA-256 values in `results/pilot_results.json`; the aggregate metrics match the recorded pilot result.
- The uploaded bundle's five-commit history is present in GitHub `main`; the original bundle tip `304aa8e65c1a1ad36c59e404c082b11d0f4f771f` is an ancestor of the verified remote head.
- The GitHub `main` file tree is byte-for-byte content-identical at the Git object level to the uploaded bundle tree. Two intervening commits added and then removed an unrelated AetherOS prompt; it is absent from the final tree and history was preserved.
- Local verification on 2026-11-19: 145 SHA-256 manifest entries passed; v9 test suite 44/44 passed; artifact verifier 137/137 checks passed; dataset-exporter `--help` returned safely.

## Reported in the conversation, but not independently replayable from this bundle

The 18–19 November dynamic-computation work reported ACT recurrent refinement, token-selective attention, teacher-supervised value-of-computation routing and pair/N-gram residual adapter experiments. The findings and failures are transcribed in `docs/research/DYNAMIC_COMPUTE_SESSION_LOG.md` with explicit provenance labels.

The recovered runtime contained the v10 specification and the v9 pilot archive, but did not contain the `src/llm_arch/`, V11 dynamic-compute scripts, per-seed V11 result JSON/CSV files, or ACT/V.O.C. checkpoints referred to in earlier generated handoff drafts. Those drafts are preserved under `history/session_handoffs/` and are marked superseded. Do not treat their claimed test counts or file tree as a currently runnable V11 implementation.

## Main findings

- The small v9 P1 hybrid candidate modestly beat dense P0 but was slower and lost to the smoothed bigram model on the tiny excerpt. It is a feasibility experiment, not evidence of useful LLM quality.
- The later session reported that a pair/N-gram residual adapter beat a near-parameter-matched wider FFN on one character corpus across three seeds. This is a promising hypothesis that must be recovered/reimplemented and re-tested on a lawful subword dataset.
- ACT and value-of-computation mechanisms can reduce abstract recurrent token-step counts, but the Python reference paths were slower on CPU. Several controller/evaluation bugs were found and corrected during the session; only corrected, validation-selected experiments should be considered in follow-up work.
- Token-selective attention routing must be strictly causal. Global top-k selection over the entire sequence was rejected because future tokens could influence earlier routing decisions.

## Immediate continuation plan

1. Recreate and commit the V11 dynamic-compute model code from the experiment log; do not pretend it is already in this bundle.
2. Add tests for causality, halting-mass conservation, ponder gradients, hard/soft routing agreement, reset isolation and parameter matching.
3. Re-run ACT, TSA and V.O.C. as separate experiments with exact launch-time code hashes and per-seed result artifacts.
4. Recover/implement the pair-N-gram adapter, compare against a parameter-matched wider FFN, and test a fixed-budget hashed/factorized subword version.
5. Obtain a complete immutable dataset revision with rights cleared for the intended training uses. Do not train commercially on the bundled excerpt without an independent rights review.
6. Run `experiments/v11_prototypes/audit_ngram_collisions.py` on a locally prepared, rights-cleared token-ID file. Record per-order collisions and table/optimizer memory estimates; note the tool's O(unique n-gram types) memory use.
7. Benchmark the untrained adapter against statistical n-gram and parameter-matched neural controls. Do not treat unit tests or collision statistics as model-quality evidence.
8. Train a small subword example model only after data/compute constraints are specified. Keep full-model scale-up blocked until matched ablations demonstrate a quality/efficiency improvement.
9. For dynamic compute, optimize actual target-hardware kernels and report wall-clock latency/throughput/energy—not only skipped theoretical token-steps.

## GitHub publication and safety

The intended target now exists and contains the project on `main`; publication is complete. See `PUBLISHING.md` for safe synchronization instructions. Do not push into the separate `Dynamic-Neural-Computing` project. GitHub currently reports this repository as public, while `NOTICE.md` still records unresolved data/source redistribution rights. Treat visibility as an open publication-risk decision; no rights clearance is implied by the successful push.
