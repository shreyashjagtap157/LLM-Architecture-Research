# LLM Architecture Research

**Snapshot:** 2026-11-19  
**Current architecture specification:** v10  
**Experiment baseline:** v9 miniature character-model pilot  
**Dynamic-computation work:** V10/V11 session experiments; evidence currently only partially reproducible from this bundle

This repository preserves the research progression from the original v2 synthesis through v9, the later v10 architecture specification, the v9 runnable example-model pilot, and the dynamic-computation experiments discussed in the 18–19 November 2026 sessions.

> **Research status:** example-model training is demonstrated at small character-model scale. The v9 hybrid candidate did not beat the smoothed bigram control on its tiny excerpt. Later session experiments reported promising N-gram/pair-adapter accuracy gains and several adaptive-computation trade-offs, but their original V11 source code and raw artifacts were not recoverable into this snapshot. A separate, newly authored hashed n-gram adapter and collision/storage audit are present under `experiments/v11_prototypes/`; they are untrained and validated only with synthetic tests. No official subword LLM, frontier-scale model, or GPU-optimized dynamic-computation result has been demonstrated.

## Start here

1. [`HANDOFF.md`](HANDOFF.md) — exact current state and continuation guidance.
2. [`docs/status/CURRENT_STATUS_2026-11-19.md`](docs/status/CURRENT_STATUS_2026-11-19.md) — what is present, verified, reported, missing, and blocked.
3. [`docs/architecture/SOTA_LLM_Architecture_v10.md`](docs/architecture/SOTA_LLM_Architecture_v10.md) — latest complete architecture specification in the recovered workspace.
4. [`docs/research/DYNAMIC_COMPUTE_SESSION_LOG.md`](docs/research/DYNAMIC_COMPUTE_SESSION_LOG.md) — ACT, token-selective compute and value-of-computation experiment findings, including failed approaches.
5. [`docs/research/EXPERIMENTAL_REVIEW.md`](docs/research/EXPERIMENTAL_REVIEW.md) — interpretation and validity review.
6. [`docs/NEXT_EXPERIMENTS.md`](docs/NEXT_EXPERIMENTS.md) — recommended experimental order.
7. [`NOTICE.md`](NOTICE.md) — data, source and licensing limitations.
8. [`docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md`](docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md) — candidate source/licence review; no dataset approved yet.
8. [`docs/research/PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md`](docs/research/PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md) — prior-art comparison and research implications.
9. [`experiments/v11_prototypes/README.md`](experiments/v11_prototypes/README.md) — scope, collision audit, rights preflight and limitations of the new isolated prototype.

## Repository map

```text
.
├── docs/
│   ├── architecture/                  # current v10 architecture specification
│   ├── research/                      # dynamic-computation log and experimental review
│   ├── status/                        # snapshot status and recovery boundaries
│   └── NEXT_EXPERIMENTS.md
├── experiments/
│   ├── v9_character_pilot/            # runnable v9 code, reports, tests and checkpoints
│   └── v11_prototypes/                # untrained adapter, collision audit and rights preflight
├── history/
│   ├── v3 ... v8/                     # versioned architecture and audit history
│   ├── v9/                            # v9 architecture, audits and reports
│   ├── packages/                      # prior v9 package snapshots
│   └── session_handoffs/              # superseded handoff drafts preserved for audit
├── inputs/                            # original user-provided v2 text, idea archive and prior packages
├── CHANGELOG.md
├── HANDOFF.md
├── NOTICE.md
└── PUBLISHING.md
```

## What can be run from this snapshot

The v9 pilot is a complete, small-scale character-model implementation with its tests, checkpoint artifacts, dataset exporter and validation tools. The current container is CPU-only (Python 3.13.5, PyTorch 2.10.0+cpu).

From the repository root:

```bash
cd experiments/v9_character_pilot
python -m pip install -r requirements.txt
python -m pip install -r pilot_requirements.txt
python -m pytest -q test_dataset_pipeline.py test_validate_architecture_package.py
python verify_pilot_artifacts.py
# Safe: this exporter exposes argparse help.
python prepare_hf_dataset.py --help

# WARNING: train_pilot.py is an experiment runner, not a CLI parser.
# It ignores --help and starts the complete six-run CPU pilot (500 steps/profile/seed),
# overwriting the current results/ outputs. Do not use it as a smoke/help command.
```

The v9 test suite was executed while assembling this repository: **44 tests passed**. The v9 artifact verifier reported **137 checks passed**. The isolated V11 prototype suite currently has **29 synthetic tests**. These tests validate the prototype's local contracts, not model quality, the missing historical V11 dynamic-compute code, or real-corpus collision rates.

See [`experiments/v9_character_pilot/TRAINING_PILOT_REPORT_v9.md`](experiments/v9_character_pilot/TRAINING_PILOT_REPORT_v9.md) for the exact experiment scope, configurations, metrics and limitations.

## Data selection status

No dataset is approved for further training. The current Tiny Shakespeare excerpt remains exploratory-only; candidate dataset-card tags are not treated as a rights determination. Read [`docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md`](docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md) before materializing any new corpus.

## What this repository does not claim

- It does not claim that the full v10 synthesis has been implemented as one model.
- It does not claim an official subword LLM has been trained.
- It does not claim that dynamic computation currently improves end-to-end latency. The reference Python routing paths were reported slower on CPU despite reducing abstract token-step counts.
- It does not claim full online-dataset training. The v9 pilot used a 29,926-byte excerpt retrieved from a public GitHub blob; remote compute returned HTTP 402 and full remote dataset ingestion was not completed.
- It does not claim commercial or production data rights for the bundled excerpt.
- It does not claim that the V11 session results are independently reproducible: the source files and raw result set referenced by the superseded V11 handoff drafts were not available in the recovered working directory at packaging time.

## Git and publication state

The canonical remote is [`shreyashjagtap157/LLM-Architecture-Research`](https://github.com/shreyashjagtap157/LLM-Architecture-Research), with `main` as its default branch. The uploaded bundle's initial root tree was verified identical to the imported GitHub tree (145 tracked files) before subsequent documentation, CI and prototype additions. The five original bundle commits remain ancestors of the live `main` history; inspect the branch's current head before synchronizing because CI may create checksum-refresh commits.

Two later commits briefly added and then removed an unrelated AetherOS prompt. The original imported tree was preserved, with later research and tooling commits added on top. The unrelated prompt remains absent from the current file tree. The commits remain in history; history was not rewritten.

**Visibility warning:** GitHub reported this repository as **public** at the last check. The repository intentionally has no blanket licence, and `NOTICE.md` records unresolved redistribution/training-use questions for supplied inputs and the Tiny Shakespeare transcription. Public visibility is therefore not equivalent to rights clearance. Review those notices before further distribution or commercial use; consider making the repository private until the rights review is complete.

For clone, bundle, and push instructions, see [`PUBLISHING.md`](PUBLISHING.md). The bundle is a historical snapshot, not a substitute for fetching the current remote `main` before making new commits.
