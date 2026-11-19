# Current status — 2026-11-19

## Executive state

The latest complete design document recovered is **SOTA LLM Architecture v10**. The runnable model-training/evidence baseline is the **v9 miniature character-model pilot**. A small, newly authored V11 research prototype for hashed local n-gram memory has since been added under `experiments/v11_prototypes/`; it passes synthetic unit tests but is untrained and has no quality or efficiency evidence. The original V11 dynamic-computation implementation, checkpoints and raw result artifacts remain unrecovered; only the conversation-level experiment record is available for that work.

The repository is therefore a **research and handoff snapshot**, not a complete V11 implementation repository and not a trained official LLM.

## Artifact-backed facts

| Area | State | Evidence location |
|---|---|---|
| Architecture research | Complete v10 specification present | `docs/architecture/SOTA_LLM_Architecture_v10.md` |
| Architecture history | v3–v9 documents, audits and changelogs present | `history/` |
| v9 example models | Dense P0, recurrent/attention P1 and local/global P1-LA code is present | `experiments/v9_character_pilot/train_pilot.py` |
| v9 data tooling | Immutable-revision exporter and split-manifest validation code present | `experiments/v9_character_pilot/prepare_hf_dataset.py` |
| v9 pilot results | Checkpoints, logs, metrics and source-identified repeat artifacts present | `experiments/v9_character_pilot/results/` |
| v9 tests | **44 passed** during this packaging session | `experiments/v9_character_pilot/test_*.py` |
| v9 artifact verification | **137 checks passed** during this packaging session | `experiments/v9_character_pilot/verify_pilot_artifacts.py` |
| v9 fresh repeat | Six source-identified runs completed 2026-11-19 07:52:46 UTC (two seeds × three profiles, 500 steps/run); metrics match the archived aggregate | `experiments/v9_character_pilot/results/pilot_results.json` |
| New V11 local-memory prototype | 13 synthetic-tensor unit tests passed locally; untrained and not a reproduction of the missing V11 code | `experiments/v11_prototypes/` |
| User inputs | v2 baseline text and ideas compilation preserved | `inputs/` |

## v9 pilot interpretation

The v9 pilot trained tiny character-level neural candidates from a 29,926-byte Tiny Shakespeare excerpt retrieved from a public GitHub blob. P1 modestly improved test NLL/accuracy relative to dense P0, but had materially lower CPU throughput and was beaten by a smoothed bigram baseline. The report concludes feasibility demonstrated, usefulness not established.

The full upstream text and other datasets were not downloaded in that experiment. The excerpt’s training-use rights remain pending. Refer to `experiments/v9_character_pilot/TRAINING_PILOT_REPORT_v9.md` and `NOTICE.md`.

## Dynamic-computation session findings (conversation-reported)

The subsequent session explored ACT-style iterative refinement, token-selective attention, teacher-supervised value-of-computation routing and pair/N-gram representation adapters. Several designs reduced abstract compute steps, but Python-level routing overhead outweighed the savings on CPU. Per-token causal routing was retained; sequence-global top-k routing was rejected due future leakage.

A pair/N-gram adapter was reported to beat a near-parameter-matched wider FFN on the small character corpus across three seeds. This is promising but domain-limited and not sufficient to establish general language-model accuracy gains.

Exact details, failed runs, evaluation corrections and the evidence limitations are recorded in `docs/research/DYNAMIC_COMPUTE_SESSION_LOG.md`.

## Recovery boundary and missing artifacts

The working directory at packaging time contained only three files in the prior `llm-architecture-research-v11/` folder: `README.md`, `HANDOFF.md` and `DATA_NOTICE.md`. Their described `src/`, `tests/`, `scripts/`, `data/results_*` and `docs/EXPERIMENTAL_REVIEW_V11.md` content was not present. No V11 dynamic model source, ACT/V.O.C. checkpoints, raw per-seed JSON/CSV metrics or validator scripts were recovered from the conversation file inventory either.

The old generated handoff drafts are preserved in `history/session_handoffs/` for traceability, but they are superseded where they describe missing files or claim tests that cannot be repeated from this snapshot. Their reported V11 test counts are not represented as verified here.

## Infrastructure and publication blockers

- Local execution: Python 3.13.5, PyTorch 2.10.0+cpu; CUDA unavailable.
- Note: `train_pilot.py` does not implement `--help`; invoking it starts its full six-run experiment and overwrites files under `results/`. This behaviour was observed during packaging and is documented; do not run it casually.
- Remote Hugging Face Jobs attempt returned HTTP 402 Payment Required; no paid job was created.
- GitHub publication is complete. The branch was rechecked at `c3b6df779becdd7dcc58e776e1f6cd7f672e2739` before the current research update.
- The published `main` tree was verified against the supplied Git bundle by root Git tree object ID. The matching pre-update root tree SHA was `610f38e65b8e9042b39f7ca548c79a6197c12f6f` and contained 145 tracked entries; the publication/correction commit preserves the original five-commit history.
- GitHub reports the repository as public. This conflicts with the conservative recommendation in `PUBLISHING.md` while redistribution rights remain unresolved; private visibility is recommended until human review is complete.
- The separate `shreyashjagtap157/Dynamic-Neural-Computing` project was not modified.

## Status labels

- **Artifact-verified:** files exist here and their checks were run during packaging.
- **Session-reported:** results were stated in the conversation but their exact raw artifacts are absent here.
- **Not demonstrated:** no experiment or external validation supports the claim.
- **Blocked:** current compute, data rights, network or publishing capability prevents completion.

## Repository synchronization verification — 2026-11-19

The published `main` tree has been compared with the supplied Git bundle by root Git tree object ID. The matching tree SHA is stronger than a filename-only comparison: it identifies the same complete tree of paths, modes and blob object IDs. The local verification run also reported 145/145 SHA-256 manifest entries passing, 44/44 v9 tests passing, and 137/137 pilot artifact checks passing. These checks do not validate the absent V11 dynamic-compute implementation or clear third-party data rights.

## Prior-art review — 2026-11-19

A repository-level literature review was added at `docs/research/PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md`. It identifies direct prior art for token-wise residual gating/adaptive depth (including a paper explicitly named Token-Selective Attention) and hashed n-gram memory with context-aware residual injection. Accordingly, the earlier session label is retained historically as `session_tsa`; new artefacts must distinguish it from the paper's `tsa_2605` method. Pair/N-gram work remains the first reproduction hypothesis but is not treated as a broad novelty claim.

The experiment plan now requires statistical n-gram and matched neural controls, validation-only selection and one sealed test evaluation, held-out complexity/length generalization, routing stability and budget reliability, and safety evaluation separated between prompt prefill and answer generation. These are documented gates. A new standalone hashed n-gram adapter prototype is now present and has passed 13 synthetic tests; it is not trained, does not reproduce the missing historical V11 source, and supplies no model-quality or efficiency evidence. Original V11 source/checkpoints/raw manifests remain absent, so the session metrics are still session-reported.


## Prototype implementation and validation update — 2026-11-19

The new `experiments/v11_prototypes/local_ngram_adapter.py` prototype computes causal hashed n-gram buckets from token prefixes and injects the looked-up representation through a context-dependent sigmoid gate and projected residual. It includes input/domain checks and handles empty sequences. The companion `audit_ngram_collisions.py` utility measures complete n-gram type collisions by order and estimates table, gradient and Adam-moment storage from a caller-supplied token-ID file; it stores unique n-grams in memory and is therefore intended for pilot-scale audits. The caller must provide already-causal hidden states; this module does not establish causality of the upstream network.

Validation performed locally against an uploaded-bundle checkout:
- Prototype tests: **29 passed**.
- Collision/storage audit tests: included in the 29-test prototype suite; the audit tool has not yet been run against a corpus.
- Dataset rights gate: 10 tests cover approved/pending states, per-source licence evidence, intended-use restrictions, immutable revisions and privacy-review requirements. The tool only validates attestations and fails closed by default.
- Existing v9 regression tests: **44 passed**.
- Existing v9 artifact verifier: **137 checks passed**.
- Environment: Python 3.13.5, PyTorch 2.10.0+cpu, pytest 9.0.2.
- No corpus training, checkpoint creation, dataset collision-rate measurement, latency benchmark, or hardware optimization was performed. The collision-audit utility has only synthetic unit-test coverage so far.

GitHub Actions run `44539198024` completed successfully on commit `bc56c68d12f083a2ad025761d5905950f69d1fbb`: the 19-test V11 prototype suite, 44-test v9 regression suite, 137-check v9 artifact verifier, checksum verification and dataset-exporter CLI check all passed. The workflow refreshed the manifest and advanced `main` to `113c4f8e132a2dd6ab7e735152894034a20addb0`. At that verified head, `SHA256SUMS.txt` contained 154 entries including the adapter, collision audit and tests.


## Dataset rights review — 2026-11-19

A preliminary source-card/licence review is recorded in `docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md`. It assessed TinyStories, WikiText, Open License Corpus/PubText, CuratedCorpus-5B and Kyoto-Corpus. None is approved for this project yet; no candidate was downloaded or trained on. TinyStories is tagged CDLA-Sharing-1.0; WikiText lists CC-BY-SA-3.0/GFDL; OLC's top-level Apache metadata coexists with source-specific CC-BY-SA subsets, a train-only repository split and an explicit residual-PII warning; CuratedCorpus-5B and Kyoto need stronger source-level rights/provenance checks. This is preliminary due diligence, not a legal opinion.


## Dataset rights preflight — 2026-11-19

Added `experiments/v11_prototypes/dataset_rights_gate.py` and a pending JSON manifest template. The preflight requires a full dataset commit SHA, hashes of the dataset-card/terms snapshots, per-included-source immutable identifiers and licence-evidence hashes, an approved reviewer/use record, and a separate approved privacy review. Its output is explicitly `ATTESTATIONS_PRESENT_FOR_REQUESTED_USE`, not legal clearance. The default template is pending and must fail closed. Local tests for this gate: 10 passed; total V11 prototype tests: 29 passed. The checker does not fetch datasets or validate the truth of human attestations.


## Latest GitHub Actions verification — 2026-11-19

The rights-preflight commit `9c3326272613465ed904bf1cfbf9095c94dc54a4` completed GitHub Actions run **44542481592** with conclusion `success`. The workflow ran the full `experiments/v11_prototypes/tests` directory (29 tests), the v9 regression suite (44 tests), v9 artifact verification (137 checks), regenerated and verified the SHA-256 manifest, and checked the dataset-exporter help path. The checksum bot advanced `main` to `5b00e36aba6c69d7ca2ad1403952412a0ceabad0`; at that head the manifest contains **158 file entries**. These automated tests verify code and archived artifacts; they do not approve a dataset, train a model or reproduce the missing historical V11 runs.
