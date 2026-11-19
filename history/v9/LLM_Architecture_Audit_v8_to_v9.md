# Technical Audit — v8 to v9

## 1. Scope
This revision audits the v8 research specification and turns its research plan into an executable preflight contract. It does not re-audit all papers from scratch or claim new model-level experimental evidence.

## 2. Findings and fixes

| Finding | Risk | v9 action |
|---|---|---|
| The v8 tail repeated Section 31.1–31.4 after the first complete occurrence. | Confusing and duplicate navigation; earlier static check did not detect the repeated full block. | Retained one canonical copy and added explicit duplicate-numbering checks. |
| Reproducibility requirements were prose/checklists but lacked a machine-checkable schema. | A run could be reported without required provenance or state/fidelity fields. | Added Draft 2020-12 experiment manifest schema and semantic validator. |
| Candidate YAML and benchmark YAML had no formal schemas. | Structural drift could go unnoticed. | Added schemas and validation to the package validator. |
| The P8/P9 fidelity contract was documented but not operationally enforced. | A diffusion-draft method could claim exact AR target fidelity without proposal-probability, verifier or state-commit evidence. | P8 conditional fields and semantic checks; P9 must declare changed-output semantics. |
| The benchmark protocol named important metrics without defining their denominators or measurement boundaries. | Incomparable acceptance, edit-quality, latency and energy numbers. | Added definitions and required manifest-level protocol fields. |
| The roadmap was dependency prose rather than a gate model. | Experiments could be run out of order or advanced without prerequisite evidence. | Added machine-readable G0–G13 roadmap with predecessor and exit gate. |
| The first manifest schema assumed a discrete accelerator and container runtime, and a single license-status field could not express split-specific rights. | It would reject CPU-only reference tests and oversimplify dataset-use governance. | Added CPU/accelerator/heterogeneous/managed hardware types, immutable device-inventory and bare-metal/container/managed environment digests, explicit role-specific allowed-use declarations and rights-review status for all runs. |
| Hardware, scale, data rights and budget remain unspecified. | Numeric configurations would be false precision. | Kept unknowns explicit and left the manifest template incomplete. |

## 3. Normative v9 additions
- The experiment manifest binds spec, source, data, hardware, software, model (including tokenizer and architecture-config digests), run, evaluation and artifacts.
- Dataset rights are checked against each declared role: training requires training use; validation requires validation plus model-selection use; test/benchmark require evaluation use; reward requires reward-model-training use. All declared uses require an approved rights-review status.
- Continued pretraining, SFT and RL require parent checkpoint identity.
- A dirty source tree requires a patch digest.
- The run manifest is hashed externally; a self-referential manifest digest is not allowed.
- Active parameters cannot exceed total parameters.
- P8 exact target fidelity requires proposal probabilities and a passing fidelity test; otherwise exactness is not a permitted label. P8 target checkpoint and P9 AR-draft checkpoint identities are cross-checked against the model checkpoint.
- P9 is explicitly changed-output unless separately redesigned and proven.
- P7 records all generation modes and the mode-router revision.
- Metrics need a declared value kind, unit and protocol; ratios carry raw numerator/denominator, confidence intervals are ordered and repeated metric names are rejected. Common run fields are split from conditional training/inference/P7/P8/P9 fields. Dataset provenance and allowed-use attestation, hardware topology, software environment and lockfile digests are common or conditional according to run type; optimizer configuration and learning-rate schedule are training-only identities.

## 4. Validation boundaries
The package validator checks data shape, schema constraints, simple cross-field invariants, document numbering/references, hashes and ZIP integrity. It cannot inspect whether a license grant is legally adequate, whether a dataset is contaminated, whether a kernel is numerically correct, whether a proof is valid, or whether the architecture is better. Those require separate domain-specific audits and actual experiments.

## 5. Remaining blockers
A valid experiment requires the user/project to select the target workload, accelerator and topology, parameter/active-parameter budget, compute budget, dataset and allowed-use status, latency/cost/energy limits, and evaluation policy. No run manifest has been completed because those values are not present in the supplied files or environment.


## Additional v9 contract-hardening pass

The first v9 packaging/preflight run exposed stale checksum entries and a distributable-file allowlist that incorrectly counted an internal build helper. These were corrected. A separate audit found that the benchmark contract still used legacy accelerator-only aliases (`accelerator_model`, `accelerator_count`, `HBM_per_device`) and did not use the same canonical field paths as the experiment manifest schema. The benchmark contract was rewritten around explicit dot-separated manifest paths and now declares role/use requirements for training, validation, test, benchmark and reward datasets. The validator rejects unknown/retired aliases and verifies that every environment path is part of common run identity.

The run contract was further tightened: inference/evaluation/ablation/speculative runs must identify an evaluation-authorized test/benchmark dataset; RL must identify a reward dataset authorized for reward-model training; P8 bounded-approximation claims must include a named measured metric no greater than a declared bound; P9 must report numeric correct-token-retention, edit-damage, correction-precision, correction-recall and refinement-convergence metrics; and P8/P9 are reserved for speculative-decoding runs. The run schema requires each metric to declare its value kind and supports metric numerators/denominators, replicate values and confidence intervals; P9 fraction values are recomputed from raw counts, and zero-denominator ratios are explicitly undefined. P8 failure evidence is preservable as `unverified` rather than rejected or mislabeled as exact. The suite now contains 27 automated regression tests. These changes strengthen preflight integrity only; they do not establish correctness of any model or sampler.
