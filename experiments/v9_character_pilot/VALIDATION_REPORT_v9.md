# Validation Report — LLM Architecture v9 + Training Feasibility Pilot

**Status:** STATIC PACKAGE, REGRESSION AND SMALL-SCALE PILOT-INTEGRITY CHECKS PASSED  
**Review date:** 18 November 2026  
**Scope:** architecture document integrity, references, JSON/YAML contracts, run-manifest semantics, dataset exporter/loader, regression fixtures, model/checkpoint invariants, checksums and archive integrity.

## Main specification
- Lines: 2,008
- Words: 29,802
- UTF-8 bytes: 228,400
- Bibliography entries: 104 (R1–R104; citation ranges are expanded and checked)
- Main sections: 34 (sequential 0–33)
- Numbered subsection identifiers: 156; no duplicates
- Markdown code fences: balanced
- Markdown/TXT mirror: byte-identical
- Section 33 records the small CPU training pilot and makes its source, sample size, results and limitations explicit.

## Machine-readable assets and data ingestion
- JSON Schemas: 6; all parse and meta-validate as Draft 2020-12 (experiment manifest, candidate profiles, benchmark contract, KV matrix, implementation roadmap, metric definitions).
- YAML contracts/inventories: candidate profiles, benchmark contract, KV policy matrix, G0–G13 implementation roadmap, metric definitions and intentionally incomplete experiment template.
- Run-manifest schema requires immutable specification/source/data/hardware/software/model/evaluation/artifact identity; training, inference, P7, P8 and P9 requirements are conditional.
- Benchmark-contract paths are canonical dot-separated paths aligned with the manifest schema. Dataset role/use requirements are cross-checked against the semantic validator.
- Metric records declare `value_kind`; ratio metrics carry raw numerator/denominator, per-replicate values and optional confidence intervals. P9 ratios must match counts; zero-denominator values are `undefined`.
- P8 supports an `unverified` state for failed/not-run fidelity evidence; exact and bounded claims require applicable evidence.
- Dataset exporter requires a 40-character immutable Hugging Face revision, supports streaming iterable export, writes train/validation/test split hashes and a manifest SHA-256 sidecar, and leaves rights pending unless explicitly reviewed.
- Dataset loader verifies sidecar/split hashes and byte counts, rejects mutable revisions and path traversal, and fails closed on pending rights unless an explicit exploratory override is set.
- The online `datasets.load_dataset` operation itself was not executed in this environment; offline exporter/loader behaviour was tested separately.

## Static package validator
**Expected clean-package preflight: 46 checks passed; 0 errors.** Checks cover:
- Markdown/TXT equality, section/subsection numbering, code-fence balance and reference definition/use coverage.
- All six JSON Schemas and all five normative YAML contracts/inventories.
- Canonical benchmark field paths, dataset role/use policy, conditional field groups and removal of obsolete accelerator-only aliases.
- P0–P9 profile completeness, KV-cache policy inventory/composition, G0–G13 roadmap dependency order and metric-definition coverage.
- Intentional rejection of the incomplete template.
- Exact SHA-256 coverage for every distributable package file.
- ZIP integrity for the original idea compilation and v3, v7 and v8 research archives.

Invoke with:
```bash
python validate_architecture_package.py
```

## Automated regression tests
- Architecture/run-manifest contract suite: **27 tests passed**.
- Dataset exporter/loader suite: **17 tests passed**. Includes streaming iterables, full immutable revision enforcement, license-statement gating, allowed-use verification, pending-rights fail-closed behaviour, explicit exploratory override, manifest/split tamper detection, path traversal rejection and missing-file/sidecar cases.

Invoke with:
```bash
python -m unittest -v test_validate_architecture_package.py
python -m unittest -v test_dataset_pipeline.py
```

## Training feasibility and usefulness pilot
Six complete miniature training runs completed: P0, P1 and P1-LA, each at two seeds, 500 steps and 64,000 sampled training characters per run. The environment was CPU-only (`torch 2.10.0+cpu`, Python 3.13.5, one PyTorch thread). Dataset: a 29,926-byte excerpt from the online Tiny Shakespeare GitHub source; 80/10/10 local sequential split, not the full source or official Hugging Face splits. Source rights review is pending.

| Candidate | Mean test NLL (nats/char) ± sample SD | Mean top-1 | Mean train seconds | Mean inference chars/s |
|---|---:|---:|---:|---:|
| P0 dense attention | 2.7708 ± 0.0444 | 25.62% | 2.06 | 270,437 |
| P1 simplified GDN-2-inspired + attention | 2.7411 ± 0.0323 | 26.88% | 7.66 | 94,097 |
| P1-LA local + global attention | 2.7598 ± 0.0451 | 25.89% | 1.90 | 269,015 |
| Smoothed bigram | 2.4318 | 27.45% | — | — |

P1 is 1.07% better in mean NLL than P0 and +1.26 top-1 points, but 3.73× slower to train and 2.87× lower in naive CPU inference throughput. Bigram outperforms all three neural candidates. The result establishes toy-scale trainability, not usefulness or architecture superiority. Generated samples are still mostly character fragments.

Checks:
- GDN-2-inspired rank-one update vs explicit matrix equation: max float64 absolute error `2.22e-16`.
- No detected future-token leakage at positions 1, 8, 16 and 31.
- Exact request-local state reset after an intervening input.
- All trainable parameters received gradients on first step; losses/gradient norms finite for all 500 steps.
- Training loss declined in every completed run.
- Repeated seed `20261118` produced identical metrics and checkpoint SHA-256 values. A later repeat also captured the exact trainer, exporter and dataset-manifest SHA-256 values at run time and matched the archived seed metrics/checkpoints; this does not retroactively identify the original six runs.
- Hash-verified local split bundle completed a separate one-step P0 loader smoke test.
- `verify_pilot_artifacts.py`: **137 checks passed**, including a source-identified seed repeat and exact trainer/exporter/dataset-manifest digest validation.

The P1 reference is not an official GDN-2 reproduction; it lacks the fused Triton/chunkwise-WY implementation. The original six runs did not capture the training-script SHA-256 at launch, so their historical script-byte identity remains unknown. A later source-identified repeat captured trainer/exporter/manifest digests and matched the seed-20261118 metrics/checkpoint hashes bit-for-bit. Checkpoint and results hashes are preserved. The full 1.11 MB online dataset, live Hugging Face download, GPU kernels, and commercial rights were not verified. Hugging Face Jobs returned HTTP 402 Payment Required; no remote paid job was created.

Full methods, results, provenance and limitations are in `TRAINING_PILOT_REPORT_v9.md` and Section 33 of the main specification.

## Integrity
- `SHA256SUMS_v9.txt` covers 112 distributable files exactly once and excludes itself.
- The deliverable archive is CRC-tested and independently re-extracted for a second validation pass.
- Historical v3–v8 specifications, audits and reports, original v2 text and IDEAS20 archive, and previous v3/v7/v8 research packages are preserved.

## Explicit non-validation
No frontier-scale LLM was trained. No full online Hugging Face train/validation/test corpus was downloaded. No commercial training rights were adjudicated. No CUDA/GPU kernel was compiled or benchmarked. No end-to-end serving, MTP, diffusion sampler or speculative-verification implementation was executed. No exact target-distribution preservation proof or contamination audit was performed. A toy CPU model, valid run manifest, passing exporter tests and integrity checks do not establish that the proposed architecture is SOTA or useful for general LLM workloads.
