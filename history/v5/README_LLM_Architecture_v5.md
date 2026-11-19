# LLM Architecture v5 — Research and Implementation Contract

**Review date:** 17 November 2026  
**Status:** Research specification; no combined model trained or benchmark-validated.

## Purpose

This package contains an evidence-grounded v5 architecture specification, the v2-to-v3 and v3-to-v4 audit trail, a new v4-to-v5 audit, the supplied source materials and machine-readable experiment templates. It is not a claim that a new model has already outperformed existing systems.

## Read in this order

1. `SOTA_LLM_Architecture_v5.md` — current specification.
2. `LLM_Architecture_Audit_v4_to_v5.md` — current third-pass findings and corrections.
3. `LLM_Architecture_Audit_v3_to_v4.md` — second-pass audit.
4. `LLM_Architecture_Audit_v2_to_v3.md` — original v2 and IDEAS20 disposition.
5. `architecture-profile-v5.yaml` — configuration family; nulls are blockers, not defaults.
6. `benchmark-contract-v5.yaml` — benchmark and statistical contract template.
7. `operator-contract-v5.yaml` — tensor/state/ownership/lifecycle interface contract.
8. `experiment-manifest-template-v5.yaml` — per-run provenance template.
9. `VALIDATION_REPORT.md`, `CHANGELOG_v4_to_v5.md`, `SHA256SUMS.txt`.

## Major v5 decisions

- GDN-2 remains a leading recurrent candidate; Muir is a preprint challenger. Neither is declared the winner.
- GQA is a simple attention baseline; MLA is an alternative representation.
- DSA token top-k, QSA micro-block selection and DHSA hierarchical chunk-to-token sparsification are distinct mechanisms.
- MLI is an optional DSA-compatible serving-graph schedule, not a new sequence mixer.
- Five memory planes are separated: recurrent state, attention KV cache, learned neural memory, bounded request-local editable slots and durable external application memory. The four-tier external-memory organization is a separate taxonomy.
- DiffuMamba creates an independent masked-diffusion challenger (P6); it is not mixed into causal autoregressive training by default.
- MoE, residual redesign, MTP, neural memory, editable slots and TTT are staged ablations, not mandatory components.
- Evaluation requires matched-data/matched-compute views (including tuning cost), paired task sets, confidence intervals, private holdouts, contamination review, real systems measurements and failure-mode regression tests.

## Scope and limits

The taxonomy covers major public architecture families through the review date, but it cannot enumerate all proprietary designs or every public paper. The new components' published results are limited to their own experimental conditions. CPU-only document validation does not establish numerical operator correctness, training stability, performance or model superiority.

## Remaining work

The document-only research and specification pass is complete for this revision. Training requires the target model/active-parameter budget, accelerator topology, compute budget, legally usable data manifest, languages/domains, serving workload, latency/cost targets and risk thresholds. Implement P0 first, then P1; introduce later components only when controlled ablations show positive marginal value. P6 remains an independent challenger.
