# LLM Architecture v4 — Research and Implementation Contract

**Review date:** 17 November 2026  
**Status:** Research specification with implementation contracts; combined model not trained or benchmark-validated.

## Purpose

This package contains an evidence-grounded v4 architecture specification, the original v2-to-v3 audit, a second-pass v3-to-v4 audit, the original supplied materials and machine-readable templates for an eventual controlled experiment. It is not a claim that a new combined model has already beaten existing models.

The scope includes Transformer and attention variants, SSM/recurrent/fast-weight models, hybrids, sparse and indexed attention, MoE, residual transport, learned/external memory, adaptive compute, diffusion LMs, multimodal systems, training/post-training and inference/serving. It cannot enumerate proprietary designs or guarantee coverage of every public paper.

## Read in this order

1. `SOTA_LLM_Architecture_v4.md` — current specification.
2. `LLM_Architecture_Audit_v3_to_v4.md` — second-pass audit and corrections.
3. `LLM_Architecture_Audit_v2_to_v3.md` — original v2 and IDEAS20 disposition.
4. `architecture-profile-v4.yaml` — architecture config template; nulls are unresolved blockers, not defaults.
5. `benchmark-contract-v4.yaml` — machine-readable experimental contract template.
6. `VALIDATION_REPORT.md` — static document/package checks and limitations.
7. `CHANGELOG_v3_to_v4.md` and `SHA256SUMS.txt`.

## Important v4 decisions

- GDN-2 is a leading recurrent candidate; Muir is a high-priority challenger, not a proven winner.
- A 3:1 recurrent/global-attention schedule is one pilot point, not a universal optimum.
- DSA token-level selection and QSA micro-block selection are separate mechanisms. MLI is an emerging DSA serving optimization, not a new model architecture.
- Four memory planes are kept separate: recurrent state, attention KV cache, learned neural memory and durable external application memory.
- Sparse attention can reduce attended positions without automatically reducing KV storage or indexer scan cost.
- FlashAttention-4 is a hardware-specific kernel choice for Blackwell-class systems, not a model-level architecture change.
- MoE, residual redesign, MTP, learned memory and test-time training are staged ablations, not mandatory components.
- Evaluation requires matched-data and matched-compute comparisons, private holdouts, contamination review, real systems measurements and regression tests.

## Source accounting

The IDEAS20 index describes 160 numbered entries: 140 standalone numbered idea entries (batches 01–06 and 08) plus 20 Sentinel mappings that map selected prior ideas. The mapping rows are not additional independent proposals. The original v2 text and original IDEAS20 ZIP are preserved under `inputs/`.

## Remaining work

The document-only design and second-pass integrity corrections are complete. The remaining work is empirical and requires explicit target model/active-parameter budget, accelerator topology, training compute, legally usable data, target languages/domains, serving workload and risk thresholds. Implement P0, then P1; do not add later profiles unless their measured marginal value is positive.
