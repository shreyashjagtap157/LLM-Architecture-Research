# LLM Architecture v3 — Research Package

**Review date:** 17 November 2026  
**Status:** Research specification and source audit complete; the proposed combined model has not been trained or benchmark-validated.

## Purpose

This package revises the supplied *The Synthesized SOTA LLM Architecture (v2 — with upgrade paths)* using the accompanying IDEAS20 archive and a research review of major publicly documented LLM architecture families. It is intended to be a reproducible design baseline for later implementation and controlled experiments—not a claim that the proposed combination is already state of the art.

The scope covers major public architecture families and representative systems, including Transformer and attention variants, sparse/local/linear attention, SSMs, recurrent and fast-weight models, hybrid sequence mixers, sparse MoE, adaptive compute, neural/external memory, diffusion language models, multimodal models, training/post-training approaches, and inference/serving techniques. It cannot enumerate undisclosed proprietary designs or guarantee coverage of every paper and checkpoint available online.

## Read in this order

1. **`SOTA_LLM_Architecture_v3.md`** — the standalone architecture specification. It defines the evidence labels, model/runtime/application boundaries, design hypothesis, component interfaces, training and inference considerations, experiment plan, acceptance gates, honest limits and research references.
2. **`LLM_Architecture_Audit_v2_to_v3.md`** — the v2 technical audit and detailed disposition of the supplied idea archive. Read this to understand what was corrected, retained, deferred, narrowed or rejected and why.
3. **`VALIDATION_REPORT.md`** — static package-integrity checks and execution limitations.
4. **`SHA256SUMS.txt`** — SHA-256 digests for the source documents and included input files.

The `.txt` versions of the v3 specification and audit are byte-for-byte identical to the Markdown files. They are included for tools that expect a plain-text extension.

## Main architectural decisions

- Treat the system as three related but distinct strata: **neural model**, **inference/serving runtime**, and **agent/application control plane**.
- Retain an adaptive hybrid recurrent/attention backbone as the central hypothesis, but do not assume a universal layer ratio.
- Evaluate Gated DeltaNet-2 as a strong recurrent-path candidate, with Mamba-2/3, Gated DeltaNet/KDA and the recent Muir preprint as comparators. Muir remains exploratory until independently replicated at larger scale.
- Use GQA as a practical attention-cache baseline; evaluate MLA and indexed sparse attention against it on real hardware and retrieval workloads.
- Keep residual-transport candidates (standard residual, Attention Residuals and Gated Residual) as separate ablations before combining them.
- Introduce sparse MoE only after the sequence backbone is stable, and measure network communication and expert imbalance—not only active FLOPs.
- Separate recurrent model state, attention KV cache and durable external memory.
- Use bounded reasoning/backtracking with typed validators; self-consistency is a heuristic, not an oracle.
- Treat MTP, speculative decoding, grammar constraints and semantic/domain validation as distinct mechanisms.
- Require reproducible matched-budget experiments before promoting any proposed component.

## Source accounting

The IDEAS20 index describes 160 numbered entries across eight idea batches. Its Sentinel-specific file explicitly maps 20 of the ideas already present in the other batches. The audit therefore distinguishes **140 standalone numbered idea entries** (batches 01–06 and 08) from **20 Sentinel mappings** (batch 07), rather than counting the mappings as 20 additional independent ideas. Batch 09 is the v2 architecture document, not another idea batch.

The package includes exact copies of the supplied v2 text and IDEAS20 ZIP under `inputs/` so the audit can be revisited without relying on the original conversation attachments.

## What remains

No further document-only architecture decisions are recommended before the experimental contract is defined. The next phase is empirical, and depends on information not supplied with the documents:

- target model size and active-parameter budget;
- accelerator type/count, HBM, interconnect and supported kernels;
- data quantity, domains, language coverage and licensing constraints;
- serving context/output distribution, concurrency and latency/cost objectives;
- capability and risk thresholds.

Once these are fixed, implement the pilot baseline and the staged ablation program in v3. The model should not be described as optimized or superior until matched-quality/cost evidence supports that claim.

## File list

- `SOTA_LLM_Architecture_v3.md`
- `SOTA_LLM_Architecture_v3.txt`
- `LLM_Architecture_Audit_v2_to_v3.md`
- `LLM_Architecture_Audit_v2_to_v3.txt`
- `VALIDATION_REPORT.md`
- `SHA256SUMS.txt`
- `inputs/v2_baseline.txt`
- `inputs/ideas20_full_compilation.zip`
