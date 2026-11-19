# LLM Architecture Audit — v5 to v6

**Review date:** 17 November 2026  
**Baseline:** `SOTA_LLM_Architecture_v5.md`  
**Result:** v6 research synthesis and implementation contracts; no model training or benchmark run is claimed.

## 1. Why v6 was necessary

The v5 taxonomy had a consequential omission: it did not yet account for DeepSeek-V4's compressed historical-attention stack and mHC residual transport. It also underrepresented the multi-bank recurrent-memory family and attention-layer MoE. These are not small terminology additions; they introduce separate experiment axes that change the implementation and evaluation matrix.

## 2. New research incorporated

| Mechanism | Source-grounded finding | v6 disposition |
|---|---|---|
| DeepSeek-V4 | Official model card and Hugging Face implementation describe CSA, HCA, sliding attention, mHC, Muon, MTP and hash-MoE bootstrap. The model card reports 1M context, Pro 1.6T/49B active, Flash 285B/13B active. The arXiv preview states 284B for Flash; the discrepancy is retained. | Use as a component-composition reference, not a monolithic blueprint. [R80–R82] |
| CSA | Overlapping low-ratio compression plus indexer selection over compressed entries. | Separate historical representation/compression from the selector axis. [R80–R82] |
| HCA | Higher-ratio, non-overlapping compression followed by dense attention over the compressed pool; no indexer. | Separate from CSA, DSA and cache eviction. [R80–R82] |
| mHC | Multi-stream residual transport with a doubly-stochastic residual mixing matrix; ICML 2026 published paper and DeepSeek-V4 adoption. | Separate residual candidate alongside standard residual, AttnRes and Gated Residual. [R80, R83] |
| MoM | Multiple independent recurrent states, routed by token; ICLR 2026 reports benefits for recall-intensive tasks while retaining the paper's linear-complexity formulation. | Add bank count, routing and state-memory ablations inside recurrent-state plane. [R84] |
| SwitchHead | Attention-layer MoE; NeurIPS 2024 reports small-scale results at 262M parameters. | Add attention-MoE as a different axis from FFN-MoE; no frontier-scale claim. [R85] |
| mHC stream analysis | Follow-up work reports depth-dependent stream use in DeepSeek-V4-Flash. | Add early-vs-late placement experiment; do not generalize the result. [R86] |

## 3. Corrections and precision improvements

1. **Do not equate compression and sparsity.** CSA creates a compressed historical representation and then selects compressed entries. HCA compresses more heavily and attends densely over the compressed pool. DSA, QSA and DHSA remain separate selector families.
2. **Do not equate KV eviction and model-side compression.** Compression changes representation; selection chooses query-visible entries; retention preserves resident entries; eviction deletes entries; offload changes storage placement.
3. **Make compressed-cache causality explicit.** v6 requires source-span metadata, compressor revision, cache version and an `available_after_position`. A compressed representation must not expose future source tokens. Tests cover block completion, overlap, partial windows, restore, rollback and stale cache invalidation.
4. **Treat MoM as recurrent state, not an extra application memory plane.** Its memory cost scales with bank count; its router can starve/collapse banks or increase interference.
5. **Treat attention-MoE and FFN-MoE separately.** Their dispatch units, kernels, communication and correctness risks differ.
6. **Treat mHC as its own residual transport topology.** The Birkhoff constraint includes identity as a feasible matrix, not a guarantee that the learned residual mixer is identity. The high-level tensor equation is explicitly marked as an interface abstraction, not a verbatim reproduction of the paper's exact parameterization.
7. **Record additional DeepSeek-V4 attention features separately.** Shared K=V MQA, partial RoPE, learnable attention sinks, grouped low-rank output projection and a local K/V branch are candidates, not hidden defaults.
8. **Retain model-card/paper parameter discrepancy.** Flash is 284B in the arXiv preview and 285B in the official model card; v6 shows both.

## 4. New implementation contracts

- Multi-stream mHC contract with stream shape, residual matrix constraints, projection checks, and layer-placement ablations.
- Compressed-cache entry schema with causal availability, compressor version and source-span invalidation.
- MoM multi-bank state contract with bank policy, router revision, state schema, reset/fork/release and bank-health tests.
- Attention-MoE contract with expert unit, router granularity, dispatch, capacity, merge order and telemetry.
- Token-ID hash-router bootstrap contract bound to tokenizer and checkpoint hashes.

## 5. Experimental plan changes

- **P1:** introduce MoM only as one-state versus multi-bank topology with a fixed recurrent update operator.
- **P2:** add CSA and HCA as compressed-attention variants, separate from DSA/QSA/DHSA selection.
- **P3:** add SwitchHead attention-MoE and hash-router bootstrap as independent, optional ablations.
- **P4:** evaluate mHC independently from AttnRes and Gated Residual; include early/late placement.

No candidate is enabled by default. The baseline remains dense autoregressive Transformer, followed by isolated operator and topology experiments.

## 6. Validation and limits

Static checks cover reference consistency, heading numbering, diagram node uniqueness, YAML parsing, Markdown/TXT equality, archive integrity and checksum generation. They do not validate neural computation, CUDA kernels, training convergence or comparative quality.

The execution environment is CPU-only (`torch 2.10.0+cpu`, CUDA unavailable). No LLM has been trained, no GPU kernel has been benchmarked, and no empirical superiority claim is made. Hardware, training budget, data manifest and target workload remain deployment/experiment inputs that must be declared before numerical architecture dimensions are fixed.
