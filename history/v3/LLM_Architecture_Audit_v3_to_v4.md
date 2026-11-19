# LLM Architecture Audit — v3 to v4
*Second-pass audit of the previous v3 specification, its evidence register and package integrity, followed by the v4 corrections.*

**Review date:** 17 November 2026  
**Baseline:** `SOTA_LLM_Architecture_v3.md`  
**Current specification:** `SOTA_LLM_Architecture_v4.md`  
**Scope:** architecture evidence, implementation semantics, memory/state boundaries, systems cost accounting, experiment reproducibility and deliverable integrity.

---

## 1. Executive Assessment

The v3 document was a sound research synthesis but still left too many decisions at prose level. It had a useful candidate backbone and experimental sequence, yet an implementer could still interpret recurrent state, attention cache, sparse attention selection, MoE routing and neural long-term memory differently. It also did not fully distinguish sparse attention's reduced compute from KV storage/scan costs, nor did it define enough evidence and artifact requirements for a fair comparison.

The second pass also found package defects that were not architectural but would undermine trust in the deliverable: a validation report with unevaluated template expressions, an incorrect LatentMoE title, incomplete Nemotron shared-expert accounting and a duplicated section heading.

## 2. Second-Pass Findings

| ID | Finding | Severity | v4 action |
|---|---|---:|---|
| V4-01 | The validation report contains literal `{len(...)}` / `{torch_info}` placeholders. | High | Recomputed real values and replaced the report. |
| V4-02 | The LatentMoE reference title does not match arXiv:2601.18089. | Medium | Corrected title to *Toward Optimal Accuracy per FLOP and Parameter in Mixture of Experts*. |
| V4-03 | Nemotron 3 Nano description omits the 2 shared experts. | High | Clarified 128 routable, 6 selected routed experts and 2 shared experts. |
| V4-04 | A section heading is duplicated. | Low | Removed duplicate. |
| V4-05 | Three memory planes did not fully represent learned neural memory separately from external persistent memory. | High | Defined four planes: recurrent, KV, learned neural and external application memory. |
| V4-06 | Sparse attention was not explicit enough about indexer scan and KV storage remaining costs. | Critical | Added DSA/QSA distinction and a separate selection/storage/eviction cost contract. |
| V4-07 | New DSA serving optimization (MLI, Nov 2026) was absent. | Medium | Added as emerging serving-schedule preprint, not a new learned architecture or universal optimization. |
| V4-08 | Attention kernel recommendations did not include the Blackwell-specific FlashAttention-4 result. | Medium | Added FA4 as hardware-specific; no universal speedup claim. |
| V4-09 | Tensor dimensions and state ownership were underspecified. | High | Added normative component contracts and request/session isolation invariants. |
| V4-10 | Cost comparisons could conflate active FLOPs with weight residency, HBM, indexer scan and MoE all-to-all. | Critical | Added first-order memory equations and required end-to-end metrics. |
| V4-11 | Evaluation protocol did not make benchmark contamination handling operational enough. | High | Added explicit benchmark provenance, private holdouts, overlap checks and SWE-bench Verified caution. |
| V4-12 | The sequence-operator proposal lacked a staged implementation profile. | High | Added P0–P5 promotion profiles and explicit stop rules. |
| V4-13 | Test-time training was not distinguished from retrieval or durable memory. | Medium | Added optional TTT candidate with strict no-update/retrieval controls and safety limits. |
| V4-14 | Package checksums/manifest could be stale after revision. | High | Regenerated after content freeze and verified against the actual archive. |

## 3. New Research Incorporated

### 3.1 DeepSeek-V3.2 / DSA

The v4 specification adds DeepSeek Sparse Attention as a distinct learned-indexer design. DSA selects historical token positions using query-conditioned index scores. It is not interchangeable with Qwen3.8's micro-block QSA. v4 therefore requires separate ablations and explicit indexer, selection, gather and cache measurements. [R67]

### 3.2 Multi-layer Indexer (MLI)

The 2 November 2026 preprint proposes overlapping one layer's indexer with another layer's attention using a previous layer's selection. Its reported results are specific to DSA-based DeepSeek-V3.2/GLM-5 serving experiments. v4 lists it as a runtime scheduling candidate with cache-version/lag invariants, not as a model architecture or a retraining-free universal speedup. [R68]

### 3.3 FlashAttention-4

MLSys 2026 describes kernel and pipeline co-design for Blackwell hardware. v4 adds it beside FlashAttention-2/3 but makes clear that a kernel result on B200/Blackwell does not transfer automatically to other GPUs or attention shapes. [R69]

### 3.4 Evaluation and contamination

RULER includes multi-needle, tracing and aggregation tasks beyond simple needle retrieval. LongBench v2 emphasizes realistic long-context reasoning. The 2026 SWE-bench Verified audit highlights contamination and grader/test design risk. v4 therefore requires private holdouts, benchmark hashes, task/grader versions, budget parity, contamination checks and separate cost-per-accepted-task reporting. [R70–R74]

### 3.5 Test-time training

The 2026 S-TTT preprint evaluates selecting evidence spans before adapting on long context. v4 includes it as a conditional candidate, with retrieval-only and no-update baselines and explicit risks around irrelevant-span adaptation, interference and privacy. [R75]

## 4. v4 Architecture Specification Changes

1. Added tensor and operator interfaces for the pre-norm block, recurrent mixer, attention, indexed sparse attention, MoE, MTP and semantic validation.
2. Defined state ownership, reset, request isolation, prefix reuse and stale-index protections.
3. Added first-order parameter, KV and recurrent-state memory equations; explicitly noted their limitations.
4. Defined separate measurements for sparse indexer scan, top-k, gather, attention and cache residency.
5. Added fair-comparison protocol for matched-data and matched-compute views.
6. Added model/data/training/checkpoint identity and provenance fields.
7. Added failure-mode regression tests and P0–P5 staged candidate profiles.
8. Kept the model scale, hidden width, head counts, expert counts and compute budget unresolved because no target hardware/deployment envelope was supplied.

## 5. What v4 Still Does Not Claim

- It does not claim every online architecture has been enumerated.
- It does not claim the v4 synthesis has been trained or benchmarked.
- It does not claim Muir, GDN-2, DSA, QSA, MLI or FlashAttention-4 is universally best.
- It does not claim that sparse-attention index selection automatically reduces KV storage.
- It does not claim that public benchmark results are uncontaminated or sufficient on their own.
- It does not prescribe numeric model dimensions or GPU costs without hardware and training constraints.

## 6. Remaining Empirical Dependencies

The next step is implementation, not more speculative components. P0 and P1 should be implemented and validated first, then P2 only if sparse attention is justified by profiling. MoE, residual redesign, MTP, learned memory, TTT and application-level reasoning should each earn promotion through independent ablations.

The blockers are explicit: target model/active parameter budget, accelerator topology, compute allocation, training data/license constraints, target language/domain mix, serving concurrency/latency/cost objectives and required risk thresholds.
