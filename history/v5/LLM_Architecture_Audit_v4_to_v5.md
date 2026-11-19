# LLM Architecture Audit — v4 to v5
*Third-pass architectural and specification audit, documenting corrections and newly incorporated research.*

**Review date:** 17 November 2026  
**Baseline:** `SOTA_LLM_Architecture_v4.md`  
**Current specification:** `SOTA_LLM_Architecture_v5.md`  
**Scope:** generation-family completeness, sparse-attention taxonomy, memory lifecycle, equations, experiment design and package integrity.

---

## 1. Executive assessment

v4 was substantially more implementable than v3, but its taxonomy still treated diffusion as a family to monitor rather than giving it a fair independent challenger path. Its memory model separated recurrent state, KV cache, learned neural memory and external application memory, but did not explicitly represent a bounded, editable, request-local slot store with write/overwrite/protection semantics. Its sparse-attention matrix grouped DSA and QSA and lacked DHSA as a distinct frozen-backbone inference-time family. The component equations also reused `V_l` for attention values and the feed-forward normalized activation.

v5 corrects these gaps without promoting any newly researched mechanism to a default or claiming that reported results transfer beyond their tested conditions.

## 2. New research and its bounded interpretation

| ID | Work | Evidence class | v5 treatment | Limitation retained |
|---|---|---|---|---|
| R76 | Dynamic Hierarchical Sparse Attention, ICML 2026 | Peer-reviewed experimental work | Added as a separate hierarchical chunk-to-token sparsifier and P2 inference candidate | Reported speed/quality claims are tied to its model, kernels, quantization, tasks and context regime |
| R77 | DiffuMamba, ICML 2026 | Peer-reviewed experimental work | Added a separate P6 masked-diffusion challenger using bidirectional Mamba and hybrid attention variants | Evaluated up to 1.3B; cannot be assumed to win for autoregressive chat, tools or larger scales |
| R78 | Memory-Managed Long-Context Attention, 2026 preprint | Preliminary mechanism study | Added bounded request-local editable memory slots with explicit lifecycle and sparse fallback | Controlled synthetic/small-model results; no claim of frontier-scale open-domain memory or general entity resolution |
| R79 | Towards Next-Generation Architectures of LLMs survey, submitted 9 Nov 2026 | Survey / taxonomy source | Used to cross-check breadth across processing, state management, residuals, adaptive depth and output commitment | Survey supports taxonomy coverage, not independent validation of each design |

## 3. Specification defects corrected

| ID | v4 defect or gap | v5 correction |
|---|---|---|
| V5-01 | `V_l` was used both for attention value cache and normalized FFN input. | Renamed attention cache tensors to `K_l^cache`, `V_l^cache` and FFN activation to `F_l`. |
| V5-02 | The memory taxonomy omitted explicit request-local editable slots. | Expanded to five distinct planes; external working/episodic/semantic/procedural memory remains a four-tier organization inside the fifth plane. |
| V5-03 | DSA and QSA were distinguished, but the taxonomy omitted DHSA. | Added a separate DHSA row, source, implementation notes and P2 experiment. |
| V5-04 | Diffusion appeared mainly as a related family, without a dedicated controlled challenger. | Added P6 parallel masked-diffusion track and dedicated evaluation dimensions; it does not enter the autoregressive promotion dependency chain. |
| V5-05 | The initial configuration did not make the independent diffusion track or editable-memory lifecycle explicit. | Updated pilot sequence, memory plan, configuration YAML and benchmark contract. |
| V5-06 | Sparse selected-set notation was underspecified. | Defined top-k selection over eligible history plus mandatory global/sink and local-window positions, with deduplication and causal masking. |
| V5-07 | Request-local memory had no first-order storage accounting or regression row. | Added slot payload/metadata/index estimate and lifecycle failure tests. |
| V5-08 | Statistical acceptance did not specify paired test sets, CIs or multiple-comparison handling. | Added paired task sets, per-seed reporting, confidence intervals, untouched final holdout and adjustment examples. |
| V5-09 | Matched-compute view did not explicitly amortize architecture-specific tuning. | Added tuning/hyperparameter-search cost to the fair-comparison contract. |
| V5-10 | The previous v4 footer repeated a legacy status paragraph. | Consolidated into one versioned final-status block. |

## 4. New implementation artifacts

- `architecture-profile-v5.yaml` — primary autoregressive profile plus independent masked-diffusion challenger; five memory planes; DSA/QSA/DHSA candidates.
- `benchmark-contract-v5.yaml` — candidate matrix, task/system/reliability/economic metrics, statistics, contamination and acceptance fields.
- `operator-contract-v5.yaml` — input/output/state ownership, cache and indexer versioning, lifecycle operations and hard invariants.
- `experiment-manifest-template-v5.yaml` — per-run provenance and configuration identity.

## 5. Experimental interpretation

P0–P5 remains the autoregressive architecture path. P6 is an independent masked-diffusion challenger: objective, generation process and stopping policy differ, so the evaluation must use both objective-appropriate metrics and shared downstream tasks at matched wall-clock/compute budgets. It is not valid to compare causal next-token perplexity with masked-diffusion loss as if they were identical.

The five memory planes are deliberately separated:

1. model-local recurrent/fast-weight state;
2. attention KV cache;
3. learned neural long-term memory;
4. bounded request-local editable slots;
5. durable external application memory.

The four-tier working/episodic/semantic/procedural organization remains a structure within external application memory; it is not counted as four additional neural memory planes.

## 6. Remaining blockers

v5 is still a research specification. It has not been trained, and no architecture winner can be named. The actual experiment cannot be frozen without a target model/active parameter budget, accelerator topology, compute ceiling, permitted data and licensing conditions, language/domain distribution, serving workload, latency/cost target and safety thresholds. Nulls in YAML templates are deliberate blockers.
