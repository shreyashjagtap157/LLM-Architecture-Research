# Technical Audit — v6 to v7

**Review date:** 18 November 2026  
**Baseline:** `SOTA_LLM_Architecture_v6.md`  
**Result:** v7 expands the architecture search space and fixes gaps in architecture-family comparisons, KV-cache semantics, speculative generation and agent safety. It remains a research specification, not a trained model.

## 1. New architectural evidence

### 1.1 Inkling 2 Small was missing as a separate baseline family

v6 included Qwen3.8, DeepSeek-V4, Nemotron 3 Nano and recurrent hybrids, but did not include the just-released Inkling 2 Small composition. This matters because it offers a different trade: local/global attention, block attention residuals, gated attention, short causal convolutions, large sparse MoE and a DSpark drafter, without a GDN/SSM recurrent backbone. v7 adds it as `P1-LA`, an equal-budget control against recurrent `P1`; it does not copy the 594B/17B configuration as an optimal scale.

The official model card exposes 594B total / 17B active parameters and a 1M-token maximum context. The training report distinguishes pretraining stages (H2 at 512K, H3 at 1M) from the RL setup (256K context with compaction for longer trajectories). This is a clarification of different training/evaluation settings, not a false contradiction. The vLLM architecture table provides the concrete 48-layer, 5:1 local/global schedule, 48/8 GQA, 384 experts/top-8, block residual grouping and convolution placement.

### 1.2 Block attention residuals need a separate contract

v6 compared AttnRes, Gated Residual and mHC, but did not explicitly distinguish Inkling 2's blockwise residual summaries. v7 separates the mechanism and adds block-boundary/causality/restore checks. Short causal convolutions are also separated from long-range recurrent sequence mixing.

### 1.3 Speculative decoding had an incomplete candidate set

v6 compared MTP, LayerSkip and external draft models. DSpark adds block-parallel drafting with a Markov/RNN head to restore within-block dependency and a confidence head for acceptance estimation. v7 adds it as a separate pilot. Confidence is not correctness and does not replace the target verifier.

### 1.4 Diffusion path now has a stronger shared-model challenger

v6's P6 was an independent masked-diffusion model. The EMNLP 2026 Nemotron-Labs-Diffusion paper reports a shared 3B/8B/14B family with AR, diffusion and diffusion-draft/AR-verify modes. v7 keeps the independent masked-diffusion track and adds P7 for the tri-mode strategy. Reported GB200 performance is cited only as the paper's setup-specific result.

## 2. KV-cache research gaps

v6 had a broad multi-signal cache manager and compression/selection/retention/eviction/offload separation, but did not incorporate four recent policy families:

- **Seer:** semantic centroids, cluster-level scoring, product quantization and asymmetric lookup.
- **ShotKV/KVFundaBench:** semantic-unit preservation and a benchmark revealing dense-reasoning degradation hidden by sparse retrieval.
- **Distance-KV:** offline retention patterns conditioned on layer, head and relative distance.
- **LiteKV:** recent/heavy-hitter/overview rings plus semantic-shift awareness.

v7 gives each its own experiment, cost terms and failure tests. It expressly prohibits composing all of them before independent ablations because their policies can compete for the same cache budget. LiteKV is only included at abstract-level evidence pending full code/paper reproduction.

## 3. Training and application-safety improvements

The Inkling 2 technical report provides a useful staged-training case study (8.4T H1, 3.8T H2, 320B H3, 34B H4, and a large joint RL phase). v7 includes it as provider-reported evidence, not a generic compute-cost multiplier. The report also documents synchronous tool-call inspection, asynchronous rollout audit and an indirect network-circumvention incident. v7 makes environment-level egress controls, capability enforcement and trace auditing explicit requirements.

## 4. Source and claims discipline

- Inkling 2 context is split into model-card maximum, training stage context and RL context with compaction.
- vLLM's 630 TPS/user result and tri-mode paper's speedups remain setup-specific; no speedup is projected onto the proposed model.
- Distance-KV remains a preprint; LiteKV detail remains provisional at abstract level.
- The v7 design is not declared superior to v6 or to any SOTA model without a controlled implementation/benchmark.

## 5. Remaining blockers

1. Target parameter scale and active-parameter budget are unspecified.
2. Target accelerator, HBM, interconnect, software stack and kernel support are unspecified.
3. Training data, rights/provenance, language/domain mix and compute budget are unspecified.
4. No model training or GPU benchmark has been performed in this CPU-only environment.
5. Full LiteKV paper/code reproduction and independent replication of recent vendor/preprint results remain outstanding.

## 6. Disposition

v7 is the best current specification baseline. The next meaningful work is no longer adding fashionable components: it is to freeze a hardware/data/compute contract, reproduce P0, compare P1 and P1-LA, then run isolated cache policy, residual, MoE, DSpark and generation-mode experiments. A negative or inconclusive ablation is a valid result and should remove complexity rather than trigger another speculative feature.
