# Prior-art review: dynamic compute and local n-gram memory

**Review date:** 19 November 2026
**Scope:** the mechanisms described in the v10 architecture document, the 18–19 November dynamic-computation session log, and the v9 miniature pilot.
**Evidence policy:** paper results below are author-reported results from the cited publications/preprints. They are not independent replications by this repository. The V11 source and raw run artifacts remain absent from this repository; the session-log numbers remain **session-reported**.

## Executive assessment

Two research directions in the current session notes have substantial prior art:

1. **Token-wise adaptive depth / residual skipping is established prior art.** A 2026 paper explicitly names a residual-update gate **Token-Selective Attention (TSA)** and reports character-level language-model experiments. FlexiDepth, GateSkip, ANIRA and SafeDepth provide additional, materially different adaptive-depth designs and evaluation lessons. The repository's earlier use of “TSA” is now ambiguous and must not be presented as a novel mechanism or as an uncontested name.
2. **Hashed pair/n-gram memory with gated residual injection is established prior art.** SCONE, Engram and several 2026 follow-ups already cover n-gram lookup, context-aware gating, residual injection, table sharing/factorization, incremental attachment and low-bit storage. The repository's pair/N-gram result can remain a promising reproduction target, but the broad concept is not novel on the evidence currently available.

This does **not** make the project redundant. It changes the defensible research question. The value now lies in careful reproduction, smaller or more efficient implementations, fair matched controls, better causal/difficulty measurements, robust safety, deployment-realistic latency, and a result that survives out-of-domain and unseen-complexity tests. The next project milestone should be a reproducible, rights-cleared comparison harness—not a novelty claim or scale-up run.

## 1. Dynamic computation and adaptive depth

### 1.1 TSA name and mechanism collision

Ahmed Abdelmuniem Abdalla Mohammed's *Adaptive Computation Depth via Learned Token Routing in Transformers* (arXiv:2605.05222, submitted 18 April 2026) explicitly introduces **Token-Selective Attention (TSA)**. It inserts a lightweight two-layer MLP gate after each block except the stem; the gate softly scales the next block's residual update during training. The author reports about 1.7% parameter overhead and 14–23% fewer token-layer operations on Tiny Shakespeare/enwik8 with less than 0.5% quality loss. Its sparse inference path gathers/scatters selected FFN tokens while keeping attention dense. The paper also reports that token-operation reduction does not imply a useful batch-one speedup: synchronization and gather/scatter overhead can erase the arithmetic savings; the reported break-even behaviour depends on batch size and hardware. These are paper-reported claims, not results reproduced here. [Paper](https://arxiv.org/abs/2605.05222).

**Repository consequence:** the session log's “TSA” label describes a similar broad class of per-token, per-layer skip decision. Do not claim that token-selective residual gating itself is new. In future code, configs and plots, use **`session_tsa`** only when referring to the mechanism previously discussed in this project's session, and use **`tsa_2605`** for the paper's method. Before making any implementation comparison, write a mathematical/operator-level diff: gate placement, gate input, soft training path, hard inference rule, which sub-operations are skipped, treatment of attention/KV state, routing budget, and threshold calibration. A shared acronym or high-level similarity is not proof that the implementations are identical.

### 1.2 Other directly relevant adaptive-depth work

| Work | Relevant mechanism / evidence | Implication for this project |
|---|---|---|
| [FlexiDepth — Adaptive Layer-skipping in Pre-trained LLMs](https://arxiv.org/abs/2503.23798), 2025 | Per-token, per-layer router plus adapter for pretrained LLMs; authors report skipping 8 of 32 Llama-3-8B layers while preserving their reported benchmark performance. | Compare against adapter-assisted skip paths, not only a dense baseline. Treat the result as a paper claim, not a transferable guarantee. |
| [GateSkip — What Layers When](https://openreview.net/forum?id=56aa4f45f74423431c41dd1c6a97a523330e36ba), ICLR 2026 | Sigmoid-linear gates on Attention and MLP residual branches; inference uses per-layer token budgets and quantile selection. The paper reports Llama-1B end-to-end latency/throughput gains in an optimized vLLM setup, alongside theoretical FLOP savings. | Separate residual gate learning from sparse execution. Compare both Attention-branch and MLP-branch skipping and include a real serving runtime. Results on different hardware/batches must not be compared as if directly portable. |
| [ANIRA — Understanding Dynamic Compute Allocation in Recurrent Transformers](https://proceedings.mlr.press/v306/moosa26a.html), ICML 2026 | A unified recurrent Transformer framework evaluates compute allocation on complexity-controlled algorithmic/synthetic tasks. The authors report that allocation can align with task complexity, but this does not ensure generalization to unseen input sizes; early decisions use structural cues, while online halting more closely tracks execution state. | Add parameterized tasks with held-out difficulty/length ranges. Measure whether routing tracks causal progress rather than shortcuts such as position, token identity or sequence length. |
| [SafeDepth — Safety-Aware Token-Level Adaptive Computation](https://arxiv.org/abs/2610.00815), submitted 30 September 2026 | Reports that token-level skipping can increase harmful-response rates relative to reference backbones; effects depend on layers skipped and whether skipping happens during prompt processing or answer generation. | Quality-only adaptive routing is insufficient. Safety must be a first-class acceptance constraint, separated by prompt-ingestion and answer-generation phases. |
| [Survey of Dynamic Token Computation](https://doi.org/10.1109/ACCESS.2026.3720106), IEEE Access, 4 August 2026 | Organizes pruning, merging, routing and adaptive-depth methods; emphasizes decision stability, budget reliability, routing overhead, and whether FLOP reduction becomes actual latency reduction. | Adopt decision-stability and budget-reliability measurements, not only accuracy-versus-nominal-operations charts. |

### 1.3 What is still a defensible project contribution?

The session's rejected sequence-global top-k design remains invalid for causal generation when future positions can change an earlier position's execution. Preserve that rejection. But merely changing from global top-k to per-token causal gating is not sufficient novelty: per-token gating appears in TSA, FlexiDepth, GateSkip and ANIRA-family approaches.

Potentially differentiating work must be demonstrated, not asserted. Examples of testable hypotheses include:

- a routing controller whose decision is explicitly conditioned on measured marginal value of compute, and whose calibration/regret remains reliable across domains and unseen difficulty levels;
- a causal routing method with lower total runtime overhead than existing gather/scatter or padding-heavy methods on a declared target runtime;
- a policy that is robust to small perturbations and input-distribution shift, with explicit compute-budget guarantees;
- a safety-aware allocation objective that preserves refusal and harmful-request detection across prompt and generation phases;
- an integration of adaptive depth with local memory where the combined system beats **both** mechanisms independently under matched active FLOPs, total parameters, memory and latency.

Combining prior mechanisms is an engineering hypothesis, not automatically a scientific contribution. The combination must provide a measurable interaction benefit and must not merely add independent parameter capacity.

## 2. Local n-gram memory and residual adapters

### 2.1 Prior art already covers the broad idea

| Work | Relevant design / reported evidence | Implication |
|---|---|---|
| [SCONE — Scaling Embedding Layers in Language Models](https://arxiv.org/abs/2502.01637), 2025 | Scales embedding capacity using frequent n-gram embeddings and contextualized token variants. | N-gram embeddings as a language-model capacity axis predate this project. |
| [Engram — Conditional Memory via Scalable Lookup](https://arxiv.org/abs/2601.07372), 2026 | Hashed suffix n-gram retrieval, multiple hash heads, context-aware query/key gating and residual injection into selected layers; proposes host-memory lookup as a separate sparsity axis. | Hashed n-gram memory plus a gated residual path is directly overlapping prior art. |
| [Incremental N-gram Embeddings](https://arxiv.org/abs/2611.20415), submitted 16 November 2026 | Attaches an Engram-style branch to a pretrained Qwen3-0.6B model; the branch uses hashed n-gram lookup, a projected vector and context-aware gate, then additively injects it into hidden states. The paper reports a 10.46-point average gain on its telecom QA task. | Even incremental attachment and domain adaptation of an n-gram residual branch are now directly studied. The paper's methodology selected hyperparameters using test-set performance; this repository must not copy that leakage. |
| [Chorus — Shared N-gram Embeddings with Layer-wise Adaptation](https://arxiv.org/abs/2611.23386), submitted 17 November 2026 | Shares n-gram tables across layers and uses layer-specific hidden-state rescaling/projections. Authors report matching quality with 87% smaller tables than compared methods, with experiments up to a 133M-parameter backbone. | Table sharing and layer adaptation are also active prior art; table footprint and per-layer adaptation need explicit controls. |
| [Low-Bit N-Gram Embeddings](https://arxiv.org/abs/2610.31646), submitted 22 October 2026 | Studies quantized n-gram table storage; authors report 4-bit tables retain most of their validation-loss improvement with substantially lower table memory. | Memory/precision trade-offs are another established axis, not a novel claim. |

The Engram paper describes the core operator explicitly: hashed n-gram retrieval, context-aware gating, causal local refinement, and residual addition. The Incremental N-gram paper's forward path likewise describes an Engram-derived branch with a projected and gated additive residual. The current session's “pair/N-gram residual representation” overlaps at this level. A narrower local pair adapter could still differ in its parameterization, training regime, tokenizer/domain, resource envelope or measured behaviour; that difference needs a precise specification and ablation.

### 2.2 Experimental implications

Keep the pair/N-gram adapter as the first accuracy hypothesis to reproduce because the session log records a promising three-seed comparison against a nearly parameter-matched wider FFN. However:

- the V11 source/checkpoints and raw manifests are absent; the reported paired mean NLL difference (−0.1672) and top-1 difference (+5.99 percentage points) remain session-reported, not independently verified;
- the reported comparison covers one small character corpus, not broad subword language modelling;
- the existing v9 test shows a smoothed statistical bigram baseline outperforming the neural candidates on its tiny test excerpt;
- a dense vocabulary-squared pair table is not scalable;
- Engram-style hashed tables can collide, and collision handling/gating must be measured;
- vocabulary size, n-gram order, table size, layer placement, precision and optimizer memory are coupled hyperparameters.

The reproduction must compare, at minimum:

1. dense causal Transformer baseline;
2. the project's exact pair/N-gram adapter;
3. parameter-matched wider FFN;
4. matched parameter-count embedding/adapter control without local pair keys;
5. smoothed bigram and a stronger classical n-gram baseline where feasible;
6. a hashed n-gram lookup with context-aware gate (Engram-like reference implementation, independently implemented; do not copy code without a license review);
7. a table-sharing or factorized variant if the main branch shows a real gain;
8. quantized table storage only after the full-precision operator is validated.

Control the total parameter count, activated parameters/FLOPs, optimizer state, training tokens, batch order, sequence length, data split and wall-clock budget. Report memory separately for model weights, embedding tables, optimizer states and activations. Count hash collisions and test the impact of table-size scaling. A lookup that avoids compute but uses a large table is a **compute/memory trade-off**, not “free” capacity.

The Incremental N-gram paper states that it selected the learning-rate, vocabulary-size and injection-position configuration using test-set performance. That is a test-set-selection leakage risk. Our contract remains stricter: hyperparameter and checkpoint selection on validation only, followed by one sealed test evaluation. Record this as a methodological distinction, not as an accusation about the paper's overall result.

## 3. Updated novelty and evidence policy

Do not write “first”, “novel”, “unprecedented” or “state of the art” based on this project's session notes. For every new proposed mechanism:

1. search by both exact name and operator-level description;
2. compare the equations, insertion points, inputs, state, routing granularity and inference execution;
3. cite the closest prior work and document what differs;
4. state whether the evidence is paper-reported, artifact-verified, independently reproduced, session-reported or speculative;
5. keep negative and invalid results; do not choose the best-looking run retrospectively;
6. publish configs, dataset revision/hash, code hash, environment, seeds and per-seed metrics.

A preprint is useful prior art but its numerical claims are not independently validated by citation alone. Conversely, overlap in an abstract concept does not prove exact implementation equivalence. Keep the comparison precise.

## 4. Updated experiment acceptance gates

### Dynamic computation

- **Causality:** prefix perturbation tests prove that changing future tokens cannot change earlier logits or earlier route decisions.
- **Decision validity:** route decisions are logged by layer/token and compared with the actually executed operations.
- **Difficulty alignment:** use synthetic/algorithmic tasks with predeclared complexity knobs and held-out lengths; control for length, position, token identity and formatting cues.
- **Stability:** report same-input rerun agreement, decision changes under small semantic-preserving perturbations, route entropy, per-token budget variance and worst-case budget overshoot.
- **Safety:** evaluate harmful-request recognition, refusal quality, false refusals and harmful completions separately for prompt-prefill and answer-generation routing; include layer-restoration interventions where practical.
- **Systems:** report batch-one and batched latency, prefill/decode separately, active FLOPs, memory, kernel count, gather/scatter/packing overhead and energy when measurable.
- **Controls:** dense, early-exit, ACT, TSA-2605-like gate, GateSkip-like residual gate and adapter-assisted skip controls as implementation budget permits.

### Local n-gram memory

- **Utility:** paired per-seed deltas against the statistical n-gram baseline and matched neural controls.
- **Generalization:** second corpus/domain and a subword tokenizer; report long-tail and unseen n-gram behaviour.
- **Memory:** table bytes, optimizer-state bytes, index/hash overhead, collision rate and cache hit rate.
- **Robustness:** context mismatch, punctuation/case/normalization, rare token, adversarial collision and out-of-vocabulary handling.
- **Selection:** validation-only selection, one sealed test evaluation, no test-driven hyperparameter search.

## 5. Reference list

1. Mohammed, A. A. A. (2026). [Adaptive Computation Depth via Learned Token Routing in Transformers](https://arxiv.org/abs/2605.05222). arXiv:2605.05222.
2. Luo, X., Wang, W., & Yan, X. (2025). [Adaptive Layer-skipping in Pre-trained LLMs](https://arxiv.org/abs/2503.23798). arXiv:2503.23798.
3. Laitenberger, F., Kopiczko, D., Snoek, C. G. M., & Asano, Y. M. (2026). [What Layers When: Learning to Skip Compute in LLMs with Residual Gates](https://openreview.net/forum?id=56aa4f45f74423431c41dd1c6a97a523330e36ba). ICLR 2026.
4. Moosa, I. M., Lohit, S., Wang, Y., Chatterjee, M., & Yin, W. (2026). [Understanding Dynamic Compute Allocation in Recurrent Transformers](https://proceedings.mlr.press/v306/moosa26a.html). ICML 2026, PMLR 306.
5. Li, N., & Harris, I. G. (2026). [SafeDepth: Safety-Aware Token-Level Adaptive Computation](https://arxiv.org/abs/2610.00815). arXiv:2610.00815.
6. [A Survey of Dynamic Token Computation in Transformers: Taxonomy, Stability, and Budget-Aware Evaluation](https://doi.org/10.1109/ACCESS.2026.3720106). IEEE Access 14 (2026), 119498–119537.
7. Yu, D. et al. (2025). [Scaling Embedding Layers in Language Models](https://arxiv.org/abs/2502.01637). arXiv:2502.01637.
8. Cheng, X. et al. (2026). [Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models](https://arxiv.org/abs/2601.07372). arXiv:2601.07372.
9. Hu, S. et al. (2026). [Augmenting Pretrained Language Models with N-gram Embeddings through Incremental Training](https://arxiv.org/abs/2611.20415). arXiv:2611.20415.
10. Chung, S., & Du, W. (2026). [Chorus: Shared N-gram Embeddings with Layer-wise Adaptation](https://arxiv.org/abs/2611.23386). arXiv:2611.23386.
11. Chung, S., & Du, W. (2026). [Low-Bit N-Gram Embeddings for Language Models](https://arxiv.org/abs/2610.31646). arXiv:2610.31646.
