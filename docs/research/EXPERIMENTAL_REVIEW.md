# Experimental review — what is useful, what is not yet useful

## 1. Current candidate ranking

This ranking is a research priority, not a proven architecture leaderboard. The prior-art comparison is in [`PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md`](PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md).

| Candidate | Current judgment | Why |
|---|---|---|
| Dense causal Transformer P0 | Keep as reference baseline | Simple, fast and essential for fair comparisons. |
| Pair/N-gram residual adapter | Highest-priority reproduction hypothesis; **not a novelty claim** | Session-reported win over a nearly parameter-matched wider FFN across three seeds on one small corpus. Raw V11 source is absent. Hashed n-gram lookup, contextual gating and residual injection already have substantial prior art. |
| Local/global attention P1-LA | Keep as architectural control | Near dense baseline in the v9 pilot; not a clear winner. |
| GDN-2-inspired hybrid P1 | Keep as experimental sequence-mixer candidate | Slightly better than P0 on tiny excerpt but slower CPU and worse than bigram control; implementation is not official GDN-2. |
| ACT recurrent refinement | Experimental; reproduce under competence gating | Can reduce abstract steps, but step savings are small/inconsistent and reference CPU latency is worse. Difficulty allocation needs calibration. |
| `session_tsa` per-token residual/skip routing | Experimental; compare to TSA-2605, GateSkip and FlexiDepth | Related mechanisms are already published. Current session's latency and multi-seed quality are not reproducible from available V11 artifacts. |
| Teacher-supervised V.O.C. | Experimental only | Value regression can fit a teacher gap while producing a poor policy. Calibration/regret and actual latency are unresolved. |
| Safety-aware routing | Mandatory acceptance dimension, not an optional later feature | Recent work reports safety regressions from adaptive skipping that depend on layer and inference phase. |
| Global top-k token routing | Reject | Violates causal isolation when future tokens influence earlier route decisions. |
| Entropy as universal halting signal | Reject as universal rule | Session evidence indicates unreliable difficulty calibration; require task- and phase-specific calibration. |

## 2. Main scientific distinction

Architecture correctness, optimization learnability, predictive quality, safety and execution usefulness are separate claims:

1. **Operator correctness:** equations and state updates match the declared definition.
2. **Causal correctness:** future inputs do not affect earlier logits or earlier routing.
3. **Training feasibility:** losses/gradients remain finite and optimization learns.
4. **Predictive usefulness:** held-out metrics beat appropriate statistical and neural baselines.
5. **Safety preservation:** routing does not degrade harmful-request recognition, refusal or benign false-refusal behaviour.
6. **Compute usefulness:** quality/safety are preserved or improved at lower measured FLOPs, memory, latency, throughput or energy.
7. **Stability and budget reliability:** routes are repeatable, perturbation-stable and stay within declared compute budgets.
8. **Scale usefulness:** effects persist on larger models, broader datasets and target hardware.

The v9 pilot establishes several claims in (1)–(3) at toy scale. It does not establish (4) against a statistical bigram control, and it does not establish (5)–(8). The subsequent session reported promising pair/N-gram results and reductions in abstract compute counts, but V11 raw artifacts are missing from this bundle.

## 3. Accuracy and local-memory mechanisms

The pair/N-gram adapter remains worth reproducing because the session record describes a paired gain against a wider FFN. However, the broad architectural idea is not novel: prior work has already studied hashed n-gram retrieval, contextual gates, additive residual injection, layer placement, incremental attachment, shared tables, factorization and quantized table memory.

Required controls:

- dense causal Transformer P0;
- smoothed bigram and stronger classical n-gram baseline;
- exact pair/N-gram session adapter;
- matched wider FFN;
- matched parameter embedding/adapter control without pair keys;
- independently implemented hashed n-gram lookup with context-aware gate;
- factorized/shared table only as separately toggled ablation;
- low-bit storage only after full-precision benefit is established.

A character-level pair table does not transfer directly to a 32k–256k subword vocabulary. Dense pair-feature tables scale quadratically. Scalable implementations need bounded hashes, factorized embeddings or a compact encoder; their collision rate, long-tail performance, table memory and optimizer-state cost must be measured. A small per-token lookup can still incur large memory and bandwidth cost.

The reported V11 N-gram values (paired mean test NLL difference −0.1672 and top-1 accuracy difference +5.99 percentage points) are **session-reported only**. They are not independently reproducible from this repository because the V11 source, checkpoints and run manifests are missing. The v9 artifact-verified pilot's smoothed bigram control beat the neural candidates on its tiny excerpt; the two evidence sets must not be conflated.

## 4. Dynamic computation mechanisms

The earlier session's per-token gating hypothesis overlaps with TSA (2026), GateSkip (ICLR 2026), FlexiDepth and other adaptive-depth work. Preserve the historical session mechanism as `session_tsa` in future labels. Before claiming a distinction, compare equations, gate placement, gate inputs, soft/hard execution, attention/KV treatment, route budgets and sparse kernels against the closest prior implementation.

ACT and token-selective routing can reduce *nominal operations* only if the implementation avoids executing the skipped operations. Python-level dynamic gather/scatter, ragged sequence packing, graph breaks or poor kernel occupancy can cost more than the saved arithmetic. A session-reported CPU slowdown remains a warning, not a serving benchmark.

Required measurements:

- average and percentile active depth per token;
- compute allocated conditioned on task difficulty, length, position and uncertainty calibration;
- quality-versus-compute curves selected on validation;
- batch=1 latency, prefill/decode and realistic batched throughput;
- peak memory, KV/recurrent state and allocator overhead;
- kernel launch count, padding waste, gather/scatter and synchronization cost;
- energy per generated token where measurement is available;
- same-input route repeatability, perturbation sensitivity and budget overshoot.

A controller that saves 30% of abstract refinement steps but runs 23% slower on CPU is not an inference optimization for that implementation. It may be useful on a different accelerator with fused kernels, but this must be measured.

## 5. Safety, controller quality and ACT details

Token-wise skipping can affect harmful request recognition and refusal differently depending on which layers are skipped and whether the token is in prompt prefill or generated answer. Full-depth quality parity is not proof of safety parity. Track harmful-response rate, refusal quality and benign false-refusal separately for both phases against a full-depth reference.

ACT should implement one mathematically consistent halting formulation. The additive halting mass, remainder rule, ponder cost and gradients must be derived from the same formulation. A detached halting-mass accumulator must not sever the gradient to the halt controller.

Training should start from a competent fixed-depth model or use an explicit competence gate. A fixed number of warm-up steps is not enough if the teacher's validation performance is still poor. Threshold selection must use the adaptive execution path and validation data only.

For value-of-computation, prediction accuracy of the value head is not policy quality. Measure:

- calibration of predicted marginal gain;
- action regret relative to the exhaustive full-depth oracle;
- ranking of examples by marginal value of extra compute;
- policy quality at fixed compute budgets;
- generalization to unseen lengths/complexities;
- end-to-end serving cost.

## 6. Minimum evidence before architecture promotion

- At least five independent seeds for early promotion screens.
- A second independently sourced dataset/domain.
- Immutable dataset revisions and use-rights review.
- Parameter-, token-, FLOP- and optimizer-budget matching where relevant.
- Pre-registered primary metrics and quality/safety margins.
- Validation-only architecture/threshold selection and a single sealed test evaluation.
- Confidence intervals and paired per-seed comparisons.
- Target-hardware profiling using optimized kernels.
- Reproducible source hashes, configs, checkpoints, logs and environment manifests.
- Route stability, perturbation robustness and compute-budget reliability.

## 7. Official LLM training gate

Do not proceed to an expensive official LLM training run merely because example models converge or a mechanism has an interesting paper result. First show that the chosen implementation provides an incremental benefit over strong prior-art controls on a rights-cleared subword corpus, preserves safety, and improves the quality/efficiency frontier on a second domain. Train small examples first, then a larger pilot, then scale only the components that survive independent ablation.
