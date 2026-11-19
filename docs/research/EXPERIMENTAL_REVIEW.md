# Experimental review — what is useful, what is not yet useful

## 1. Current candidate ranking

This ranking is a research priority, not a proven architecture leaderboard.

| Candidate | Current judgment | Why |
|---|---|---|
| Dense causal Transformer P0 | Keep as reference baseline | Simple, fast and essential for fair comparisons. |
| Pair/N-gram residual adapter | Highest-priority accuracy hypothesis | Session-reported win over a nearly parameter-matched wider FFN across three seeds on one small corpus. Must be reproduced from source and tested on subword data. |
| Local/global attention P1-LA | Keep as architectural control | Near dense baseline in the v9 pilot; not a clear winner. |
| GDN-2-inspired hybrid P1 | Keep as experimental sequence-mixer candidate | Slightly better than P0 on tiny excerpt but slower CPU and worse than bigram control; implementation is not official GDN-2. |
| ACT recurrent refinement | Experimental only | Can reduce abstract steps, but step savings are small/inconsistent and reference CPU latency is worse. Difficulty allocation needs calibration. |
| Causal token-selective attention | Experimental only | Causal per-token design passed session tests and reduced token-layer operations in a seed; latency and multi-seed quality are not established. |
| Teacher-supervised V.O.C. | Experimental only | Value regression can fit a teacher gap while producing a poor policy. Calibration/regret and actual latency are unresolved. |
| Global top-k token routing | Reject | Violates causal isolation. |
| Entropy as universal halting signal | Reject as universal rule | Session evidence indicates unreliable difficulty calibration. |

## 2. Main scientific distinction

Architecture correctness, optimization learnability, predictive quality and execution usefulness are separate claims:

1. **Operator correctness:** equations and state updates match reference definitions.
2. **Causal correctness:** future inputs do not affect earlier logits or earlier routing.
3. **Training feasibility:** losses/gradients remain finite and optimization learns.
4. **Predictive usefulness:** held-out metrics beat appropriate baselines.
5. **Compute usefulness:** quality is preserved/improved at lower measured FLOPs, memory, latency, throughput or energy.
6. **Scale usefulness:** effects persist on larger models, broader datasets and target hardware.

The v9 pilot establishes several claims in (1)–(3) at toy scale. It does not establish (4) against a statistical bigram control, and it does not establish (5) or (6). The subsequent session reported promising (4)-like N-gram results and reduced abstract compute counts, but V11 raw artifacts are missing from this bundle.

## 3. Accuracy mechanisms

The pair/N-gram adapter is worth pursuing because it may provide direct local-context information without requiring the main attention stack to relearn every short-range dependency. However:

- A character-level pair table does not transfer directly to a 32k–256k subword vocabulary.
- Dense pair-feature tables can scale quadratically with vocabulary size.
- A scalable implementation should use hashing, low-rank/factorized embeddings, bounded context lengths, or a learned compact local encoder.
- The improvement must survive matched parameter, FLOP, token-budget and optimizer controls.
- A second dataset/domain is mandatory before claiming general accuracy improvement.

## 4. Dynamic computation mechanisms

ACT and token-selective routing can reduce *nominal operations* only if the implementation avoids the skipped operations. Python-level dynamic gather/scatter, ragged sequence packing, graph breaks or poor kernel occupancy can cost more than the saved arithmetic.

Required measurements:

- average and percentile active depth per token;
- compute allocated conditioned on task difficulty, length and uncertainty calibration;
- quality versus compute curves selected on validation;
- batch=1 latency and realistic batched throughput;
- peak memory and allocator overhead;
- kernel launch count, padding waste and gather/scatter cost;
- energy per generated token where measurement is available.

A controller that saves 30% of abstract refinement steps but runs 23% slower on CPU is not an inference optimization for that implementation. It may still be useful on a different accelerator with fused kernels, but that must be measured.

## 5. Controller quality and ACT details

ACT should implement one mathematically consistent halting formulation. The additive halting mass, remainder rule, ponder cost and gradients must be derived from the same formulation. A detached halting-mass accumulator must not sever the gradient to the halt controller.

Training should start from a competent fixed-depth model or use an explicit competence gate. A fixed number of warm-up steps is not enough if the teacher's validation performance is still poor. Threshold selection must use the adaptive execution path and validation data only.

For value-of-computation, prediction accuracy of the value head is not policy quality. Measure:

- calibration of predicted marginal gain;
- action regret relative to the exhaustive full-depth oracle;
- ranking of examples by marginal value of extra compute;
- policy quality at fixed compute budgets;
- end-to-end serving cost.

## 6. Minimum evidence before architecture promotion

- At least five independent seeds for early promotion screens.
- A second independently sourced dataset/domain.
- Immutable dataset revisions and use-rights review.
- Parameter-, token-, FLOP- and optimizer-budget matching where relevant.
- Pre-registered primary metrics and quality margins.
- Validation-only architecture/threshold selection.
- A single sealed test evaluation after selection.
- Confidence intervals and paired per-seed comparisons.
- Target-hardware profiling using optimized kernels.
- Reproducible source hashes, configs, checkpoints, logs and environment manifests.

## 7. Official LLM training gate

Do not proceed to an expensive official LLM training run merely because example models converge. First show that the selected mechanism improves a relevant metric at an acceptable cost on a rights-cleared subword corpus, and that the improvement replicates. Train small examples first, then a larger pilot, then scale only the components that survive independent ablation.
