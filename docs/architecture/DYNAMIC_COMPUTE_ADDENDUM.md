# Dynamic neural computation — architecture addendum

**Status:** research proposal plus small-scale session experiments; not production-validated.
**Applies to:** v10 synthesized architecture specification.
**Prior-art review:** [`../research/PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md`](../research/PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md).

## 1. Architectural intent

Dynamic neural computation means the model or serving system can conditionally allocate computation based on the current token/sequence state and an explicit compute budget. It is not a single mechanism. The architecture should separate at least four control levels:

1. **Sequence/task budget:** choose an allowed inference/training compute envelope before decoding or reasoning begins.
2. **Layer/token selection:** decide whether a token requires a particular expensive update, while preserving causal dependencies.
3. **Iterative refinement/halting:** decide whether a representation needs another recurrent refinement step.
4. **Capacity routing:** decide which experts or memory banks are activated for a token.

These mechanisms require separate controllers and metrics. A single global uncertainty vector or entropy value is not a sufficient universal controller.

Prior work now directly covers several variants of token-wise adaptive depth: FlexiDepth, TSA (arXiv:2605.05222), GateSkip and ANIRA; SafeDepth adds explicit safety-aware routing. The project must position its contribution against these works rather than claim per-token routing or residual gating as new. This addendum specifies a candidate architecture and its research requirements, not a claim of novelty.

## 2. Candidate mechanisms

| Mechanism | State/decision | Expected advantage | Main risk / closest prior art |
|---|---|---|---|
| ACT recurrent refinement | Per-token halt mass / recurrent iteration count | Stop iterative processing when marginal refinement is low | Premature halting, unstable ponder regularization, controller collapse; compare ACT literature and ANIRA online halting |
| `session_tsa` causal residual skip | Per-token decision whether to execute a later block/branch update | Skip unnecessary token-layer work | Routing overhead, padding/packing, causality leakage; direct comparison required with TSA-2605, GateSkip and FlexiDepth |
| Teacher-supervised V.O.C. | Predicted marginal quality gain of another compute step | Train a policy from teacher-measured value | Value-head R² may not imply good action ranking or policy regret |
| Task-level compute router | Budget class and candidate path | Spend more compute on difficult task classes | Misclassification and budget misallocation |
| Sparse expert routing | Active expert set | Conditional capacity | Communication, imbalance, expert collapse |

ACT, session-TSA and V.O.C. are separate candidates, not one fused design. Test them individually before combining. Use `tsa_2605` only when directly referring to the published Token-Selective Attention paper; historical project observations remain named `session_tsa` in new manifests and figures.

## 3. Causal token routing contract

For a causal sequence with hidden states \(h_1,\ldots,h_T\), a routing decision at position \(t\) must be a function only of the prefix available at that point:

\[
 r_t = g(h_{\le t}, s_t, b_t),
\]

where \(s_t\) is permitted recurrent state and \(b_t\) is the remaining compute budget. It must not depend on future hidden states, future router scores or a global top-k computed over positions \(t+1,\ldots,T\).

The first whole-sequence top-k prototype was rejected because future tokens could affect whether earlier tokens were processed. Per-token causal routing is required for an autoregressive path, but is not itself a novelty claim. Add prefix-perturbation tests: perturb all tokens after position \(t\) and assert that logits and route decisions through \(t\) remain unchanged.

## 4. Residual-gate and sparse-execution contract

A soft gate during training is not automatically an inference speedup. Distinguish explicitly:

- **Soft training gate:** scales a branch or residual update while still executing its kernels.
- **Hard routing decision:** chooses which token/branch is retained for execution.
- **Sparse execution implementation:** physically avoids the selected computation in the runtime.
- **End-to-end benefit:** reduces latency/energy after routing, gathers, scatters, synchronization, padding and cache maintenance are included.

The closest published designs make different choices. TSA (arXiv:2605.05222) gates residual updates and describes sparse FFN execution while leaving attention dense; GateSkip gates Attention and MLP residual branches and applies per-layer token budgets; FlexiDepth uses a router plus adapter path. Do not compare their nominal skipped-operation counts as equivalent without aligning what operation is skipped.

Every implementation manifest must state: gate insertion point; gate input; gate function; budget/threshold rule; training surrogate; hard inference path; whether attention executes; whether K/V cache entries are copied, updated or omitted; packing strategy; and exact operations counted as skipped.

## 5. ACT contract

An ACT implementation must use one internally consistent halting formulation. If additive halting mass is used, the remainder/ponder terms and gradients must be derived from that same additive-mass definition. The implementation must test:

- halting mass conservation;
- finite, non-zero halt-controller gradient from the ponder objective;
- hard execution decisions versus the intended soft/straight-through training path;
- no state leakage across independent requests;
- deterministic behavior under fixed seed/configuration;
- adaptive-path checkpoint selection and evaluation;
- difficulty allocation after controlling for sequence length and position;
- extrapolation to held-out task difficulty/sequence lengths.

A fixed warm-up step count is not enough. ACT regularization should activate only after a validation-based competence condition confirms that the fixed-depth model has learned the task to the required level.

## 6. V.O.C. controller contract

The teacher may provide a target for marginal loss reduction, but controller quality is not established by a high value-prediction R² alone. The policy needs separate measurements:

- marginal-value calibration;
- rank correlation with the true value of additional compute;
- action regret versus exhaustive full-depth evaluation;
- quality at fixed compute budgets;
- performance after controlling for sequence length and task category;
- generalization to unseen complexity values;
- wall-clock and energy cost of the controller itself.

If a controller assigns fewer steps to examples with larger residual teacher error, it has failed as a compute-allocation policy even if the scalar value regressor fits its target well.

## 7. Safety and inference-phase contract

Adaptive routing can affect safety even when average benchmark accuracy is preserved. Prompt ingestion and answer generation are separate phases; do not reuse a single skip policy by assumption. Evaluate:

- harmful-request recognition and harmful-completion rate;
- refusal quality and refusal consistency;
- benign false-refusal rate;
- full-depth reference versus adaptive path;
- per-layer and per-phase restoration interventions;
- route distributions for refusal and ordinary benign tokens.

A candidate that exceeds the predeclared safety regression bound is not promoted by a quality/latency win. Safety evaluation must use controlled, appropriately licensed datasets and a documented response-review protocol.

## 8. Runtime, stability and budget contract

Skipped nominal operations count as a benefit only when the implementation avoids executing those kernels. All candidate routers must report:

- active token-layer operations and actual estimated FLOPs;
- active refinement steps distribution (mean and percentiles);
- router, gather/scatter, packing, padding, synchronization and kernel-launch overhead;
- batch=1 latency, prefill time, time-to-first-token and realistic batched throughput;
- peak memory and state-cache memory;
- energy where measurable;
- quality and safety at each selected compute budget;
- same-input route repeatability;
- route changes under small semantics-preserving perturbations;
- actual budget versus requested budget, including worst-case and percentile overshoot.

The session's Python gather/scatter and recurrent reference paths were reported slower on CPU despite reducing abstract step counts. They are correctness/reference implementations only, not serving kernels.

## 9. Training and promotion order

1. Train a competent fixed-depth baseline.
2. Select one adaptive mechanism with a precise operator contract and closest-prior-art comparison.
3. Add one adaptive mechanism with identical parameters and training tokens where the hypothesis requires a matched comparison.
4. Verify causal, safety, stability and gradient invariants.
5. Select the dynamic policy using adaptive validation execution only.
6. Evaluate test once after selection.
7. Run multiple seeds and a second corpus/domain, including unseen difficulty/length where applicable.
8. Implement fused kernels on target hardware.
9. Promote only if the quality/safety/compute Pareto frontier improves and actual latency/throughput/energy gains are measured.

Dynamic computation must remain behind an experiment flag until these gates pass.
