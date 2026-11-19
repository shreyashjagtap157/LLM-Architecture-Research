# Dynamic neural computation — architecture addendum

**Status:** research proposal plus small-scale session experiments; not production-validated.  
**Applies to:** v10 synthesized architecture specification.

## 1. Architectural intent

Dynamic neural computation means the model or serving system can conditionally allocate computation based on the current token/sequence state and an explicit compute budget. It is not a single mechanism. The architecture should separate at least four control levels:

1. **Sequence/task budget:** choose an allowed inference/training compute envelope before decoding or reasoning begins.
2. **Layer/token selection:** decide whether a token requires a particular expensive update, while preserving causal dependencies.
3. **Iterative refinement/halting:** decide whether a representation needs another recurrent refinement step.
4. **Capacity routing:** decide which experts or memory banks are activated for a token.

These mechanisms require separate controllers and metrics. A single global uncertainty vector or entropy value is not a sufficient universal controller.

## 2. Candidate mechanisms

| Mechanism | State/decision | Expected advantage | Main risk |
|---|---|---|---|
| ACT recurrent refinement | Per-token halt mass / recurrent iteration count | Stop iterative processing when marginal refinement is low | Premature halting, unstable ponder regularization, controller collapse |
| Causal token-selective attention | Per-token binary/soft skip decision for later layer updates | Skip unnecessary token-layer work | Routing overhead, padding/packing, causality leakage |
| Teacher-supervised V.O.C. | Predicted marginal quality gain of another compute step | Train a policy from teacher-measured value | Value-head R² may not imply good action ranking or policy regret |
| Task-level compute router | Budget class and candidate path | Spend more compute on difficult task classes | Misclassification and budget misallocation |
| Sparse expert routing | Active expert set | Conditional capacity | Communication, imbalance, expert collapse |

ACT/TSA/V.O.C. are separate candidates, not one fused design. Test them individually before combining.

## 3. Causal token routing contract

For a causal sequence with hidden states \(h_1,\ldots,h_T\), a routing decision at position \(t\) must be a function only of the prefix available at that point:

\[
 r_t = g(h_{\le t}, s_t, b_t),
\]

where \(s_t\) is permitted recurrent state and \(b_t\) is the remaining compute budget. It must not depend on future hidden states, future router scores or a global top-k computed over positions \(t+1,\ldots,T\).

The first whole-sequence top-k prototype was rejected because future tokens could affect whether earlier tokens were processed. Per-token causal routing is the only retained token-selective path in this addendum.

## 4. ACT contract

An ACT implementation must use one internally consistent halting formulation. If additive halting mass is used, the remainder/ponder terms and gradients must be derived from that same additive-mass definition. The implementation must test:

- halting mass conservation;
- finite, non-zero halt-controller gradient from the ponder objective;
- hard execution decisions versus the intended soft/straight-through training path;
- no state leakage across independent requests;
- deterministic behavior under fixed seed/configuration;
- adaptive-path checkpoint selection and evaluation.

A fixed warm-up step count is not enough. ACT regularization should activate only after a validation-based competence condition confirms that the fixed-depth model has learned the task to the required level.

## 5. V.O.C. controller contract

The teacher may provide a target for marginal loss reduction, but controller quality is not established by a high value-prediction R² alone. The policy needs separate measurements:

- marginal-value calibration;
- rank correlation with the true value of additional compute;
- action regret versus exhaustive full-depth evaluation;
- quality at fixed compute budgets;
- performance after controlling for sequence length and task category;
- wall-clock and energy cost of the controller itself.

If a controller assigns fewer steps to examples with larger residual teacher error, it has failed as a compute-allocation policy even if the scalar value regressor fits its target well.

## 6. Runtime contract

Skipped nominal operations count as a benefit only when the implementation avoids executing those kernels. All candidate routers must report:

- active token-layer operations and actual estimated FLOPs;
- active refinement steps distribution (mean and percentiles);
- router, gather/scatter, packing, padding and kernel-launch overhead;
- batch=1 latency and realistic batched throughput;
- peak memory and state-cache memory;
- energy where measurable;
- quality at each selected compute budget.

The session's Python gather/scatter and recurrent reference paths were reported slower on CPU despite reducing abstract step counts. They are correctness/reference implementations only, not serving kernels.

## 7. Training and promotion order

1. Train a competent fixed-depth baseline.
2. Add one adaptive mechanism with identical parameters and training tokens.
3. Verify causal and gradient invariants.
4. Select the dynamic policy using validation execution only.
5. Evaluate test once after selection.
6. Run multiple seeds and a second corpus/domain.
7. Implement fused kernels on target hardware.
8. Promote only if the quality/compute Pareto frontier improves and actual latency/throughput/energy gains are measured.

Dynamic computation must remain behind an experiment flag until these gates pass.
