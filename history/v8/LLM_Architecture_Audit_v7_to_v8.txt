# Technical Audit — v7 to v8

**Review date:** 18 November 2026  
**Baseline:** `SOTA_LLM_Architecture_v7.md`  
**Result:** v8 makes architecture comparison more systematic and separates speculative generation paths by their output/fidelity contracts. It remains a research specification, not a trained model.

## 1. Architecture-space completeness

v7 had broad component coverage but still presented much of the field as a collection of families. The November 2026 architecture survey's four axes—processing structure, transformation dynamics, state management and output commitment—provide a better organizing frame. v8 makes these dimensions explicit and explains why MoE, KV compression, recurrence and speculative decoding do not solve the same cost. The survey is a literature synthesis, not a matched-condition ranking. [R95]

## 2. Diffusion-generation paths were still conflated

v7 covered a shared tri-mode model and DSpark, but it did not separately specify diffusion-as-drafter, iterative draft editing and AR-seeded diffusion refinement:

- **DiffuSpec:** a DLM emits a token lattice; CPS extracts a verifier-aligned causal path and ADL adapts draft length; AR target verifies. Up to 3× wall-clock speedup is the paper's tested result, not a transferable promise. [R96]
- **DEdit:** a DLM draft is iteratively edited with bidirectional context before target verification. Published Qwen3-4B/8B results include different greedy and stochastic speedups; edit passes add draft cost and can have task-dependent value. [R97]
- **SpecRef:** an AR draft initializes masked diffusion. The output may deliberately change; no AR target distribution is preserved by the refinement step alone. The study identifies scaffold, edit-damage, metric-ranking and non-AR evaluation-harness confounds. [R98]

v8 makes P8 (DiffuSpec/DEdit with target verification) and P9 (AR-seeded changed-output refinement) separate candidate tracks. It requires the exactness claim to be supported by the actual verifier procedure.

## 3. Speculation must be evaluated as systems co-design

An IEEE Access survey of 312 works formalizes Draft Source, Draft Geometry, Verification Fidelity and Runtime Execution, then defines Representability, Verifiability, State Realizability and Profitability gates. v8 converts these to explicit implementation and promotion checks. The paper is a taxonomy rather than an independent benchmark of all surveyed systems. [R100]

## 4. Training schedule

CGLS studies progressive layer stacking together with increasing data difficulty, with reported experiments at 100M and 1.2B parameters. v8 adds a 2×2 factorial design to separate layer growth from curriculum effects. It requires equalized tokens/compute and explicit initialization/optimizer-state-transfer details. The paper does not establish frontier-scale transfer. [R99]

## 5. Additional static quality controls

v8 updates the end-to-end diagram, candidate-profile YAML, benchmark contract, failure-mode test matrix, decision record, reference register and package README. It adds a Pareto selection rule: hard safety, fidelity, memory and latency limits are feasibility constraints; average quality cannot compensate for violating them.

## 6. Remaining blockers

1. Target parameter and active-parameter budgets remain unspecified.
2. Target accelerator, memory capacity, interconnect, drivers and supported kernels remain unspecified.
3. Training corpus, licenses, domain/language distribution and compute budget remain unspecified.
4. No model training, GPU kernel test or end-to-end benchmark has been executed.
5. The reported paper/vendor speedups remain method- and hardware-specific; independent replication is required.

## 7. Disposition

v8 should be the next implementation specification baseline. The next progress should be empirical: implement dense P0; compare P1 recurrent and P1-LA local/global under matched parameter, token and compute budgets; then add only the mechanisms that win their isolated ablations. P8/P9 and CGLS should be run only if the use case justifies them. Further architecture additions should pause until one of the measured comparisons produces new evidence or a concrete workload requirement.
