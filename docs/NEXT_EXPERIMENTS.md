# Next experiments — ordered, gated and prior-art-aware

## Gate 0 — freeze provenance, rights and source before experiments

The V11 dynamic-computation implementation referenced in prior session handoffs is not included in this snapshot. Reconstruct it as new source code; do not represent previous run metrics as exact reproductions. At launch, record Git commit, source tree digest, dataset manifest digest, dependency lock, Python/framework versions, hardware, precision, model config, seed, batch-order seed, and command line. Keep immutable raw outputs and a per-run manifest.

Before selecting a dataset, audit the precise rights for training, evaluation, derivative models, public checkpoint distribution and commercial use. A publicly downloadable dataset is not automatically cleared for all of these uses. Separate data preparation from training so the source revision and train/validation/test splits can be inspected before execution.

Read [`PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md`](research/PRIOR_ART_DYNAMIC_COMPUTE_AND_NGRAM_2026-11-19.md) before proposing novelty claims. The session's V11 source/checkpoints remain missing; its metrics are session-reported.

## Gate 1 — reproduce and characterize local pair/N-gram memory

The broad ideas of hashed n-gram lookup, context-aware gating and additive residual injection already exist in SCONE, Engram and 2026 follow-up work. This gate tests whether the project's specific compact/local variant delivers useful value under matched resource budgets; it does not presume novelty.

- Start with `experiments/v11_prototypes/local_ngram_adapter.py` and its 13 synthetic tests. Treat this as a new reference prototype, not the unrecovered V11 implementation or a trained model.
- Add a dataset-independent collision audit that reports distinct n-grams, bucket occupancy, collision rate by order, table bytes and expected optimizer-state bytes. Test multiple table sizes before training.
- Reconstruct the exact pair/N-gram adapter described in the session log. Preserve its source and math as an independent experiment profile.
- Re-run the original tiny character task first for operator/metric continuity, with the same fixed data and split manifest.
- Compare against dense baseline, smoothed bigram, a stronger classical n-gram control, matched wider FFN, matched embedding parameters without n-gram keys, and an independently implemented hashed n-gram + context gate control.
- Sweep bounded hash-table sizes and report collisions; do not create a dense vocabulary-squared pair table.
- Test one compact subword variant only after character-level operator tests pass. Consider low-rank/factorized storage and shared tables as separate ablations, not bundled features.
- Use immutable rights-cleared datasets with independent train/validation/test sources. Use at least five seeds; report paired per-seed deltas, mean, standard deviation and confidence intervals.
- Match parameter count, activated FLOPs, optimizer-state budget, training tokens, batch order and wall-clock budget where the comparison requires it.
- Select table sizes, n-gram orders, injection positions and checkpoints on validation only. Evaluate test once after freezing the configuration.

**Promotion:** improvement must replicate on a second corpus/domain and survive matched parameter/token/FLOP/optimizer controls; no unacceptable regression on long-tail or out-of-domain tokens; resource use must be stated honestly as compute versus memory trade-off.

## Gate 2 — reproduce ACT with mathematically consistent halting

- Implement one coherent Graves-style ACT definition with additive halting mass.
- Verify mass conservation, remainder semantics and exact gradient propagation to halt logits.
- Train a competent full-depth teacher; activate ponder regularization only after a validation-based competence gate.
- Select threshold/checkpoint using adaptive validation inference, not the full-depth teacher path.
- Evaluate once on test.
- Compare against fixed depth at matched model parameter count and training tokens.
- Include same-input rerun stability and route allocation conditioned on difficulty, with length/position controls.

**Promotion:** better quality/compute Pareto frontier and improved end-to-end target-hardware latency or energy. Abstract steps alone do not pass.

## Gate 3 — compare causal per-token skip routing against published mechanisms

The earlier session used the label “TSA”. A 2026 paper already names a closely related residual-update gate Token-Selective Attention (TSA). Keep historical session results under `session_tsa`; use `tsa_2605` for the published paper, and complete an operator-level comparison before claiming any difference.

- Per-token routing only; no future-dependent whole-sequence top-k.
- Add property-based prefix-causality tests for both logits and routing decisions.
- Compare dense, early-exit, soft residual gate, hard-gate sparse execution, straight-through training, ACT, and—where code/compute permit—TSA-2605-like, GateSkip-like and FlexiDepth-like controls.
- Distinguish attention and FFN branch skipping; log whether KV cache and attention operations remain dense.
- Verify training/inference routing agreement and that each claimed skipped operation is actually not executed.
- Test packed/ragged and gather/scatter execution, including batch=1 and realistic batches. Include router, synchronization, padding and memory-movement overhead.
- Measure routing stability under small semantics-preserving perturbations and report budget overshoot/variance.

**Promotion:** quality within a preregistered margin, reliable causal routing, and measurable target-hardware wall-clock improvement; route choices must remain stable enough for deployment.

## Gate 4 — safety-aware evaluation of adaptive execution

- Measure harmful-request recognition and refusal behaviour separately from benign task quality.
- Evaluate prompt-prefill and answer-generation routing as separate execution phases; do not assume the same layers are safe to skip in both.
- Include harmful-completion rate, refusal quality, benign false-refusal rate, task accuracy and quality/compute curves.
- Add routing interventions: restore selected layers for prompt tokens or generated tokens and measure the change.
- Keep a full-depth reference model as the safety baseline.

**Promotion:** adaptive execution must not create an unacceptable safety regression relative to the full-depth reference; quality and safety must be reported together.

## Gate 5 — reproduce value-of-computation as a policy, not only a regressor

- Use teacher marginal-loss reduction as a target, but evaluate policy regret separately from value-prediction R².
- Plot compute allocation versus teacher error after controlling for sequence length, position and task category.
- Add calibration, ranking, regret and threshold-stability metrics.
- Compare fixed threshold, ACT, entropy baseline and oracle allocation upper bound.
- Evaluate difficulty allocation on parameterized algorithmic/synthetic tasks with held-out input lengths; verify that a controller does not merely exploit static sequence cues.

**Promotion:** the controller allocates more compute to examples with larger marginal benefit and improves quality/compute frontier on unseen lengths and another domain.

## Gate 6 — rights-cleared dataset integration

- Pin immutable dataset commit/revision and license/terms snapshots.
- Audit training, evaluation, derived-weight distribution and commercial-use rights separately.
- Download/export train, validation and test separately.
- Verify hashes and row/token counts.
- Record contamination, deduplication, normalization and preprocessing controls.
- Never use test metrics to select hyperparameters or checkpoint.
- Never treat public availability as training-use permission.

## Gate 7 — subword example model

Train a small subword decoder-only reference and one candidate with only the promoted mechanism. Include a well-tuned baseline, n-gram/byte-level/statistical controls where relevant, multiple seeds, second-domain and held-out-length tests. Avoid enabling MoE, long-context sparse attention, multi-mode decoding and external memory simultaneously. Each additional mechanism requires its own ablation.

## Gate 8 — target-hardware optimization

Only after predictive and safety benefits are established, implement fused kernels and benchmark:

- batch=1 decode latency and time-to-first-token;
- concurrent/batched throughput and tail latency;
- prefill latency separately from decode;
- peak memory and KV/recurrent-state/table memory;
- total and active FLOPs, router overhead, kernel count and padding waste;
- gather/scatter and host-memory transfer cost;
- energy if available;
- quality and safety at each compute budget;
- route stability and budget overshoot.

## Scale-up gate

Official training is justified only when a selected mechanism improves the quality/efficiency/safety frontier across at least two datasets/domains, including a controlled difficulty or length extrapolation setting, and the implementation is stable, rights-compliant, hardware-feasible and independently reproducible.
