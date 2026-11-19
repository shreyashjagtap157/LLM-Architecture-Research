# Next experiments — ordered and gated

## Gate 0 — recover and freeze the V11 experiment source

The V11 dynamic-computation implementation referenced in prior session handoffs is not included in this snapshot. Reconstruct it as new source code; do not represent the previous run metrics as exact reproductions. At launch, record git commit, source tree digest, dataset manifest digest, dependency lock, seed, batch-order seed, model config and hardware.

## Gate 1 — reproduce the accuracy adapter

- Reconstruct the pair/N-gram residual adapter described in the session log.
- Test on the original tiny character task first for operator/metric continuity.
- Add matched wider-FFN, matched embedding-parameter and no-adapter controls.
- Convert it to bounded hashed/factorized local features for a subword vocabulary.
- Use an immutable rights-cleared dataset with separate train/validation/test sources.
- Use at least five seeds; report paired per-seed deltas, mean, SD and confidence intervals.

**Promotion:** improvement must replicate on a second corpus/domain and survive parameter/token/FLOP-matched controls.

## Gate 2 — reproduce ACT

- Implement one coherent Graves-style ACT definition with additive halting mass.
- Verify mass conservation and exact gradient propagation to halt logits.
- Train a competent full-depth teacher; activate the ponder penalty only after a validation-based competence gate.
- Select threshold/checkpoint using adaptive validation inference.
- Evaluate once on test.
- Compare against fixed depth at matched model parameter count and training tokens.

**Promotion:** better quality/compute Pareto frontier and improved end-to-end target-hardware latency or energy. Abstract steps alone do not pass.

## Gate 3 — reproduce causal token-selective attention

- Per-token routing only; no future-dependent whole-sequence top-k.
- Add property-based causality tests for logits and routing decisions.
- Compare dense, soft-gate, hard-gate, straight-through and packed-ragged implementations.
- Verify training/inference routing agreement.
- Include router overhead and padding waste in the measured FLOPs/latency.

**Promotion:** quality within a predeclared margin and measurable target-hardware wall-clock improvement.

## Gate 4 — reproduce value-of-computation

- Use teacher marginal-loss reduction as a target, but evaluate policy regret separately from value prediction R².
- Plot compute allocation versus teacher error after controlling for sequence length and task category.
- Add calibration, ranking and regret metrics.
- Compare with fixed threshold, ACT, entropy baseline and oracle allocation upper bound.

**Promotion:** the controller allocates more compute to examples with larger marginal benefit, and improves quality/compute frontier.

## Gate 5 — complete online dataset integration

- Pin immutable dataset commit/revision.
- Audit exact data terms for training, evaluation, model distribution and commercial use.
- Download/export train, validation and test separately.
- Verify hashes and row/token counts.
- Record contamination and deduplication controls.
- Never treat public availability as training-use permission.

## Gate 6 — subword example model

Train a small subword decoder-only reference and one candidate with only the promoted mechanism. Include a well-tuned baseline, n-gram/byte-level controls where relevant, multiple seeds, and domain holdouts. Avoid enabling MoE, long-context sparse attention, multi-mode decoding and external memory simultaneously.

## Gate 7 — target-hardware optimization

Only after the architecture's predictive benefit is established, implement fused kernels and benchmark:

- batch=1 decode latency;
- concurrent/batched throughput;
- prefill latency;
- peak memory and KV/recurrent-state memory;
- total and active FLOPs;
- energy if available;
- compile/graph capture overhead;
- quality at each compute budget.

## Scale-up gate

Official training is justified only when the chosen mechanism improves a Pareto frontier across at least two datasets/domains and the implementation is stable, rights-compliant, hardware-feasible and independently reproducible.
