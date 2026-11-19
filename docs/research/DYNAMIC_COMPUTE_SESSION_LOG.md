# Dynamic neural computation — session experiment log

**Coverage:** 18–19 November 2026 sessions  
**Provenance:** conversation-visible experiment updates and generated handoff drafts; V11 source/result files were not recovered into this bundle. All figures below are therefore **session-reported**, not independently replayed from raw artifacts in this repository.

## 1. Objective

Test whether a language model can allocate computation dynamically at token or sequence level, and whether adding a compact local-pair/N-gram representation improves next-token prediction accuracy. The experiments were intended as example-model validation before any official LLM-scale training.

The mechanisms investigated were:

- Adaptive Computation Time (ACT): recurrent refinement steps with learned halting.
- Token-Selective Attention (TSA): causal per-token decision to execute or skip a later Transformer update.
- Teacher-supervised Value of Computation (V.O.C.): predict marginal value of another refinement step from a full-depth teacher.
- Local pair/N-gram residual representation: add short-range token-context features to increase local predictive capacity.

## 2. Design constraints adopted

1. Every token's compute decision must depend only on causal information available at that position.
2. Global top-k token selection over a whole sequence is invalid for an autoregressive causal model if future tokens can influence earlier routing.
3. Fixed-depth and adaptive models must be parameter-matched and start from the same competent model when the experiment is testing compute allocation.
4. Dynamic thresholds must be selected on validation data only, then frozen before test evaluation.
5. Report actual wall-clock latency, throughput and memory alongside abstract skipped token-layer operations.
6. A soft gate does not save compute unless the hard inference path truly skips work.
7. Failed/invalid experiments must remain in the record; do not merge superseded runs into canonical metrics.

## 3. ACT experiments

### 3.1 Initial recurrent refinement pilot

The first small-text ACT model learned to halt most tokens after about two of four possible refinement steps. The adaptive and fixed-depth variants had nearly identical held-out character loss, while the adaptive implementation used about half the maximum recurrent FFN token-steps. The compact two-character embedding path improved accuracy in both fixed and ACT variants; the combined ACT+N-gram model reportedly retained most of that gain with fewer refinement steps.

These early results were from small pilot settings and should not be compared directly with later corrected parity or language-model runs.

### 3.2 Synthetic parity task

The first ACT parity attempt used an excessively strong ponder penalty. It halted almost immediately and reduced parity accuracy to approximately chance, while the fixed-depth model reached 98.7%.

A curriculum/full-depth warm-up recovered performance on one run: ACT reached 99.18% held-out parity accuracy and used 5.92 of 6 steps on average. This preserved accuracy but gave little compute saving.

A subsequent ponder sweep reported one setting with 98.70% accuracy versus 98.72% fixed-depth, using 4.13 of 6 steps on average (31.2% fewer refinement steps). The reported controller used more steps on longer sequences (3.50 on lengths 4–8 versus 4.54 on lengths 25–32).

An independent seed failed: its fixed-depth teacher reached only 80.5% validation accuracy at the end of the fixed warm-up, and ACT regularization caused the controller to collapse toward one step and chance-level accuracy. A later minibatch-order change also caused the fixed-depth teacher to remain near chance, indicating sensitivity to initialization/data order. These failures must not be omitted.

### 3.3 Corrected ACT objective and evaluator

A code audit found that one implementation used Graves-style additive halting mass but computed ponder cost with a multiplicative survival formula. Those are different ACT formulations. Another version detached the accumulated halting mass used by the ponder regularizer, breaking the gradient into the halt controller. The session reports that both issues were corrected and that a finite non-zero controller gradient test was added.

A separate evaluation bug scored all positions in overlapping context windows, counting the same target multiple times. The corrected evaluator scored exactly one next-character target per window. A split boundary was also corrected to match the documented 23,940 / 2,993 / 2,993 character train/validation/test split.

### 3.4 Validation-selected quality/compute sweep

A corrected ACT experiment selected dynamic checkpoints and halting thresholds using the adaptive path and validation data. One reported three-seed result showed selected policies saving roughly 6–13% of recurrent token-steps; a seed had materially worse test NLL than its fixed-depth control. A later strict quality-budget result for seed 3 saved 29.0% of recurrent token-steps, with approximately 23% slower CPU execution. A looser budget saved 35.8% of steps but had worse test loss.

These are separate revisions/configurations. They are not to be averaged together without the underlying run manifests and exact source revisions.

### 3.5 V11 corrected parameter-matched ACT snapshot

A corrected experiment compared fixed-depth and ACT models with 29,891 parameters each. The first identified seed reached 99.25% test accuracy but averaged 5.97 of 6 refinement steps. Its compute allocation had approximately 0.006 correlation with example difficulty after controlling for sequence length, and the Python ACT path was reported 2–3× slower on CPU. This demonstrates why a low ponder budget or a lower nominal step count is not enough: the selected policy must allocate compute to hard examples and improve end-to-end serving cost.

## 4. Token-selective attention experiments

### 4.1 Invalid global top-k design

The first top-k token-budget implementation selected the highest-scoring tokens globally across a sequence. A future token could affect whether an earlier token was processed, violating causal invariance. This design was rejected.

### 4.2 Causal per-token routing

The replacement used a per-token halting/skip decision based only on that token's causal hidden state. The session reports that it passed a causality test and that hard sparse inference decisions matched the straight-through training decisions.

One TSA + N-gram seed reportedly skipped about 26% of total token-layer operations while remaining within 0.014 nats/character of the matched dense N-gram model. Results varied with seed and threshold; this remains a pilot observation, not a general quality/latency result.

Crucially, token-layer operations skipped are not equivalent to FLOPs or end-to-end savings unless routing overhead is included and the implementation actually avoids the skipped kernel work.

## 5. Teacher-supervised Value of Computation

A small controller was trained to predict the marginal loss reduction from another refinement step using a full-depth teacher.

A first result reportedly reduced recurrent token-steps by 26% on one seed while remaining within 0.014 NLL of the full-depth teacher, but the gather/scatter reference implementation was 47% slower on CPU and controller calibration was weak.

A later three-seed analysis reported that the value head estimated the teacher's remaining loss gap with validation R² around 0.96–0.98. This did not produce good routing: in two seeds, allocated compute was negatively correlated with final teacher loss after controlling for length, meaning the controller often assigned fewer steps to harder examples.

High R² for a predicted scalar does not prove that the policy chooses the correct action. The policy must be evaluated for regret, quality at a fixed compute budget, and actual execution cost.

## 6. Pair/N-gram representation and accuracy

The later example-model run reportedly compared five profiles over three seeds at 1,000 steps. The short-context pair/N-gram adapter (`BASE_NGRAM`) beat a nearly parameter-matched wider FFN on all three seeds, with paired mean test NLL difference −0.1672 and top-1 accuracy difference +5.99 percentage points.

This is the most promising accuracy-oriented hypothesis in the session. It is still limited to one tiny character-level corpus, with no subword vocabulary, no broad-domain corpus, no full rights-cleared online dataset and no official model evaluation. The next version should use bounded fixed-size hashed/factorized pair features or equivalent subword-compatible features; a dense vocabulary-squared table is not scalable.

## 7. Failed or invalidated approaches to preserve

| Attempt | Failure | Disposition |
|---|---|---|
| Sequence-global top-k routing | Future tokens can influence earlier compute decisions | Rejected |
| Overlapping-window all-position scoring | Duplicate target positions bias evaluation | Superseded; score one defined target per window |
| Additive ACT halting with multiplicative ponder survival | Objective does not match halting rule | Corrected in session, but V11 code absent here |
| Detached halting mass in ponder objective | No controller gradient from ponder penalty | Corrected in session, but regression source absent here |
| High initial halt bias / excessive ponder penalty | Premature exits and accuracy collapse | Rejected; tune from competent full-depth model |
| Fixed warm-up length without competence gate | One seed's teacher was still underfit | Rejected; use validation-based competence gate |
| Entropy/confidence as universal difficulty score | Poor difficulty correlation/calibration | Not promoted |
| Python gather/scatter as serving implementation | Overhead exceeded skipped compute | Reference-only; fused kernels required |

## 8. What the evidence says

- Dynamic computation is mechanically implementable: controllers can stop/skip token-level or recurrent refinement operations.
- The controller must learn *where* compute has marginal value, not just minimize average steps.
- Accuracy must be selected on the adaptive execution path, not the full-depth teacher path.
- Thresholds must be chosen on validation, never test.
- Causality, hard/soft routing equivalence, halt-mass conservation and gradient flow are hard invariants.
- Abstract token-step savings did not translate into CPU latency improvement in the Python reference paths.
- The pair/N-gram adapter is a stronger current accuracy hypothesis than adaptive halting, but remains unproven beyond a tiny character corpus.

## 9. Reproduction gap

The V11 session's full `src/llm_arch/`, tests, scripts, `data/results_*`, V.O.C. checkpoints and raw per-seed manifests were not present in the recovered filesystem or available file listing at packaging time. Earlier generated `README.md` and `HANDOFF.md` drafts describe those paths, but those drafts are preserved as superseded session handoffs rather than treated as proof that the files exist.

Therefore, all V11 metrics above are marked **session-reported**. Rebuild the implementation from the logged specification and rerun before using these values for publication, model selection or architecture promotion.

## 10. Next required experiment

1. Reconstruct V11 code as an independently versioned experiment, starting from the runnable v9 pilot.
2. Freeze data split, tokenizer, model/optimizer configuration, random seeds, batch order and launch-time code hashes.
3. Compare fixed-depth, ACT, causal TSA and V.O.C. independently; use parameter-matched controls.
4. For the pair/N-gram adapter, compare against both matched wider FFN and matched embedding/parameter controls.
5. Use at least five seeds and a second corpus/domain.
6. Select all checkpoints and thresholds on validation; run test once after selection.
7. Add target-hardware kernels and benchmark batch=1 latency, batched throughput, memory and energy.
8. Promote only when the quality/efficiency frontier is better than the simplest feasible baseline under predeclared acceptance bounds.
