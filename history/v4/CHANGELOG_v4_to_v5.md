# v4 to v5 Change Log

**Date:** 17 November 2026

## Main additions

- Added DHSA (ICML 2026) as a distinct hierarchical chunk-to-token sparse-attention inference candidate; kept it separate from DSA token top-k and QSA micro-block selection.
- Added DiffuMamba (ICML 2026) and created P6, an independent masked-diffusion challenger that does not share the autoregressive promotion chain.
- Expanded model/runtime memory taxonomy to five distinct planes by adding bounded request-local editable slots with explicit write, update, protect, invalidate, expire, retrieve and sparse-fallback lifecycle operations.
- Added lifecycle data schema, storage estimate, conflict/provenance requirements, deletion behavior and failure-mode tests for request-local memory.
- Fixed a tensor-symbol collision in the pre-norm block equations (`V_l` cache vs FFN activation).
- Made indexed sparse-attention selection notation explicit: top-k eligible history plus mandatory global/sink and local-window positions, de-duplication and causal masking.
- Strengthened statistical acceptance with paired test sets, per-seed results, confidence intervals, multiple-comparison controls and an untouched final holdout.
- Included hyperparameter-search cost in matched-compute comparisons.
- Added v5 architecture, benchmark, operator and experiment manifest templates.
- Removed the repeated legacy status paragraph and updated package/version language.

## New sources

R76 DHSA; R77 DiffuMamba; R78 editable request-local memory; R79 next-generation architecture survey.

## Explicitly unresolved

Target model scale, accelerator topology, compute, data licenses, workload distribution, risk thresholds, slot budget and diffusion refinement budget remain null until declared. No architecture-level performance claim is made for the combined v5 proposal.
