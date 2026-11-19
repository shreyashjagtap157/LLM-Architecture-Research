# Continuation handoff — LLM Architecture V11

**Snapshot date:** 19 November 2026  
**Repository version:** V11.1 research reconstruction (V11 architecture specification)  
**Current branch policy:** local `main`; no remote configured; no push performed.

## Read these first

1. `README.md`
2. `docs/EXPERIMENTAL_REVIEW_V11.md`
3. `docs/SOTA_LLM_Architecture_v11.md`
4. `docs/DATA_PROVENANCE.md`
5. `docs/NEXT_EXPERIMENTS.md`

## What is established

- A tiny character-level dense Transformer, pair/N-gram adapter, ACT model and teacher-supervised value-of-computation model are implemented in `src/llm_arch/`.
- Causality, ACT mass conservation, gradients, threshold monotonicity, parameter controls, data split checks and random-window boundary validation pass in the included 13-test suite.
- Main training: five profiles, three seeds each, 1,000 optimizer steps per run. Revalidated checkpoints and metrics are in `data/results_1000_final/`.
- Main split was corrected to 23,940 train / 2,993 validation / 2,993 test characters. The training prefix did not change; all validation/test scores and threshold choices were recomputed.
- `load_sequential_splits` honours custom train/validation fractions and validates fraction combinations. `random_batch` validates dimensions and includes the final legal next-token window. These behaviours are covered by regression tests.
- The pair/N-gram residual adapter outperformed a nearly parameter-matched wider FFN in all three paired seeds on this corpus.
- ACT and teacher-supervised V.O.C. both execute adaptive per-token recurrent refinement. Both reduced abstract recurrent token-step counts; neither produced CPU latency improvement in the reference implementation.
- V.O.C. selected dynamic execution preserves most but not all full-depth quality; the value head's validation R² is very low.

## What is not established

- No official subword LLM has been trained.
- No complete online corpus download was performed in the current experiment. A 29,926-byte excerpt was retrieved from a public GitHub source and used.
- Dataset rights remain pending; no production/commercial training authorization exists.
- The v10 training scripts named in the previous spec were not recovered. V11 is an independent reconstruction, not an exact reproduction.
- No GPU kernel speed, multi-GPU scaling, energy, latency-at-scale, or frontier quality has been measured.
- The full proposed architecture (MoE, Mamba/GDN/DeltaNet replacement backbone, external/episodic memory, multimodal paths, diffusion decoder) has not been trained.

## Canonical results

- `data/reports/v11_evidence_summary.json`
- `data/reports/v11_profile_metrics.csv`
- `data/reports/v11_act_quality_budget.csv`
- `data/reports/v11_voc_metrics.csv`
- Main run records: `data/results_1000_final/revalidated_*.json`
- V.O.C. run records: `data/results_voc_distilled2/REVALIDATED_*.json`

Earlier result files and probe folders are preserved as research history. Do not mix their scores with the canonical corrected-split results.

## Engineering recommendation

1. Preserve the pair adapter as a promising accuracy hypothesis, but implement a fixed-budget hashed/factorized version for subword vocabularies. Dense `V²` tables are not scalable.
2. Keep dynamic computation behind a separate experiment flag. Do not promote ACT/V.O.C. to the default until the controller is calibrated and the fused target-hardware path improves measured latency/energy at a predeclared quality margin.
3. Train a subword example model on a complete rights-reviewed dataset before an official LLM configuration.
4. Use at least five seeds initially, parameter/token/FLOP matched controls, multiple domains, immutable datasets and a pre-registered threshold/selection rule.
5. Run the accuracy-adapter and dynamic-compute ablations independently first; only then run a factorial combined test.

## First commands in a new environment

```bash
python -m pip install -e ".[dev]"
python -m compileall -q src scripts tests
python -m pytest -q
python scripts/build_evidence_summary.py
python scripts/verify_research_bundle.py
```

## Open decisions that require project-owner input

- What should the first official example model optimize for: general English, coding, mathematics, multilingual, or a specific domain?
- What accelerator(s), memory capacity and training budget are available?
- What approximate parameter/active-parameter budget should be targeted?
- Which datasets can be legally used for training and commercial distribution?
- Should dynamic compute optimize throughput, single-request latency, energy, or a quality-constrained compute budget?

Do not infer these answers. The architecture should remain configurable until they are supplied.

## Copy-ready continuation prompt

> Continue the LLM Architecture Research V11.1 repository from this bundle. First inspect `HANDOFF.md`, `README.md`, `docs/EXPERIMENTAL_REVIEW_V11.md`, `docs/SOTA_LLM_Architecture_v11.md`, `docs/DATA_PROVENANCE.md`, and `docs/NEXT_EXPERIMENTS.md`. Treat `data/reports/v11_evidence_summary.json` and the `revalidated_*` records as canonical evidence; do not mix superseded probe runs into aggregates. Preserve the dataset-rights warning and never claim an official LLM or GPU result. Run the test suite before changing code. The strongest candidate is the pair/N-gram adapter versus a parameter-matched wider FFN; next implement fixed-budget hashed/factorized subword features and test them on a rights-cleared, immutable subword corpus with at least five seeds. Keep ACT and V.O.C. experimental until fused target-hardware execution demonstrates a real latency/throughput/energy gain at a predeclared quality bound. Make changes only in this local repository unless explicitly authorized to publish elsewhere, preserve all historical evidence, and update the handoff and validation report after each milestone.
