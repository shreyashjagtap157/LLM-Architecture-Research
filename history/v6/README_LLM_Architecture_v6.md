# SOTA LLM Architecture v6 — Research Package

**Review date:** 17 November 2026  
**Status:** research specification and experimental contract; not a trained or benchmarked model.

## Main documents

- `SOTA_LLM_Architecture_v6.md` — full v6 architecture synthesis, 86 references, taxonomy, design, interfaces, cost model and experimental programme.
- `SOTA_LLM_Architecture_v6.txt` — text-identical export.
- `LLM_Architecture_Audit_v5_to_v6.md` — rationale, new evidence, corrections and implementation/experimental consequences.
- `CHANGELOG_v5_to_v6.md` — revision summary.

## Machine-readable contracts

- `architecture-profile-v6.yaml` — candidate architecture family and optional feature defaults.
- `benchmark-contract-v6.yaml` — candidate matrix, metrics, reproducibility, statistics and acceptance controls.
- `operator-contract-v6.yaml` — logical tensor/state/operator contracts and invariants.
- `experiment-manifest-template-v6.yaml` — one-run provenance and result manifest.

All YAML values not yet known are explicitly `null`; candidate mechanisms are disabled by default until an experiment enables them.

## Inputs and history

- `inputs/` contains the original v2 text and idea compilation supplied in this conversation.
- `history/v4/` and `history/v5/` preserve previous architecture/specification snapshots and supporting contracts.
- The v2-to-v3 audit retains the disposition of the original idea compilation and its Sentinel mappings.

## Important architectural distinctions

1. Compression (CSA/HCA), sparse selection (DSA/QSA/DHSA), retention, eviction and offload are separate mechanisms.
2. Recurrent state (including MoM banks), attention KV cache, learned neural memory, request-local editable slots and external application memory are separate state planes.
3. FFN-MoE and attention-MoE are separate sparse-capacity axes.
4. Standard residual, AttnRes, Gated Residual and mHC are alternatives until combination is earned by ablation.
5. Autoregressive and masked-diffusion objectives are separate training/generation tracks.

## Recommended implementation order

1. Freeze data, hardware, scale, workload and statistical contract.
2. Implement a dense autoregressive P0 reference.
3. Compare recurrent operator and MoM topology at P1.
4. Compare full, sparse and compressed attention at P2.
5. Compare FFN-MoE, attention-MoE and router initialization at P3.
6. Compare residual transport and MTP at P4.
7. Add memory/reasoning systems only after independent neural-model effects are measured.
8. Keep masked diffusion as a separate challenger.

## Validation limits

Static document validation is not model validation. The active environment is CPU-only (`torch 2.10.0+cpu`, no CUDA device); no model training, GPU-kernel correctness/performance or task benchmark run was performed. See `VALIDATION_REPORT.md` and `.json` for checks performed on the package itself.
