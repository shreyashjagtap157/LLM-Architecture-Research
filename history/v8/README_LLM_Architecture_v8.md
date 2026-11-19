# LLM Architecture v8 Research Package

**Snapshot:** 18 November 2026  
**Status:** literature-backed research specification; no model trained or benchmarked.

## Start here

1. `SOTA_LLM_Architecture_v8.md` — complete architecture specification.
2. `LLM_Architecture_Audit_v7_to_v8.md` — new research, corrections and remaining blockers.
3. `CHANGELOG_v7_to_v8.md` — revision summary.
4. `VALIDATION_REPORT_v8.md` — measured static checks and validation limits.
5. `candidate_profiles_v8.yaml` — candidate architecture/generation/training experiment profiles.
6. `kv_policy_matrix_v7.yaml` — KV-cache policy comparison matrix carried forward from v7.
7. `benchmark_contract_v8.yaml` — benchmark and measurement contract.
8. `SHA256SUMS_v8.txt` — package integrity manifest.

`inputs/` retains the original v2 baseline, IDEAS20 archive, v3 package and previous v7 package. `history/` retains v3–v7 specifications and revision records.

## Important limitation

This is a research design, not a trained model. Static validation checks the package, Markdown structure, citations and YAML syntax only. It does not establish numerical correctness of kernels, convergence, capability, safety or performance. Candidate settings remain symbolic until a target hardware, model scale, data and compute contract is supplied.
