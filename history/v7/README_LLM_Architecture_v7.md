# LLM Architecture v7 Research Package

**Snapshot:** 18 November 2026  
**Status:** Research specification and audit; model is not trained or benchmarked.

## Start here

1. `SOTA_LLM_Architecture_v7.md` — main architecture specification.
2. `LLM_Architecture_Audit_v6_to_v7.md` — what changed and why.
3. `CHANGELOG_v6_to_v7.md` — concise revision record.
4. `VALIDATION_REPORT_v7.md` — reproducible static validation results and limitations.
5. `candidate_profiles_v7.yaml` — hardware-neutral experiment profiles.
6. `kv_policy_matrix_v7.yaml` — cache-policy comparison matrix.
7. `benchmark_contract_v7.yaml` — benchmark and measurement contract.
8. `SHA256SUMS_v7.txt` — file integrity manifest.

The v2 baseline and IDEAS20 archive are retained in the package under `inputs/`. Prior v3–v6 documents and revision records are retained under `history/` for provenance.

## Important limitation

The proposed combination has not been trained. Static validation only checks document/configuration integrity, not numerical correctness, convergence, model quality, kernel behavior or performance. Candidate settings remain symbolic until target hardware, model scale, data and compute budget are supplied.
