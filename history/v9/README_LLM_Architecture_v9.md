# LLM Architecture v9 Research and Training Package

## Contents
- `SOTA_LLM_Architecture_v9.md` / `.txt`: main research specification, including Section 33's pilot evidence.
- `LLM_Architecture_Audit_v8_to_v9.md` / `.txt`: v8 audit and v9 change rationale.
- `CHANGELOG_v8_to_v9.md`: concise architecture revision record.
- `TRAINING_PILOT_REPORT_v9.md`: actual small-scale training and usefulness results, data provenance and limitations.
- `candidate_profiles_v9.yaml`: P0–P9 candidate inventory; not a launch configuration.
- `benchmark_contract_v9.yaml`: comparison rules and required reporting fields.
- `kv_policy_matrix_v9.yaml`: independent cache-policy candidates.
- `implementation_roadmap_v9.yaml` / `implementation_roadmap.schema.json`: gated G0–G13 sequence and schema.
- `metric_definitions_v9.yaml` / `metric_definitions.schema.json`: metric formulas, denominators, units and comparison boundaries.
- `experiment_manifest.schema.json`: normative run manifest schema.
- `candidate_profiles.schema.json`, `benchmark_contract.schema.json`, `kv_policy_matrix.schema.json`: support schemas.
- `experiment_manifest.template.yaml`: deliberately incomplete template.
- `train_pilot.py`: miniature P0/P1/P1-LA training harness.
- `prepare_hf_dataset.py`: immutable-revision Hugging Face Dataset exporter with streaming split export and rights gates.
- `merge_pilot_results.py`, `verify_pilot_artifacts.py`: result aggregation and evidence integrity checks.
- `test_dataset_pipeline.py`: offline tests for streaming export, immutable revisions, rights gating, path safety and manifest/split hashing (17 tests).
- `results/`: six completed training runs, checkpoints, the ordinary and source-identified repeat runs, per-run/aggregate CSV summaries, manifests, loader smoke tests and verification report.
- `tinyshakespeare_30k.txt`: exact local sample used in the CPU pilot; SHA-256 is in `results/dataset_manifest.json`.
- `requirements.txt`, `pilot_requirements.txt`, `dataset_requirements.txt`: validator, recorded CPU-pilot and optional Hugging Face dataset dependencies.
- `validate_architecture_package.py`: offline package preflight validator.
- `VALIDATION_REPORT_v9.md` and `SHA256SUMS_v9.txt`: package validation and checksums.
- `history/`: prior v3–v8 specifications, audits and reports.
- `inputs/`: original v2 text, idea archive and earlier research packages.

## Validate the extracted package
Requires Python 3.10+ with `PyYAML` and `jsonschema` installed. Install validator dependencies with:

```bash
python -m pip install -r requirements.txt
python validate_architecture_package.py
python -m unittest -v test_validate_architecture_package.py
python -m unittest -v test_dataset_pipeline.py
python verify_pilot_artifacts.py
```

The package preflight validates the architecture specification and machine-readable contracts. The dataset tests validate the online-exporter's local streaming/manifest behaviour; they do not claim a live dataset download. The pilot verifier checks the six completed model/seed records, checkpoint hashes, repeatability, causal/state-reset invariants and split-bundle smoke test.

## Reproduce the miniature CPU training pilot

The recorded pilot used Python 3.13.5, PyTorch 2.10.0+cpu, one thread, context length 32, batch size four, 500 steps and two seeds. A full run is:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python train_pilot.py
```

Install the recorded pilot dependencies with `python -m pip install -r pilot_requirements.txt`. The defaults run P0, P1 and P1-LA for seeds `20261118,20261119`. Training checkpoints and the aggregated JSON are written under `results/`.

For separate one-seed runs and deterministic reproduction, use:

```bash
PILOT_SEEDS=20261118 PILOT_RESULTS_NAME=pilot_seed_20261118_current.json python train_pilot.py
PILOT_SEEDS=20261119 PILOT_RESULTS_NAME=pilot_seed_20261119_current.json python train_pilot.py
python merge_pilot_results.py results/pilot_seed_20261118_current.json results/pilot_seed_20261119_current.json --output results/pilot_results.json --checkpoint-dir results --repeat-file results/repeat_20261118_current.json
python verify_pilot_artifacts.py
```

## Export a full online dataset (not executed in this pilot)

Install `dataset_requirements.txt`, choose a full immutable 40-character dataset commit, and review the source/license and actual permitted uses. For WikiText-2 raw:

```bash
python -m pip install -r dataset_requirements.txt
python prepare_hf_dataset.py --dataset Salesforce/wikitext --config wikitext-2-raw-v1 --revision <40-character-commit-sha> --streaming --output-dir data/wikitext2
```

The command leaves rights pending. To record a reviewed approval, provide a specific license statement and allowed uses after a real human review:

```bash
python prepare_hf_dataset.py --dataset Salesforce/wikitext --config wikitext-2-raw-v1 --revision <40-character-commit-sha> --streaming --output-dir data/wikitext2 --rights-reviewed --license-statement "<reviewed statement>" --allowed-use training --allowed-use validation --allowed-use model_selection --allowed-use evaluation
```

Do not use `--rights-reviewed` as a convenience flag. The dataset page currently displays CC-BY-SA-3.0 and GFDL tags; resolve applicable terms before approving training or commercial use. The exporter uses immutable revisions, produces hash-locked train/validation/test files and a manifest checksum sidecar. The trainer verifies these before loading.

For an explicitly exploratory, non-production local smoke test on a pending-rights bundle only, `PILOT_ALLOW_UNREVIEWED_DATA=1` is required. Do not use that override to authorize production or commercial training.

## Important empirical result

The actual v9 pilot trained on a 29,926-byte excerpt retrieved from the Tiny Shakespeare GitHub source, not the complete online dataset. P1 slightly beat P0 on this tiny character task, but the smoothed bigram baseline beat every neural candidate; P1 was also slower in the naive CPU implementation (3.73× longer to train and 2.87× lower inference throughput). The full online dataset download, live Hugging Face API path, target accelerator kernels and frontier-scale training remain unverified. See `TRAINING_PILOT_REPORT_v9.md` and Section 33 of the main specification.

A passing validator is not scientific validation, proof of usefulness, license advice or an approval to promote the architecture.
