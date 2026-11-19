# Validation report — repository snapshot 2026-11-19

## Environment

- Python: 3.13.5
- PyTorch: 2.10.0+cpu
- CUDA available: no
- pytest: 9.0.2
- OS/runtime: Linux container

## Executed checks

| Check | Result |
|---|---:|
| `python -m pytest -q test_dataset_pipeline.py test_validate_architecture_package.py` in `experiments/v9_character_pilot/` | 44 passed |
| `python verify_pilot_artifacts.py` in `experiments/v9_character_pilot/` | 137 checks passed |
| Full v9 repeat (two seeds × three profiles, 500 steps/run) | completed 2026-11-19 07:52:46 UTC; six checkpoints written; all run records contain source/data hashes |

The v9 repeat used trainer SHA-256 `64bbc094f1773c31307dce0215c182bc06bcfe6b051391169d34885bf7489f4a`, exporter SHA-256 `c119967fb5548fc24478c14fade3d28b41aa38e4023ddb1141047105f41a4d00`, and dataset-manifest SHA-256 `f575acf98eeb16196f78521061849a93def228b9b582366f78980bd0ad8ea597`. All three profiles over both seeds completed; metrics match the recorded pilot aggregate.

These are v9 artifact-backed tests and a new source-identified v9 repeat. They do not validate or rerun V11 dynamic-computation session experiments because their V11 source/results were not recovered into this snapshot.

## Important runner behavior

`train_pilot.py` does not have an argparse help path. Invoking it with `--help` still launches the full six-run experiment and overwrites `results/`. This happened during packaging; the run completed successfully and is recorded above. Do not use `--help` as a dry-run.

## Not run / not claimed

- No official subword LLM training.
- No full online dataset download or rights approval.
- No CUDA/GPU kernel benchmark or multi-GPU training.
- No end-to-end V11 dynamic-computation reproduction.
- No GitHub push.

## Package-level checks to be performed before release

- `git status --porcelain` clean after final commit.
- `git bundle verify` passes.
- A fresh clone from the bundle has `main` at the expected commit.
- ZIP extraction contains all expected source/docs/inputs and excludes `.git`/temporary files.
- SHA-256 manifests verify after extraction.
