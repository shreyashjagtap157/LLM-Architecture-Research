# Validation report — LLM Architecture Research snapshot 2026-11-19

## Snapshot identity

- Repository: `LLM-Architecture-Research`
- Branch: `main`
- Final snapshot commit for this report: recorded in Git at packaging time; inspect with `git rev-parse HEAD`.
- Repository content checksum manifest: `SHA256SUMS.txt` (144 payload files; excludes itself and `.git/`).
- Local Git history: four commits, beginning with the assembled research import. No upstream Git history was available in the recovered workspace.
- Remote: none configured.

## Environment

- Python: 3.13.5
- PyTorch: 2.10.0+cpu
- CUDA available: no
- pytest: 9.0.2
- OS/runtime: Linux container

## Executed model and artifact checks

| Check | Result |
|---|---:|
| `python -m pytest -q test_dataset_pipeline.py test_validate_architecture_package.py` in `experiments/v9_character_pilot/` | 44 passed |
| `python verify_pilot_artifacts.py` in `experiments/v9_character_pilot/` | 137 checks passed |
| Source-identified v9 repeat (two seeds × three profiles, 500 steps/run) | completed 2026-11-19 07:52:46 UTC; six checkpoints written; run records contain source/data hashes |

The v9 repeat used trainer SHA-256 `64bbc094f1773c31307dce0215c182bc06bcfe6b051391169d34885bf7489f4a`, exporter SHA-256 `c119967fb5548fc24478c14fade3d28b41aa38e4023ddb1141047105f41a4d00`, and dataset-manifest SHA-256 `f575acf98eeb16196f78521061849a93def228b9b582366f78980bd0ad8ea597`. All three profiles over both seeds completed; metrics match the recorded pilot aggregate.

These are v9 artifact-backed tests and a source-identified v9 repeat. They do not validate or rerun the V11 dynamic-computation session experiments because their source code and raw result sets were not recovered into this snapshot.

## Executable entry-point warning

`train_pilot.py` does not implement an argparse help path. Invoking it with `--help` still launches the full six-run experiment and overwrites the current `results/` outputs. This happened during packaging; the run completed successfully and is recorded in `TRAINING_RUN_2026-11-19.md`. Do not use `--help` as a dry-run.

## Portable bundle validation

| Check | Result |
|---|---|
| `git bundle create --all` | Passed |
| `git bundle verify` | Passed; complete local history recorded |
| Fresh clone from Git bundle | Passed; checked out `main` at the expected snapshot commit |
| ZIP extraction | Passed; 145 files present before tests create local caches |
| Root SHA-256 manifest after ZIP extraction | 144/144 entries passed |
| Tests from fresh ZIP extraction | 44 passed |
| v9 artifact verifier from fresh ZIP extraction | 137 checks passed |
| Working repository status before final packaging | Clean |

The Git bundle preserves the four-commit local history. The ZIP contains the source tree and documents but intentionally excludes `.git`; use the bundle when preserving commit history matters.

## Research status and non-claims

- No official subword LLM training.
- No full online dataset download or rights approval.
- No CUDA/GPU kernel benchmark or multi-GPU training.
- No end-to-end V11 dynamic-computation reproduction.
- No measured production usefulness, frontier capability, or SOTA superiority.
- No GitHub push; publication blocker and exact steps are documented in `PUBLISHING.md`.

The current architecture specification recovered in this repository is v10. The runnable code baseline is the v9 miniature character-model pilot. Dynamic-computation findings from the 18–19 November session are documented as session-reported where source-level artifacts were unavailable; they must be reimplemented and independently reproduced before being treated as validated results.
