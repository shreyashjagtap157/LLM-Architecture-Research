# Publishing and synchronizing this repository with GitHub

## Verified remote state — 2026-11-19

- Canonical repository: `https://github.com/shreyashjagtap157/LLM-Architecture-Research`
- Default branch: `main`
- Visibility reported by GitHub: **public**
- Verified `main` head: `d54ba5bd1e980fb40b4e80ae3758fe6b41548799`
- Verified root tree: `610f38e65b8e9042b39f7ca548c79a6197c12f6f`
- The root tree exactly matches the uploaded bundle's `main` tree and contains 145 tracked files. The original bundle history is retained.
- The remote history also contains an add/remove pair for an unrelated AetherOS prompt. That content is absent from the current tree; the commits were retained rather than rewriting history.

## Publication-risk warning

GitHub currently reports the repository as **public**. This project has no repository-wide licence. `NOTICE.md` and `docs/DATA_PROVENANCE.md` record unresolved rights for some user-supplied archives and the Tiny Shakespeare transcription. Do not infer redistribution, commercial-training or model-distribution permission from public accessibility. Consider switching the repository to private until a human rights review is complete. This document does not certify that review.

## Working from the existing GitHub repository

Prefer a fresh clone of the canonical remote:

```bash
git clone https://github.com/shreyashjagtap157/LLM-Architecture-Research.git
cd LLM-Architecture-Research
git status --short --branch
git log --oneline --decorate -8
```

Before committing work, fetch and fast-forward to the current `main`:

```bash
git fetch origin
git switch main
git merge --ff-only origin/main
```

Push only a normal fast-forward update:

```bash
git push origin main
```

Do not force-push or rewrite the recovered history. If the fast-forward fails, inspect and reconcile the divergent commits before pushing.

## Working from the supplied Git bundle

A clone made from a bundle normally configures `origin` to point to that bundle file. **Do not use `git remote add origin`** in that clone; `origin` already exists. Replace the bundle URL with the canonical GitHub URL, then fetch the current remote history:

```bash
git clone LLM-Architecture-Research.bundle LLM-Architecture-Research
cd LLM-Architecture-Research
git remote set-url origin https://github.com/shreyashjagtap157/LLM-Architecture-Research.git
git fetch origin main
git switch main
git merge --ff-only origin/main
```

Authenticate through Git Credential Manager or `gh auth login`; never place a personal access token in a command, URL, file or chat message. Verify the resulting upstream and local state:

```bash
git remote -v
git status --short --branch
git log --oneline --decorate -8
git ls-remote --heads origin main
```

## Reproducibility checks

From the repository root, validate the checksum manifest and the v9 pilot artifacts before changing research outputs:

```bash
sha256sum --check SHA256SUMS.txt
python -m pip install -r experiments/v9_character_pilot/requirements.txt -r experiments/v9_character_pilot/pilot_requirements.txt pytest
python -m pytest -q experiments/v9_character_pilot/test_dataset_pipeline.py experiments/v9_character_pilot/test_validate_architecture_package.py
python experiments/v9_character_pilot/verify_pilot_artifacts.py
python experiments/v9_character_pilot/prepare_hf_dataset.py --help
```

Do not invoke `train_pilot.py --help`: the script is an experiment runner and starts the six-run CPU pilot rather than displaying help. It can overwrite files under `experiments/v9_character_pilot/results/`.

The checksum manifest does not hash itself. Regenerate it whenever tracked research artifacts or documentation change.
