# Source, data and licensing notice

This repository combines original research documents, generated technical specifications, user-supplied source files, third-party research references, and training-pilot artifacts. **No single repository-wide licence is declared.** Do not assume that the entire repository or every included file is available for unrestricted reuse.

## User-supplied inputs

- `inputs/v2_baseline.txt` and `inputs/ideas20_full_compilation.zip` were supplied by the project owner. Their inclusion here does not grant third-party redistribution rights beyond the owner's rights in those materials.
- Historical research ZIPs under `inputs/` and `history/` are preserved for continuity; respect the licensing/provenance notices inside each archive.

## Dataset excerpt

The v9 pilot's `tinyshakespeare_30k.txt` is a 29,926-byte excerpt retrieved from `karpathy/char-rnn`, path `data/tinyshakespeare/input.txt`, Git blob SHA `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`. The source repository metadata available during the experiment did not declare a licence. Shakespeare's underlying works are public domain, but this does not by itself establish rights to a particular transcription, compilation or distribution. Its status remains `pending_exploratory_only`.

Do not use the bundled excerpt for official or commercial training until the exact source/transcription rights are independently reviewed and the allowed uses are recorded.

## Research references

The architecture specifications cite external papers, technical reports and code repositories. Those materials remain subject to their respective licences and terms. A citation is not a licence grant.

## Code

The v9 training scripts and tests are preserved as historical project artifacts. Check their own headers and dependency/source licences before redistribution. No blanket MIT or Apache licence is added to this mixed-content repository.
