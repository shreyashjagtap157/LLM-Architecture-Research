# Data provenance

## v9 miniature character-model pilot

- Source repository: `karpathy/char-rnn`
- Upstream path: `data/tinyshakespeare/input.txt`
- Git blob SHA: `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`
- Upstream file size reported by GitHub: 1,115,394 bytes
- Materialized excerpt used in v9: 29,926 bytes
- Local SHA-256 from v9 report: `3229111e25b798391b379a7707aad6b45cb8cfe8d26e0ec463a2120ec15e18be`
- Split: 23,940 train / 2,993 validation / 2,993 test characters; vocabulary 58
- Status: `pending_exploratory_only`

The excerpt is preserved at `experiments/v9_character_pilot/tinyshakespeare_30k.txt`. Exact byte parity to a declared upstream range/full dataset was not independently verified. GitHub repository metadata did not resolve a licence for the particular transcription. Shakespeare's underlying works being public domain is not sufficient proof that this transcription/compilation may be used for commercial training or redistribution.

## Other dataset candidates

WikiText-2 was considered, but no full remote download/training run was completed. Its repository displays multiple licence tags that require human review against its dataset card and source terms. The v9 report has direct source links and additional detail.

## V11 dynamic-compute session data

Conversation updates describe use of the same tiny character corpus and synthetic parity sequences. However, V11 split files, dataset manifests, trainer code and exact source digests were not recovered into this snapshot. Results are therefore session-reported and must not be represented as a fully reproducible dataset campaign.

## Future data acceptance requirements

Before official or commercial training, record:

- immutable dataset revision/commit and source URL;
- exact split identities and hashes;
- license/terms and allowed uses for training, evaluation, weight distribution and commercial use;
- attribution/retention/redistribution requirements;
- deduplication, contamination and benchmark-exposure controls;
- processing code and dependency hashes.

Public download availability does not establish permission for a particular training or redistribution use.
