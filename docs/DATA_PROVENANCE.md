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

A structured candidate/source review was completed on 2026-11-19 in `docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md`. Candidates reviewed include TinyStories (CDLA-Sharing-1.0), WikiText (CC-BY-SA-3.0/GFDL), Open License Corpus/PubText (aggregate Apache metadata with source-specific licence categories and PII caveats), CuratedCorpus-5B (Apache metadata but insufficient source-level provenance), and Kyoto-Corpus (aggregate MIT metadata over multiple upstream datasets). **None is approved for this project yet.** No candidate dataset was downloaded or trained on as part of this review.

The review records terms/risks and a required acceptance manifest. Re-check the full immutable revision, every included subset's upstream licence, data-protection obligations and the intended commercial/checkpoint distribution use before selecting any candidate. A top-level repository licence tag is not, by itself, a source-rights audit.

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

Public download availability does not establish permission for a particular training or redistribution use. The candidate status and source-by-source due-diligence requirements are recorded in `docs/research/DATASET_RIGHTS_REVIEW_2026-11-19.md`.
