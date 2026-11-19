# Dataset rights and provenance review — 2026-11-19

**Review scope:** candidate text corpora for the next small subword-model experiment and eventual commercial/release ambitions.  
**Review date:** 19 November 2026.  
**Decision:** **No corpus is approved for this project yet. No new dataset has been downloaded or used for training.**  
**Nature of review:** preliminary source-card and licence-text review, not legal advice or a final rights opinion.

A dataset's Hugging Face licence tag is a useful discovery field, not proof that every row is lawfully licensed by the person/entity entitled to license it, that every upstream source is covered, or that the licence permits every intended use. We must distinguish (1) licence of the curated database/selection, (2) licence of each underlying item, (3) privacy/data-protection and confidentiality constraints, and (4) our intended activity: local training, evaluation, publication of source data, model-weight distribution, hosted service and commercial use.

## 1. Candidate assessment

| Candidate | Public card/licence facts observed | Blocking issues and project decision |
|---|---|---|
| [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) | The current card labels the dataset CDLA-Sharing-1.0 and describes short stories generated with GPT-3.5 and GPT-4. The card shows a train split with over two million rows and a validation resource. | **Conditional small-scale candidate only; not yet approved.** CDLA-Sharing-1.0 grants rights to use and publish data subject to conditions. If we publish received data or Enhanced Data, the agreement requires the unmodified agreement, retained attribution and notices that changes were made; the agreement says it imposes no obligations/restrictions on publication of Results. Whether any particular derived artifact is Data, Enhanced Data or Results must not be guessed. Preserve exact upstream notices, identify the selected train/validation files and revision, and obtain legal review of how our weights and distribution plan interact with the agreement before use. This is synthetic short-story data, not a broad-domain language corpus. |
| [Salesforce WikiText](https://huggingface.co/datasets/Salesforce/wikitext) | The dataset card shows CC-BY-SA-3.0 and GFDL; its summary says WikiText was extracted from verified Good and Featured Wikipedia articles. | **Do not use for the first project training run without legal review.** Attribution and ShareAlike obligations, GFDL terms, version/variant selection, and implications for distributing trained weights need a documented assessment. The dataset is useful for reproducible language-modelling research, but it is not a low-friction commercial-rights choice. |
| [Open License Corpus / PubText](https://huggingface.co/datasets/kernelmachine/open-license-corpus) | Card metadata says Apache-2.0 and describes a 228B-token corpus. Its source table separately lists Public Domain, MIT/BSD/Apache and CC-BY-SA portions. The card explicitly says the corpus is not universally safe to use and recommends legal review. | **Potential large-corpus source for a filtered subset, not approved as a single all-in dataset.** The card says this repository contains training data only; the paper used separate Pile validation data. It also says it does not perform additional PII filtering and warns that privacy risks may remain. A candidate subset must be selected source-by-source, explicitly excluding or separately reviewing CC-BY-SA components as required by the intended release policy. We need a defensible validation/test source, source-level attribution and PII/sensitive-data review. The top-level Apache tag does not erase the listed component licences. |
| [CuratedCorpus-5B](https://huggingface.co/datasets/MiamnusAI/CuratedCorpus-5B) | Its card declares Apache-2.0 and describes a five-billion-token corpus with train and validation shards. It describes educational web, textbooks, STEM/math and synthetic narratives; the card claims deduplication, heuristic filtering, PII scrubbing and toxicity filtering. | **Not approved pending source provenance.** The card's high-level source descriptions do not provide a sufficiently granular upstream dataset/item licence inventory for us to independently verify rights for the educational-web and textbook portions. The card's Apache statement and filtering claims are publisher assertions, not an independently audited chain-of-title. Obtain the exact source inventory, upstream licences/permissions, removal process and immutable snapshot; verify the claims and dataset bytes before relying on it. |
| [Kyoto-Corpus](https://huggingface.co/datasets/Nikity/Kyoto-Corpus) | The card labels the aggregate dataset MIT and lists multiple upstream datasets, including UltraChat, SmolTalk, WildChat, WizardLM, IFEval-like data, Tulu personas, MMLU, GSM8K, MathQA and MetaMathQA. It records the upstream source name per row. | **Not approved pending source-by-source and privacy review.** An aggregate MIT tag cannot by itself establish that every third-party item can be relicensed under MIT. The WildChat and other conversational sources need specific personal-data, consent, retention and source terms review. The card describes instruction-tuning data; it is not a general pretraining corpus. |
| Bundled Tiny Shakespeare excerpt | NOTICE.md and docs/DATA_PROVENANCE.md say the 29,926-byte excerpt came from a public GitHub blob whose repository metadata did not establish a licence for the transcription. | **Excluded from official and commercial training pending rights review.** Public-domain underlying Shakespeare works do not automatically clear a particular transcription, compilation or distribution. It remains exploratory-only and must not be silently reused as a rights-cleared corpus. |

## 2. Licence-specific observations

### CDLA-Sharing-1.0

The official CDLA text grants a worldwide, non-exclusive right to Use and Publish Data subject to the agreement's conditions. If received Data or Enhanced Data is published, the agreement requires the unmodified agreement to accompany it, attribution to be retained, modifications to be marked where applicable, and recipients not to be restricted from publicly publishing it. Section 3.5 states that the agreement imposes no obligations or restrictions on use/publication of Results.

This is not equivalent to saying "TinyStories is MIT/Apache". It is a data-sharing agreement with publication conditions. We must preserve its exact text and notices, and legal review must decide how it applies to the exact artifacts we intend to release. Do not impose additional restrictions on covered Data/Enhanced Data without reviewing Section 3.3. The distinction between dataset/Enhanced Data and model outputs/weights is an issue for qualified review, not an assumption in this project.

### CC-BY-SA / GFDL material

The WikiText card displays both CC-BY-SA-3.0 and GFDL. That means source/version attribution and ShareAlike obligations must be analyzed against the intended use and release model. Do not classify this as "commercially forbidden" based only on the licence name; equally, do not assume a training run or distributed weights are automatically outside its obligations. Obtain advice for our concrete training, model release and hosted-service plans.

### Open License Corpus

The OLC card's summary is unusually helpful because it enumerates source groups and their declared licences. It still has two important limitations for our workflow: this Hub repository exposes the training data only, with the paper using a separate Pile validation set; and the card explicitly warns that PII may remain because no additional PII filtering was applied. We may investigate a smaller, source-filtered subset after verifying each subset's exact configuration, source, licence and hashes. We must not label the entire aggregate as "Apache-only" or "PII-cleared".

## 3. Required acceptance evidence before any dataset is selected

For the chosen immutable revision, archive the dataset card, licence text, Terms of Use and relevant source notices. Build a row/subset-level provenance manifest sufficient to identify every included source and its declared licence. At minimum record:

- repository ID, dataset configuration, full immutable commit SHA, source URL and access date;
- exact files/subsets included and excluded, with SHA-256 hashes and sizes;
- upstream origin/source and source-level licence/permission, attribution and ShareAlike requirements;
- allowed uses separately for training, validation, checkpoint/model-weight distribution, hosted service, research publication and commercial use;
- data controller/processor and privacy obligations where applicable; personal/sensitive data, secrets, contact details and removal/opt-out handling;
- deduplication, benchmark contamination, redaction, language/domain filtering and preprocessing code hashes;
- train/validation/test identities, non-overlap checks, split hashes and the policy that test data is used once after selection;
- a reviewer and approval record. A machine-readable licence tag alone is not approval.

Keep a copy of the exact source terms in the run's evidence bundle. If a source's rights are ambiguous, exclude that source rather than treating uncertainty as permission. Keep a path to remove/refresh affected training data and document whether already-trained checkpoints need to be withdrawn or retrained.

## 4. Project decision and next step

No dataset is approved by this review. Do not download an entire large corpus or start training as a side effect of testing an adapter. The next task is to choose one narrowly scoped candidate and verify its source snapshot, rights, validation/test plan and intended model-release constraints before data materialization.

For a low-cost synthetic-language experiment, TinyStories is the closest conceptual fit, subject to CDLA-specific review and preservation of agreement/attribution. For a larger language-model pretraining experiment, OLC may offer a route through explicitly enumerated source subsets, but its privacy and held-out validation/test gaps need independent work. CuratedCorpus-5B should not be selected solely on its aggregate Apache tag; source-level provenance is a prerequisite. WikiText and Kyoto remain higher-obligation/mixed-source candidates.

Until a candidate passes the gate, continue synthetic tensor/operator testing only. This document records metadata and terms reviewed on 19 November 2026; re-check the live cards, immutable commits and legal texts before any later run.

## 5. Sources reviewed

- TinyStories dataset card: https://huggingface.co/datasets/roneneldan/TinyStories
- CDLA-Sharing-1.0 agreement: https://cdla.dev/sharing-1-0/
- CDLA FAQ (Sharing vs Permissive): https://cdla.dev/faq-resources/faq/
- WikiText dataset card: https://huggingface.co/datasets/Salesforce/wikitext
- Open License Corpus / PubText dataset card: https://huggingface.co/datasets/kernelmachine/open-license-corpus
- CuratedCorpus-5B dataset card: https://huggingface.co/datasets/MiamnusAI/CuratedCorpus-5B
- Kyoto-Corpus dataset card: https://huggingface.co/datasets/Nikity/Kyoto-Corpus

This is a research due-diligence note, not a legal opinion. It does not certify title, licence compliance, privacy compliance or commercial training permission for any listed dataset.
