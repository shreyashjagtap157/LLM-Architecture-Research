# v9 Training Feasibility and Usefulness Pilot

**Run date:** 18 November 2026  
**Status:** SMALL CPU TRAINING COMPLETED; ONLINE FULL-DATASET TRAINING NOT COMPLETED  
**Evidence class:** exploratory small-scale reference training; not frontier-model validation

## Executive result

The v9 direction is trainable at miniature scale. Three 72.8k-parameter character-level models (dense attention P0, GDN-2-inspired hybrid P1, and local/global attention P1-LA) completed 500 optimizer steps for each of two seeds. All runs had finite losses/gradients, all trainable parameters received gradients, and causal/state-reset invariants passed.

Usefulness is **not established**. P1 modestly beat P0 in test NLL and top-1 accuracy on this two-seed tiny-corpus run, but it trained about 3.73 times more slowly and its naive CPU inference was about 2.87 times lower throughput. More importantly, a smoothed bigram model beat every neural candidate on both test NLL and top-1 accuracy. Generated continuations remained mostly character fragments, not coherent Shakespeare-like text. Do not promote P1 or P1-LA as winners.

The trained data was a 29,926-byte excerpt retrieved from an online GitHub repository, not the full online dataset. The full Tiny Shakespeare corpus and WikiText-2 are available online, but network access was blocked in this execution environment. A Hugging Face Jobs attempt returned HTTP 402 Payment Required; no paid job was created. The live `datasets.load_dataset` path has not been executed against a remote repository in this run.

## Online data sources and actual data used

### Actual training source

- Repository: `karpathy/char-rnn`
- Path: `data/tinyshakespeare/input.txt`
- Source blob SHA: `7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9`
- Full upstream file size reported by GitHub: 1,115,394 bytes (about 1.06 MiB)
- Locally materialized sample: 29,926 bytes (2.683% of reported full size)
- Local sample SHA-256: `3229111e25b798391b379a7707aad6b45cb8cfe8d26e0ec463a2120ec15e18be`
- Local split: sequential 80/10/10; 23,940 train characters, 2,993 validation, 2,993 test; vocabulary 58.
- This was a locally recovered excerpt from an online source. Exact byte parity with the complete upstream file or a particular upstream byte range was not independently verified.
- Rights: pending. Shakespeare's underlying works are public domain, but the repository provides no resolved transcription/edition rights determination for this use. Do not use this record as commercial-training approval.

The Hugging Face `karpathy/tiny_shakespeare` dataset card lists 1,115,394 downloaded bytes and separate train/validation/test splits. That full dataset was not downloaded in the pilot. See [dataset card](https://huggingface.co/datasets/karpathy/tiny_shakespeare) and [upstream repository data](https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare).

### Additional candidate dataset not downloaded

`Salesforce/wikitext`, configuration `wikitext-2-raw-v1`, publishes Parquet files for train, validation and test. Its dataset repository displays both `cc-by-sa-3.0` and `gfdl` licence tags. Those records require human review against the dataset card, upstream source and intended uses before training. No WikiText-2 data was fetched or used here. See [dataset file inventory](https://huggingface.co/datasets/Salesforce/wikitext/tree/main/wikitext-2-raw-v1).

## Model configurations

| Profile | Implementation | Parameters |
|---|---|---:|
| P0 | Two dense causal-attention blocks with FFNs | 72,832 |
| P1 | One GDN-2-inspired recurrent reference operator + one dense attention block | 72,928 |
| P1-LA | One local causal-attention block (window 16) + one dense global-attention block | 72,832 |

Shared settings: hidden width 64, four heads, two blocks, context 32, batch size four, AdamW learning rate `3e-4`, gradient clipping 1.0, 500 optimizer steps and 64,000 sampled training characters per run. Each neural profile was run with seeds `20261118` and `20261119`.

P1 is not an official GDN-2 reproduction. It uses simplified projections and sigmoid parameterizations with a naive sequential PyTorch scan. It does not implement NVIDIA's fused Triton kernels or chunkwise WY algorithm. NVIDIA reports GDN-2 experiments at 1.3B parameters and 100B FineWeb-Edu tokens; this pilot is only a toy-scale update-rule/learning feasibility check. Official reference: [NVlabs/GatedDeltaNet-2](https://github.com/NVlabs/GatedDeltaNet-2).

## Results

| Candidate | Test NLL (nats/char), mean ± sample SD | Top-1 next-char accuracy | Mean train seconds | Mean inference chars/s |
|---|---:|---:|---:|---:|
| P0 dense attention | 2.7708 ± 0.0444 | 25.62% | 2.06 | 270,437 |
| P1 GDN-2-inspired + attention | **2.7411 ± 0.0323** | **26.88%** | 7.66 | 94,097 |
| P1-LA local + global | 2.7598 ± 0.0451 | 25.89% | 1.90 | 269,015 |
| Smoothed unigram | 3.2555 | 15.12% | — | — |
| Smoothed bigram | **2.4318** | **27.45%** | — | — |

P1 vs P0:

- Test NLL: 1.07% relative improvement (lower is better).
- Top-1: +1.26 percentage points.
- Training time: 3.73× longer.
- CPU inference: 2.87× lower throughput.

P1 vs bigram:

- P1 test NLL is 12.72% higher (worse).
- P1 top-1 accuracy is 0.57 percentage points lower.

P1-LA vs P0 shows only a 0.40% mean NLL change and +0.27 top-1 percentage points. Two seeds do not justify selecting it.

The bigram comparison matters: the learned neural models did improve substantially over random initialization, but the more complex architecture did not beat a simple non-neural method on the held-out task. The generated 220-character samples were mostly malformed character fragments. This is not a useful language model yet.

## Correctness and repeatability checks

- GDN-2-inspired rank-one update vs explicit matrix equation: max absolute error `2.22e-16` in float64.
- Causality perturbation tests at positions 1, 8, 16 and 31: zero change to earlier-position logits.
- Request-local state reset after an intervening request: zero repeat difference.
- Every trainable parameter received a gradient on the first step.
- All 500 steps had finite losses and gradient norms; gradient clipping used `error_if_nonfinite=True`.
- Median training loss over final 25 steps was lower than over initial 25 steps for all six runs.
- Seed `20261118` was repeated for all three profiles; losses, accuracies and checkpoint SHA-256 values were bit-identical.
- A later seed-`20261118` repeat captured run-time SHA-256 values for the trainer, dataset exporter and dataset manifest. All three profiles matched the archived seed-20261118 metrics and checkpoint digests bit-for-bit. This provides source-identified evidence for that repeat, but does not retroactively identify the exact source bytes of the original six runs.
- `verify_pilot_artifacts.py`: 137 checks passed.

## Dataset importer verification

`prepare_hf_dataset.py` now:

- pins a full 40-character immutable Hugging Face dataset revision;
- supports streaming iterables to avoid materializing every split in memory;
- exports separate train/validation/test files with row, character, byte and SHA-256 metadata;
- writes `dataset_manifest.sha256` as an external manifest-integrity sidecar;
- defaults rights to pending; human review must specify a license statement and allowed uses;
- allows the trainer to fail closed on pending rights, with an explicit exploratory override only for non-production smoke tests.

The trainer verifies the manifest sidecar, immutable revision, each split hash and byte count, and rejects path traversal. The loader was exercised against a generated local split bundle, including a one-step P0 training smoke. The exporter/loader suite now passes **17/17** offline regression tests. These tests validate exporter/loader logic but do not exercise live network download. `datasets.load_dataset` against the remote `Salesforce/wikitext` dataset remains untested in this environment.

## Limitations

1. Only a 29.9 KB excerpt was trained; not the complete 1.11 MB Tiny Shakespeare source or WikiText-2.
2. Character-level modelling is not representative of subword-token LLM pretraining, instruction following, coding, reasoning or multilingual workloads.
3. Only two seeds were completed; no strong statistical conclusion or confidence interval is justified.
4. The P1 recurrence is a simplified sequential PyTorch reference, not the fused production-style operator.
5. CPU throughput is not an architecture-level GPU comparison. Optimized attention kernels were measured against a naive recurrent scan.
6. The live online dataset downloader, remote streaming, dataset license approval, CUDA kernels and frontier-scale model training remain unverified.
7. The exact `train_pilot.py` source digest was not captured at the moment the original six training runs launched. The script was subsequently hardened in the dataset-loading path. Model/optimizer definitions and the six result/checkpoint digests are preserved, but exact script-byte identity for the historic runs cannot be claimed.

## Decision and next experiment

**Current decision: feasibility pass; usefulness fail/inconclusive; no architecture promotion.** Keep P0 as the debug reference and P1/P1-LA as experimental candidates only.

Next required experiment:

1. Use a network-enabled environment to pin an immutable Tiny Shakespeare or WikiText-2 dataset revision.
2. Review licence/source rights and explicitly approve declared training, validation, model-selection and evaluation uses.
3. Run the streaming exporter on the full official splits; preserve raw file and manifest hashes.
4. Train P0, P1 and P1-LA on matched token/FLOP budgets, with at least five seeds, using identical data order and held-out evaluation.
5. Add a tuned small Transformer and n-gram controls; report mean, standard deviation and appropriate uncertainty intervals.
6. Run fused recurrent kernels and optimized attention on the same accelerator; profile memory, tokens/s, latency and energy.
7. Evaluate on larger word/subword language-model corpora, long-context retrieval and target-task benchmarks before any usefulness claim.

The current result does not justify adding further architectural mechanisms. It justifies completing a proper full-dataset and hardware-matched experiment first.
