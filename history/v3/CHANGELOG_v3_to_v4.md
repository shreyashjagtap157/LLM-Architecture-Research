# v3 to v4 Change Log

**Date:** 17 November 2026

## Main additions

- Added DeepSeek-V3.2 / DSA and separated token-level DSA selection from Qwen3.8 micro-block QSA.
- Added Multi-layer Indexer (MLI, 2 Nov 2026) as an emerging DSA serving optimization, with explicit cache-version and selection-lag safety checks.
- Added FlashAttention-4 as a Blackwell-specific attention kernel option.
- Expanded memory architecture from three to four separate planes: recurrent state, attention KV cache, learned neural memory, external application memory.
- Clarified that sparse attention compute reduction does not imply KV storage reduction or eliminate indexer scans.
- Added component/tensor contracts, state-isolation invariants, cost equations, matched-data/matched-compute evaluation, benchmark contamination controls and failure-mode tests.
- Added P0–P5 staged candidate profiles and a stop/promotion order.
- Added optional test-time training as a separately evaluated mechanism.
- Corrected LatentMoE bibliographic title, Nemotron shared-expert description, duplicated heading and earlier validation-report template bug.

## New sources

R67 DeepSeek-V3.2; R68 MLI; R69 FlashAttention-4; R70 RULER; R71 LongBench v2; R72 PagedAttention; R73 ConStat; R74 OpenAI SWE-bench Verified audit; R75 Self-Guided TTT.

## Explicitly unresolved

Model dimensions, parameter counts, expert counts, optimizer settings, hardware-specific kernels, context length and cost targets remain null until a deployment/training envelope is supplied. This is deliberate, not a missing default.
