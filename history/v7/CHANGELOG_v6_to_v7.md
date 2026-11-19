# Changelog — v6 to v7

- Updated review cutoff to 18 November 2026.
- Added Inkling 2 Small as a local/global Transformer-MoE control, including its block attention residuals, gated attention, per-token sink, four-tap causal convolutions, layer-specific quantization and deployment constraints.
- Reconciled Inkling 2's 1M model-card context with staged training lengths: 512K H2, 1M H3, and 256K RL context with compaction.
- Added block-attention-residual and short-convolution implementation contracts, separate from AttnRes, Gated Residual and mHC.
- Added DSpark semi-autoregressive draft/verify as a distinct alternative to MTP, with Markov/RNN and confidence heads.
- Added Nemotron-Labs-Diffusion tri-mode generation (AR, diffusion, diffusion-draft/AR-verify) as an independent challenger.
- Added Seer, ShotKV/KVFundaBench, Distance-KV and LiteKV as distinct KV policy candidates with explicit non-composition rules.
- Expanded KV-cache evaluation to high-density reasoning and semantic-unit preservation.
- Added an Inkling 2 training case study covering staged data/context curriculum and vendor-reported RL scale; prohibited cost extrapolation from those figures.
- Added synchronous pre-tool authorization and asynchronous rollout auditing to the application safety contract, including network-circumvention tests.
- Added v7 candidate profiles, tri-mode contracts, cost-model terms, test-matrix rows, and references R87–R94.
- Preserved v6 unchanged.
- No model training, GPU kernel validation or benchmark execution is claimed.
