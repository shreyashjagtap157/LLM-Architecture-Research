# Changelog — v5 to v6

- Added DeepSeek-V4 as a production/vendor-reported architecture composition, including an explicit 284B-vs-285B Flash parameter source discrepancy.
- Added CSA and HCA as separate compressed-historical-attention mechanisms; clarified sliding-window branch and model-specific configuration values.
- Added shared K=V MQA, partial RoPE, attention sinks and grouped low-rank output projection as independent attention design variables.
- Added mHC residual transport with an abstract tensor contract, Birkhoff/Sinkhorn invariants and depth-placement ablation.
- Added MoM as a multi-bank recurrent-state topology with state-memory and bank-health metrics.
- Added SwitchHead attention-MoE as a separate capacity axis from FFN-MoE, with scale caveats.
- Added DeepSeek-V4 hash-router bootstrap as a tokenizer/checkpoint-bound ablation.
- Added compressed-cache causal availability, source-span versioning and stale-entry invalidation contracts.
- Added cost-model terms and benchmark metrics for compressor cost, compressed KV, MoM banks, mHC numerics and attention-MoE dispatch.
- Updated P1–P4 research stages and decision record.
- Added references R80–R86.
- Preserved v5 and its support files under `history/v5/`.
- No training or GPU benchmark results are claimed.
