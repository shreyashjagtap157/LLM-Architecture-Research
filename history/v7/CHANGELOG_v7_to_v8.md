# Changelog — v7 to v8

- Added the four-axis architecture co-design map: processing structure, transformation dynamics, state management and output commitment.
- Added DiffuSpec as a diffusion-lattice drafter with causal-consistent path selection and AR target verification.
- Added DEdit as an iteratively edited diffusion drafter, with explicit measurement of edits that damage already-correct positions and the cost of added editing passes.
- Added Speculative Refinement as a distinct AR-seeded masked-diffusion refinement pipeline; clearly classified as changed-output generation unless a separate exact verifier is used.
- Added the four speculative co-design dimensions and four cumulative promotion gates: Representability, Verifiability, State Realizability and Profitability.
- Made exactness conditional on the tractable final post-search/post-edit proposal probability and the complete target acceptance/correction rule; causal path validity alone is not enough.
- Added CGLS progressive layer stacking / data-difficulty curriculum as a 2×2 factorial training ablation, not a default training recipe.
- Expanded benchmark controls for scaffold effects, metric divergence, non-AR harness correctness, refinement damage and output-distribution claims.
- Added P8 and P9 independent generation profiles and updated the end-to-end Mermaid architecture.
- Added a Pareto-based selection and stopping rule; infeasible candidates cannot trade away mandatory requirements for average score.
- Added references R95–R100 and updated static validation manifests.
- Preserved v7 unchanged; no model training or GPU benchmark is claimed.
