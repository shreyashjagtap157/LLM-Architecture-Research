# LLM Architecture Audit — v2 to v3
*Independent technical review of the supplied “The Synthesized SOTA LLM Architecture (v2 — with upgrade paths)” and the accompanying `ideas20_full_compilation.zip`.*

**Review date:** 17 November 2026  
**Primary source:** `Pasted text(20261117-071440).txt` (the 179-line v2 document)  
**Supplementary archive:** `ideas20_full_compilation.zip`  
**Rewritten specification:** `SOTA_LLM_Architecture_v3.md` and identical Markdown-source `.txt` file.

---

## 1. Executive Assessment

The v2 document has a useful high-level instinct: combine efficient sequence processing, sparse capacity, memory, validation-oriented reasoning and efficient decoding. Its strongest elements are the recognition that attention and recurrent sequence models have complementary trade-offs; auxiliary-loss-free MoE balancing is worth considering; grammar-constrained output needs semantic checks; and verifiable feedback should be used where possible.

However, v2 is not yet a defensible SOTA architecture specification. Its principal weaknesses are not merely missing citations. It conflates different system layers, uses several overconfident claims, assumes that attractive mechanisms compose without interaction costs, and proposes multiple components whose training or inference behavior is not specified sufficiently to reproduce.

### 1.1 High-severity findings

| ID | Finding | Severity | v3 action |
|---|---|---:|---|
| A-01 | The thesis says “Five pillars” but lists six, including cross-cutting. | Low | Corrected to three architectural strata and explicit model/runtime/application boundaries. |
| A-02 | Citations are malformed fragments with no usable bibliography or source provenance. | Critical | Replaced with a numbered research register and stable paper / official technical-report URLs. |
| A-03 | Nemotron 3 Nano expert count is wrong: v2 says 256 total experts per MoE layer. | High | Corrected to 128 routable experts, 6 activated per token, with official report values for total/active parameters. |
| A-04 | Fixed 1:7 attention-to-Mamba ratio is framed as generally validated and near-optimal. | High | Reclassified as one design point; v3 mandates matched layer-schedule ablations. |
| A-05 | MLA is treated as though it supplies a four-tier persistent memory system. | Critical | Separated recurrent model state, attention KV cache and external agent memory. |
| A-06 | Counterfactual KL is treated as a universal causal score for individual KV entries. | High | Recast as an expensive, output- and query-conditioned diagnostic; a learned proxy remains optional research. |
| A-07 | Monotonic reasoning forbids re-entry into prior states. | High | Replaced by a bounded state graph with explicit revision and backtracking. |
| A-08 | Self-consistency is called an oracle backend. | High | Kept only as a weak agreement heuristic; formal/empirical validators are separately typed. |
| A-09 | Entropy is used as a proxy for certainty, progress and commitment. | High | Rejected as a universal stop/confidence signal. |
| A-10 | MTP is described as if it guarantees approximately 2× inference acceleration. | High | Separated MTP training, prediction heads and speculative decoding; no fixed speedup claim. |
| A-11 | One 256-dimensional uncertainty tensor mixes signals with different meanings and calibration. | High | Replaced with a typed control plane and independently scoped measurements. |
| A-12 | Claimed 1.3–1.6× training overhead and inference-time zero/negligible overhead are unsupported. | High | Removed numeric estimates; require actual hardware and kernel profiling. |
| A-13 | The output decoder reuses the KV causal-influence proxy as a semantic token filter. | Critical | Removed; semantic constraints require explicit schema/domain validators. |
| A-14 | The synthesis treats model architecture, serving policies and agent orchestration as a single neural pipeline. | Critical | Separated model, inference/serving and agent/application layers. |

### 1.2 Overall verdict

**v2 is a promising concept note, not a validated or implementation-ready architecture.** The v3 specification preserves the useful design direction but substantially changes the architecture and evidence model. It intentionally does not assert that the new combination is better than an existing model until matched ablations have been run.

---

## 2. Audit Method and Scope

The review used the supplied 179-line v2 source, all files in the supplied ZIP, and public papers / official model technical reports and repositories available by 17 November 2026. Priority was given to primary papers, conference proceedings, official technical reports and official repositories. Vendor-reported numbers are identified as model-specific reports; preprints are not promoted to production evidence.

“Search all LLM architectures available online” cannot literally be completed: proprietary systems do not disclose full architectures, new papers and checkpoints appear continuously, and many papers describe variants rather than reproducible model families. The practical scope used for v3 is therefore **major public architecture families and representative implementations relevant to a modern general-purpose LLM**, not every architecture ever published.

The evidence labels in v3 are intended to prevent these distinct statements being collapsed:

- a mathematical property is supported;
- a research implementation works at a specified scale;
- a technique appears in a released model;
- a specific combination has been validated at the target scale and hardware.

These are four different levels of evidence.

---

## 3. Detailed Review of the Supplied v2 Document

### 3.1 Section 0 — Design thesis

**Issue:** Five pillars are listed, but the cross-cutting uncertainty bus is described as a sixth pillar. The document also says that it combines a “proven core” with proposed upgrades, while several items called proven are not actually demonstrated by the supplied citations.

**Correction:** v3 defines three strata—neural model, inference/serving runtime, agent/application control plane—and then defines measurable model objectives. This keeps a neural component distinct from orchestration code and runtime policy.

### 3.2 Section 1 — Hybrid Attention / State-Space interleave

**Issue 1: fixed ratio generalized beyond its evidence.** A 1:7 or one-attention-in-eight pattern is a configuration used by particular hybrid designs. Other public models use different schedules and sequence mixers. Jamba, Qwen3-Next, Qwen3.8-Flash-Next and Nemotron 3 Nano are evidence for hybrid designs, not evidence for one universal ratio.

**Issue 2: benchmark figures are presented without source identity and protocol.** Statements such as “no measurable performance difference,” “up to 3× inference speed,” “95% exact-match recall,” and “256K on a single 80GB GPU” need the precise model, benchmark, batch size, hardware, cache regime and baseline. Those details are absent from the v2 source fragments.

**Issue 3: layer-type gating is under-specified.** The proposal computes both attention and Mamba paths during the supernet phase, then hard-routes at inference. The asserted 1.3–1.5× cost is not derived. It ignores recurrent-state initialization/continuity, cache compatibility, path-specific normalization, kernel launch and dispatch overhead, gradient interactions and the cost of obtaining representative gate decisions.

**v3 disposition:** preserve the hybrid hypothesis, reject a universal ratio, and search fixed schedules first. GDN-2, Mamba-2/3, Muir, GQA, MLA and indexed sparse attention become candidates in a controlled operator/schedule sweep. Conditional layer-type routing is deferred until a fixed schedule is a validated baseline.

### 3.3 Section 2 — Sparse MoE

**Issue 1: Nemotron 3 Nano specification.** The baseline says 256 total experts and one shared expert. NVIDIA's technical report says 128 routable experts and 6 activated per token; total parameters are 31.6B and active parameters are 3.2B (3.6B including embeddings). The exact shared-expert reporting should be read from the specific released configuration/report rather than copied from a secondary summary; v3 avoids repeating an unnecessary shared-expert count. [R28]

**Issue 2: load-balancing theory is overstated.** Even if a particular update can be interpreted through a primal-dual or assignment formulation under specified assumptions, that does not establish stability for every implementation, router noise level, distributed layout or hyperparameter choice.

**Issue 3: sequence-level routing momentum is not obviously desirable.** Token-level specialization may need to change rapidly inside one sequence. A switching penalty can suppress valid experts and blur specialization. The proposed “perplexity under the current expert” is not a free pre-routing signal if the expert's actual computation is required to obtain it.

**v3 disposition:** retain auxiliary-loss-free load balancing as a candidate; monitor imbalance, token overflow, all-to-all traffic and quality. Reject forced sequence-coherent routing as the default. Evaluate latent experts and variable top-k separately.

### 3.4 Section 3 — MLA and memory

**Issue:** MLA compresses attention key/value representation and the cache associated with that attention path. It does not implement a four-tier persistent memory store, episodic event consolidation, semantic entity resolution, procedural workflow retention, deletion or cross-session retrieval.

**Issue:** A KL-divergence change caused by removing a KV entry depends on the prompt, sequence state, output distribution, ablation method and point in the generation. It is not an intrinsic, context-free causal value of the KV item. Attention-weight history alone is not causal attribution.

**v3 disposition:** explicitly separate (a) recurrent neural state, (b) attention KV cache and (c) external agent memory. Compare established cache policies (H2O, StreamingLLM, SnapKV, PyramidKV) and treat the counterfactual proxy as optional, calibrated research. External four-tier memory remains a system-level design, with provenance and expiry semantics.

### 3.5 Section 4 — Reasoning

**Issue 1: strictly monotonic state machine.** The proposed fixed progression cannot revisit a failed hypothesis or update a premise after receiving new evidence. That prevents normal debugging and search behavior.

**Issue 2: “oracle” category is too broad.** A compiler, solver, unit-test suite, external API and LLM judge establish different things under different assumptions. Tests establish behavior only over tested cases; a solver establishes a property of a formalized model; an LLM judge gives a fallible estimate.

**Issue 3: self-consistency is not an oracle.** Sampled paths can share the same model bias or missing evidence. Agreement is a confidence feature or selection heuristic, not a truth guarantee.

**Issue 4: entropy-gated forced commitment is not justified.** Low next-token entropy does not prove correctness or progress. A high-entropy creative response may be appropriate; a low-entropy hallucination may be confidently wrong.

**v3 disposition:** a bounded task state graph permits revision/backtracking, tracks progress and duplicate hypotheses, enforces resource limits, and chooses a typed validator. Self-consistency and reward models stay in a weak-evidence tier. Abstention is calibrated against held-out data.

### 3.6 Section 5 — Output

**Issue 1:** MTP is a training objective and/or prediction-head design; speculative decoding is a distinct inference algorithm. They are related but not identical.

**Issue 2:** “significantly increasing generation speed” and any implied 2× claim are not guaranteed by MTP. The effect depends on acceptance length, verifier cost, batch, kernel and target workload.

**Issue 3:** A grammar defines allowed syntax, not the truth of values or the safety of an action.

**Issue 4:** A KV-retention importance predictor has no valid reason to act as a semantic token probability filter.

**v3 disposition:** separate MTP from exact speculative verification. Use grammar-constrained decoding for syntax and post-generation schema, type, domain, authorization and execution checks for semantics.

### 3.7 Section 6 — Shared uncertainty bus

**Issue:** router load violation, memory retrieval confidence, validator agreement and reasoning progress are different observables. A single learned vector does not automatically align their semantics or calibrate their scales. A shared state also creates feedback loops that are difficult to attribute and validate.

**v3 disposition:** use typed records with source, meaning, scope, calibration, update frequency and consumers. A compact learned joint-conditioning vector may be tested as an optional model component, but the runtime must preserve explicit typed evidence and never rely on an opaque vector as its only control interface.

### 3.8 Section 7 — Precision and training efficiency

**Issue:** FP8 is not a universal “precision floor.” BF16, FP8, FP4/NVFP4 and mixed precision are determined by accelerator hardware, kernels, model architecture, stability and quality requirements. DeepSeek-V3's published GPU-hour figure is not a complete commercial cost-of-development figure; it reflects reported training runs under the stated hardware accounting.

**Issue:** Chinchilla's compute-optimal result is not a universal parameter/token law for every data-quality mix, post-training regime, inference-heavy application or architecture.

**v3 disposition:** measure precision variants and scaling curves in the intended environment. Remove all unsupported aggregate training overhead estimates.

### 3.9 Section 8 — End-to-end diagram

**Issue:** the diagram depicts sequential modules for work that partly belongs inside each neural layer and partly outside the model. It routes semantic reasoning through model output and cache eviction as if these are one serial neural pipeline.

**v3 disposition:** the new diagram separates model blocks, serving runtime, external memory, validators and trusted tool execution. Arrows represent explicit interfaces rather than implied shared differentiable state.

### 3.10 Section 9 — Honest limits

**Issue:** v2 honestly notes that the combined architecture is unvalidated, but its earlier claims and costs are more confident than that disclaimer supports.

**v3 disposition:** all specific claims are evidence-qualified; the final status states clearly that the combined design has not been trained or benchmarked.

---

## 4. Citation and Source-Quality Audit

### 4.1 Original citation format is unusable

The supplied v2 includes malformed fragments such as `(cite index="17-1">... </cite>`. These do not identify the paper, technical report, publisher, publication date or stable URL. Several different claims cite the same opaque index without making it possible to check the scope of the evidence.

**Correction:** the v3 reference register uses `[R#]` citations and stable URLs. Each reference is tied to a specific architecture or claim. The v3 status tag identifies the type of evidence, while the reference supplies provenance.

### 4.2 Evidence interpretation rules

1. An official model card is useful for released model configuration; a research paper is usually stronger for method and ablation details.
2. Vendor benchmark claims remain attributed to the vendor and are not generalized to other hardware or workloads.
3. A preprint may be important and technically detailed, but should be tagged as emerging until peer review and independent replication.
4. A model's published use of a component does not validate a new combination of that component with every other component.
5. The scale of an evaluation is part of the result. A 1.3B model trained on 100B tokens cannot alone establish a 100B-scale production advantage.
6. Search snippets and model cards should not be used as the only evidence for subtle implementation or mathematical claims if the full paper is available.

### 4.3 Specific updates reflected in v3

- DeepSeek-V3: 671B total / 37B active; 14.8T pretraining tokens; 2.788M reported H800 GPU-hours for full training; MLA, DeepSeekMoE, auxiliary-loss-free balancing and MTP. Do not convert GPU-hours into a complete business/development cost without assumptions. [R6]
- Nemotron 3 Nano: 31.6B total / 3.2B active (3.6B including embeddings); 128 routable experts / 6 activated; hybrid Mamba-2 + GQA + MoE; published up-to-1M context claim. [R28]
- Qwen3.8-Flash-Next: 125B main model plus 51B N-gram embeddings, 6B active; GDN + Qwen Sparse Attention, four-branch Gated Residual and Muon-based training. This is its reported configuration, not the v3 model's fixed requirement. [R26, R27]
- GDN-2: paper reports 1.3B parameters and 100B FineWeb-Edu tokens; it compares several recurrent baselines under its evaluation. [R11]
- Muir: paper is a recent November 2026 preprint and reports evaluation up to 1.34B; v3 therefore treats it as an early high-priority replication candidate. [R12]
- Kimi K3: paper reports 2.8T total parameters, 104B active, KDA, Attention Residuals and Stable LatentMoE; these are model-specific architecture claims. [R24, R29]

---

## 5. IDEAS20 Archive — Accounting and Disposition

### 5.1 Archive accounting correction

The ZIP index states “160 ideas total across 8 batches.” The archive contains six ordinary batches `01`–`06` (120 entries), batch `08` (20 entries), a `07` Sentinel-specific mapping of 20 ideas that it explicitly says are selected “from the batches above,” and a copy of v2 in `09`. Thus the archive contains 160 numbered **entries if the 20 mapped Sentinel entries are counted again**, but it does not establish 160 distinct independent ideas. The Sentinel mapping is a second view of selected ideas, not an additional disjoint set.

### 5.2 Disposition vocabulary

- **Adopt in v3:** a strong, scoped design principle is incorporated.
- **Retain as system/runtime:** useful but not part of the neural model.
- **Research candidate:** plausible; requires an explicit experiment.
- **Narrow / reframe:** useful only with stricter assumptions.
- **Reject as stated:** invalid generalization, unsupported mechanism or incorrect claimed guarantee.
- **Out of LLM scope:** potentially valuable computer-science project, but not an LLM architecture improvement.

### 5.3 Batch 01 — Uncharted computer-science product ideas

| # | Idea | Disposition | Review |
|---:|---|---|---|
| 1 | Compiler Error Semantic Diffing | Retain as system/runtime | Useful for code-agent diagnosis and test feedback; it is developer tooling, not a neural-layer design. |
| 2 | Intent-Preserving Code Refactor Ledger | Retain as system/runtime | Durable decision provenance can improve agent memory and reproducibility. |
| 3 | Deterministic Fuzzing via Execution Fingerprinting | Out of LLM scope | Fuzzing is valuable for testing generated code, but an execution fingerprint does not automatically generate high-coverage inputs. |
| 4 | Silent Dependency Drift Detector | Retain as system/runtime | Replay-based dependency regression testing can harden tool execution. |
| 5 | Grammar-Native Config Language | Retain as system/runtime | Useful for model/runtime configuration and safe schema parsing. |
| 6 | Execution Budget Language Primitive | Retain as system/runtime | The budget concept belongs in a trusted runtime or language effect system; a language syntax alone cannot enforce it. |
| 7 | Causality-Aware Logging | Retain as system/runtime | Record observed causal/data dependencies; do not claim neural-token causality from spans. |
| 8 | Type-System-Enforced API Versioning | Retain as system/runtime | Useful for typed tool APIs and version-safe integrations. |
| 9 | Semantic Diff for Databases | Retain as system/runtime | Can validate agent-generated migrations against explicit business invariants. |
| 10 | Offline-First Collaborative CRDT Notebook | Out of LLM scope | Useful collaborative application; CRDT convergence does not establish semantic correctness of outputs. |
| 11 | Proof-Carrying Build Artifacts | Retain as system/runtime | Relevant to verification of code generated by agents; proofs are only as strong as their formal assumptions. |
| 12 | Reactive Type Inference | Out of LLM scope | Compiler/editor feature, not a language-model architecture. |
| 13 | Execution-Time Ontology | Retain as system/runtime | Symbolic rules can handle deterministic subproblems and reduce unnecessary model calls. |
| 14 | Morphic Version Control | Out of LLM scope | Potential VCS research; not an LLM component. |
| 15 | Process Algebra-Based Microservice Orchestrator | Retain as system/runtime | Formal orchestration can constrain multi-agent and tool workflows; state-space explosion remains a practical concern. |
| 16 | Hardware-Topology-Aware Task Scheduler | Retain as serving research | Hardware-aware scheduling matters, but a userspace scheduler cannot safely replace the OS scheduler in general. |
| 17 | Symbolic Execution as a Test Generator | Retain as system/runtime | Valuable for generating executable tests for code agents; path explosion and unsupported language features limit coverage. |
| 18 | Immutable Append-Only Config Store | Retain as system/runtime | Helps audit model configs, policy changes and experiments; requires explicit secret retention/redaction. |
| 19 | Polyglot AST Bridge | Retain as system/runtime | Useful for parsing, validation and cross-language code-agent workflows. |
| 20 | Probabilistic Invariant Mining | Narrow / reframe | Invariants mined from traces are hypotheses with confidence and coverage limits, not proven program properties. |

### 5.4 Batch 02 — Foundational computing ideas

| # | Idea | Disposition | Review |
|---:|---|---|---|
| 1 | Intent-Typed Memory Allocator | Retain as serving research | Could inform typed memory pools for tensors/KV/state, but semantic “intent” does not by itself guarantee safe allocation. |
| 2 | Temporal Memory Model | Out of LLM scope | Full hardware-register-level time travel is not an ordinary MMU feature; storage-backed snapshots and logging are more realistic. |
| 3 | Ownership-Native Heap | Out of LLM scope | Relevant to runtime safety, but allocator-only enforcement of language ownership is not generally achievable without interface constraints. |
| 4 | Lazy CPU Instruction Scheduler | Reject as stated | Arbitrary machine instructions cannot simply be made lazy without preserving side effects, exceptions, memory ordering and control flow. |
| 5 | Causal Execution Threads | Narrow / reframe | Carry parent/span metadata for observability; causal depth alone is not a safe scheduling priority. |
| 6 | Reversible Computation Runtime | Research only / out of LLM scope | General programs are not reversible by default; inverse-operation logs are not sufficient when information is destroyed or side effects escape. |
| 7 | Semantic Filesystem | Retain as system/runtime | Could support document and corpus management, but semantic search still needs schemas, indexing and path interoperability. |
| 8 | Causality-Linked File Provenance | Retain as system/runtime | Useful lineage for retrieved evidence and generated artifacts; trace completeness must be stated. |
| 9 | Demand-Paged Virtual Filesystem | Retain as system/runtime | Can support large datasets and lazy document access; not a learned memory mechanism. |
| 10 | Type-Safe Packet Schemas | Retain as system/runtime | Useful for typed model-serving and agent service interfaces. |
| 11 | Causal Network Ordering Primitive | Retain as system/runtime | Vector clocks and causal metadata help distributed workflows; they do not provide synchronized physical time. |
| 12 | Session-Typed Socket Layer | Retain as system/runtime | Good basis for protocol-safe tool/service interactions. |
| 13 | Capability-Native Process Model | Adopt as security principle | Enforce permissions in trusted host/runtime boundaries; capability semantics need an unforgeable implementation and audit. |
| 14 | Algebraic Process Scheduler | Narrow / reframe | Reorder only operations proven independent/commutative; otherwise preserve dependency order. |
| 15 | Deterministic Syscall Replay | Retain as system/runtime | Useful for sandboxed tool replay when nondeterministic inputs and external IO are recorded or virtualized. |
| 16 | Separation Logic Native IR | Out of LLM scope | Compiler research can strengthen generated-code assurance; not a neural architecture feature. |
| 17 | Effect-Typed Linker | Retain as system/runtime | Effect metadata could constrain plugin and tool integration; depends on language/linker design. |
| 18 | Structural Binary Diffing | Retain as system/runtime | Useful for auditing generated binaries and upgrades; decompiler IR is approximate and platform-dependent. |
| 19 | Constraint-Native Data Model | Adopt as memory-system principle | Store facts with schema and constraints; preserve provenance and allow retraction/supersession where needed. |
| 20 | Monotonic Computation Primitive | Narrow / reframe | Semilattice accumulation supports convergent state; real-world knowledge requires corrections, tombstones and non-monotonic truth maintenance. |

### 5.5 Batch 03 — AI/LLM and agent software ideas

| # | Idea | Disposition | Review |
|---:|---|---|---|
| 1 | Prompt Algebra | Research candidate | Formal equivalence laws may work for restricted prompt fragments; natural-language composition is not generally commutative or semantically invariant. |
| 2 | Effect-Typed LLM Pipelines | Adopt at application layer | Model calls should declare IO, state, tool side effects and trust boundaries. |
| 3 | Composable Prompt Type System | Adopt at application layer | Typed prompt inputs/outputs reduce schema errors; types do not guarantee model semantic compliance. |
| 4 | Context-Sensitive Grammar-Constrained Generation | Narrow / reframe | Support practical incremental grammars/automata; arbitrary context-sensitive grammars can be too expensive or undecidable for practical decoding. |
| 5 | Causal Output Attribution | Research candidate | Requires interventions and a defined causal estimand; attention weights and post-hoc correlations are not causal proof. |
| 6 | Session-Typed Agent Protocols | Adopt at application layer | Typed protocols can prevent invalid tool-state transitions before runtime. |
| 7 | Process Algebra Agent Specification | Retain as system/runtime | Model-check workflows with a bounded state space; formal verification must include external tool assumptions. |
| 8 | Capability-Native Agent Permissions | Adopt as security principle | Check capabilities at the trusted execution boundary; never delegate authorization to prompt text. |
| 9 | Recursive Symbolic Self-Modeling Agent | Narrow / reframe | Keep an explicit capability/tool registry with evidence and freshness; do not let inferred success rates become permanent facts. |
| 10 | Monotonic Agent Memory | Narrow / reframe | CRDT convergence is useful for distributed merges, but semantic facts must support correction, expiry, authority and tombstones. |
| 11 | Deterministic Agent Replay | Adopt with caveat | Deterministic replay applies to recorded IO and environment state, not arbitrary fresh live provider/service calls. |
| 12 | Agentic Execution Diff | Adopt at application layer | Compare event DAGs to locate first divergent decision and downstream consequences. |
| 13 | Agent Execution Budget Primitive | Adopt at application layer | Enforce token, tool, wall-time, network and memory ceilings in trusted runtime code. |
| 14 | Intent-Typed Agent Checkpointing | Adopt at application layer | Persist goal, plan, side effects, idempotency keys and unresolved work. |
| 15 | Semantic Agent Versioning | Narrow / reframe | Behavioral equivalence is generally undecidable; use versioned evaluation profiles and empirical regression signatures. |
| 16 | Multi-LLM Semantic Consensus | Narrow / reframe | Can rank or surface disagreement, but errors are correlated and additional calls increase cost; never treat it as proof. |
| 17 | Adversarial Context Minimization | Research candidate | Bisection assumes a useful monotonicity that may not hold; preserve semantics via task-specific evaluation and counterexamples. |
| 18 | Temporal Causal Context Compression | Narrow / reframe | Use time, recency, task relevance and explicit dependency metadata; causal claims require stronger evidence. |
| 19 | Zero-Knowledge Agent Audit | Research only | ZK can prove a precisely encoded computation/property; it does not cheaply prove arbitrary neural reasoning or truth of external inputs. |
| 20 | LLM-Native Formal Spec Extractor | Adopt as assistant tool | Generate candidate TLA+/Alloy/Lean specs, then independently type-check/prove/test and review assumptions. |

### 5.6 Batch 04 — LLM internals, architecture and pipelines

| # | Idea | Disposition | Review |
|---:|---|---|---|
| 1 | Attention Head Behavioral Taxonomy Engine | Research candidate | Useful interpretability tool; probe labels are hypotheses and can be context/model-dependent. |
| 2 | Attention Pattern Cache | Reject as stated | Similar inputs do not imply reusable attention matrices; exact prefix/KV caching is a different, well-defined optimization. |
| 3 | Attention Sink Redistribution Layer | Research candidate | Sink-aware handling is established in some cache methods; post-hoc redistribution may alter learned behavior and needs retraining/ablation. |
| 4 | Cross-Layer Attention Correlation Miner | Research candidate | Correlated attention maps do not prove layer equivalence or safe layer skipping. |
| 5 | Residual Stream Semantic Diff | Retain as interpretability tooling | Can diagnose representation divergence but needs defined probes and validated interpretation. |
| 6 | Logit Trajectory Profiler | Retain as interpretability tooling | A logit lens is diagnostic, not a faithful causal account of when a model “decides.” |
| 7 | MLP Neuron Behavioral Atlas | Research candidate | Activation/suppression studies can characterize units, but roles are distributed and context-sensitive. |
| 8 | Context Causal Graph Builder | Research candidate | Use controlled interventions; causal graph completeness and token-level attribution are not automatic. |
| 9 | Circuit Extraction Automator | Research candidate | Valuable interpretability goal; requires robust intervention and equivalence evaluation. |
| 10 | Speculative Layer Skipping | Narrow / reframe | Retain only with training/verification such as LayerSkip; arbitrary inference-time skipping is unsafe. |
| 11 | Semantic Speculative Decoding Verifier | Reject as stated | Semantic equivalence is not the probability correction required to preserve the target sampling distribution. |
| 12 | Inference-Time Layer Fusion | Reject as stated | Nonlinear layer compositions are not generally algebraically equivalent; kernel fusion is not semantic layer merging. |
| 13 | KV Cache Causal Eviction | Narrow / reframe | Keep counterfactual KL as a sampled diagnostic; baseline against H2O, StreamingLLM, SnapKV and PyramidKV. |
| 14 | Context Topology Reorderer | Reject as default | Reordering can alter position, discourse, code, chronology and negation semantics. Only reorder fields explicitly proven order-invariant. |
| 15 | Semantic Context Fixpoint Compressor | Research candidate | Output equivalence is hard to establish and model-dependent; use task-specific compression tests and preserve originals. |
| 16 | Entropy-Adaptive Temperature Scheduler | Narrow / reframe | Temperature changes output distribution; high entropy is not automatically a signal to increase randomness. |
| 17 | Confidence Component Decomposer | Research candidate | Define separate calibrated events (knowledge, execution, grounding); hidden-state probes are not trustworthy confidence by default. |
| 18 | Prompt Sensitivity Heatmap Generator | Retain as evaluation tooling | Ablation measures sensitivity to a perturbation, not causality or semantic importance alone. |
| 19 | Tokenization Semantic Boundary Auditor | Retain as evaluation tooling | Measure fertility, fragmentation and task impact; mutual information alone is not a tokenizer quality criterion. |
| 20 | Gradient Convergence-Driven Layer Freezer | Research candidate | Low gradient variance does not prove a layer is no longer needed; compare freeze/unfreeze baselines. |

### 5.7 Batch 05 — LLM efficiency, compression and architecture improvement

| # | Idea | Disposition | Review |
|---:|---|---|---|
| 1 | Behavioral Equivalence Pruning | Research candidate | Behavioral tests and activation patching can guide pruning, but finite samples do not establish equivalence over the input distribution. |
| 2 | Cross-Layer Delta Compression | Research candidate | Layer sharing plus deltas is plausible; compare against standard weight sharing, low-rank adapters and no sharing. |
| 3 | Semantic Axis Decomposition | Reject as stated / research | “Semantic axes” need a reproducible definition; SVD and learned low-rank baselines must be beaten under matched quality. |
| 4 | Attention Head Functional Merger | Research candidate | CKA similarity does not guarantee functionally interchangeable heads; intervention and retraining tests required. |
| 5 | Tensor Ring Decomposition for Embeddings | Research candidate | A known tensor-compression family; memory and quality must be measured against SVD/quantization baselines. |
| 6 | Semantic Complexity-Adaptive Expert Budgeting | Research candidate | Need a low-cost complexity predictor that correlates with marginal expert utility; dynamic dispatch may erase savings. |
| 7 | Expert Specialization Enforcement Loss | Research candidate | Maximizing expert divergence can create arbitrary specialization and harm shared features; measure useful specialization. |
| 8 | Expert Router as Pushdown Automaton | Reject as default | A formal automaton needs hand-coded state/rules and is not a drop-in learned router for open-domain semantics. |
| 9 | Dead Expert Detection & Recycling | Research candidate | Expert reinitialization can destabilize training and invalidate optimizer/momentum state; test controlled recycling and rollback. |
| 10 | Speculative Layer Skipping via Per-Layer Confidence | Narrow / reframe | Use LayerSkip-style training and verification; a “2-parameter gate” is not enough evidence of safe early exit. |
| 11 | Activation Sparsity Induction | Research candidate | Sparsity penalties may reduce quality; inference gains require supported sparse kernels and measured activation sparsity. |
| 12 | Semantic Tiling for Flash Attention | Reject as stated | FlashAttention tiles for hardware/IO locality; semantic grouping is not a free exact-kernel optimization and may change sparse coverage. |
| 13 | Continuous Batching with Semantic Completion Proximity | Reject as default | Semantic “closeness to completion” is difficult to estimate and can impair fairness; schedule by measurable shape, queue and SLO. |
| 14 | Gradient Flow Topology Pruning | Research candidate | Historical gradient paths do not prove future irrelevance; require regrowth, ablation and task-level validation. |
| 15 | Layer Importance-Scheduled Gradient Updates | Research candidate | Skipping updates can destabilize optimization or leave stale layers; compare with normal optimizer schedules. |
| 16 | Second-Order Gradient Compression via Kronecker Factoring | Narrow / reframe | K-FAC approximates curvature; it is not automatically a communication-compression mechanism or a cheaper optimizer at LLM scale. |
| 17 | Causal KV Cache Eviction | Narrow / reframe | Same proposal as Batch 04 #13; retain only as an experimentally calibrated cache-policy candidate. |
| 18 | Semantic KV Clustering with Centroid Retrieval | Reject for model KV; retain for external vectors | Vector-centroid search is suitable for embedding retrieval; approximating KV tensors by semantic centroids has no general bounded output-error guarantee. |
| 19 | Hierarchical Context Encoding at Multi-Resolution | Research candidate | Useful for external retrieval/reranking; simultaneous token/phrase/paragraph model pathways add complexity and need ablation. |
| 20 | Entropy-Stratified Token Budget Allocator | Narrow / reframe | Allocate compute using calibrated expected value of additional processing, not entropy alone. |

### 5.8 Batch 06 — LLM inference speed ideas

| # | Idea | Disposition | Review |
|---:|---|---|---|
| 1 | Attention Pattern Memoization Cache | Reject as stated | Attention matrices are input-activation-specific; only exact matching prefixes or proven deterministic intermediate reuse can be safely cached. |
| 2 | Per-Head Adaptive Dropping via Proxy Metric | Research candidate | Residual norm alone is not a proof of negligible functional contribution; needs trained/calibrated gate and quality evaluation. |
| 3 | LSH-Based Inference-Time Attention Sparsification | Research candidate | Approximate locality-sensitive masks may help; “any trained model” is too broad and missed-key recall matters. |
| 4 | Semantic Tiling with Mutual Information Scoring | Research candidate | Could guide sparse block selection, but mutual-information estimation overhead and approximation error require benchmarking. |
| 5 | Logit-Compressed Speculative Verification | Reject as stated | Top-k truncation changes or loses probability mass unless a mathematically correct residual/tail treatment is specified. |
| 6 | Asynchronous Dual-Phase Speculative Decoding | Research candidate | Pipelining draft and verification is constrained by data dependencies; test actual device overlap and accepted-token latency. |
| 7 | Speculative Ensemble Micro-Drafting | Research candidate | Parallel draft models cost extra compute; consensus does not remove need for target verification. |
| 8 | Decode Path Prefix-Tree Prewarming | Narrow / reframe | Prewarm only known exact shared prefixes; semantic/action-path similarity does not imply token-prefix KV reuse. |
| 9 | KV Delta Compression Across Layers | Research candidate | Cross-layer KV tensors are not generally near-identical; test quantization/low-rank residuals with reconstruction and task metrics. |
| 10 | Zero-Copy KV Offload with Predictive Prefetching | Research candidate | Useful only where CPU-GPU transfer overlaps compute and predictor accuracy is high enough; hardware-dependent. |
| 11 | Per-Token Dynamic Precision KV Allocation | Research candidate | Mixed precision is plausible; causal importance scores must be calibrated and quantization error tested per head/layer. |
| 12 | Batch Semantic Regrouping for Kernel Locality | Reject as default | Semantic similarity does not necessarily improve kernel locality; batching by sequence shape and SLO is measurable and fairer. |
| 13 | Prefill-Decode Phase Splitter with Predictive Block Pre-allocation | Narrow / reframe | Predicting output length can inform resource reservations but has error; preallocation policy needs over/under-allocation tests. |
| 14 | Decode Compute Recycling for Stable Token Regions | Narrow / reframe | Reuse exact shared prefix computations where inputs and cache state match; similar residual patterns are not enough. |
| 15 | Residual Stream Confidence Early Exit Gate | Narrow / reframe | Use trained early-exit/successor-verification mechanisms; confidence is not a reliable correctness certificate. |
| 16 | Learned Sparse Attention Mask Pre-computation | Research candidate | Must measure recall and quality under false negatives; architecture training may be needed for robust masks. |
| 17 | Inference-Time Neuron Cluster Gating | Research candidate | Potential sparse-kernel optimization; co-activation clusters and gate overhead must outperform dense kernels. |
| 18 | Token Merging at Language Inference | Reject as generic default | Language token identity, order and syntax matter; merging tokens can change meaning and positions. |
| 19 | Grammar-Constrained Pre-Softmax Token Elimination | Reject as stated | Grammar validity is often known before sampling, but exact token legality is applied to logits; avoiding the vocabulary projection requires a specialized sparse output head/index and proof of equivalence. |
| 20 | Confidence-Gated Multi-Step Greedy Commit | Reject as speed claim | Argmax versus sampling is usually small relative to the model forward pass and changes the target sampling distribution. |

### 5.9 Batch 08 — Reasoning-cycle reduction ideas

| # | Idea | Disposition | Review |
|---:|---|---|---|
| 1 | Upfront Constraint Propagation Prefix | Adopt with qualification | Extract formal constraints before generation where parsers/solvers can do so; tag inferred constraints and uncertainty separately. |
| 2 | Contradiction Map Pre-computation Pass | Adopt with qualification | Detect typed conflicts and source disagreement; do not convert heuristic semantic disagreement into a hard contradiction. |
| 3 | PRM with Explicit Backtrack Signal | Research candidate | Train on verified cycle/failure labels; guard against reward hacking and domain shift. |
| 4 | Speculative Reasoning with Parallel Execution Oracle | Adopt for verifiable tasks | Generate candidates in parallel where cost is acceptable; each needs independent execution/solver validation and failure isolation. |
| 5 | Hypothesis Lattice with Formal Dominance Pruning | Research candidate | Dominance pruning is safe only when the partial order and objective are formally valid. |
| 6 | Monotonic Reasoning State Machine | Reject as strict no-reentry | Replace with bounded state graph and explicit backtracking. |
| 7 | Causal Dependency Graph as Reasoning Substrate | Narrow / reframe | Use explicit code/data dependency graphs where supported; do not infer real-world causality from text association alone. |
| 8 | Entropy-Gated Reasoning Termination | Reject as stated | Entropy is not answer correctness or progress. |
| 9 | Formal UNSAT-Detecting Reasoning Scratchpad | Retain for formal domains | Solver UNSAT is meaningful only for a complete, correct formalization; natural-language premises may be missing or mistranslated. |
| 10 | Semantic Beam Search with Cross-Beam Deduplication | Narrow / reframe | Use canonical structured states where possible; approximate semantic dedup can merge distinct hypotheses. |
| 11 | Global Context Register Architecture | Research candidate | Special global tokens can help preserve constraints, but cannot make context forgetting structurally impossible. |
| 12 | Contrastive Reasoning Step Loss | Research candidate | Penalizing semantic similarity can reward novelty/verbosity rather than progress; supervise validated state transitions instead. |
| 13 | Multi-Resolution Context Compression Prefix | Research candidate | Keep raw source and provenance; summaries at multiple levels need retrieval/attention ablations and may duplicate content. |
| 14 | E-Graph Equality Saturation for Code Fix Reasoning | Adopt for code tools | Valuable for transformations over a supported equivalence theory; not general natural-language reasoning. |
| 15 | Causal Intervention Testing During Thought | Narrow / reframe | Use interventions only where the environment/model and causal variable are well-defined and safe to manipulate. |
| 16 | Structured Output Schema Pre-commitment | Adopt at output interface | Establish the required output contract early; it does not guarantee semantic correctness. |
| 17 | Grounded World Model with State Consistency Checking | Adopt at system layer | Store typed facts/contracts with provenance and queryable state; track stale and missing facts. |
| 18 | Layered Abstraction Attention with Cross-Layer Grounding | Research candidate | An immutable context path is plausible but adds architecture cost; test versus retrieval and recurrent memory. |
| 19 | Adaptive Reasoning Depth Router | Adopt with calibration | Select task budgets using measured expected benefit and risk, not uncalibrated complexity labels. |
| 20 | Symbolic Execution Trace Integration | Adopt for code subset | Execution traces can ground bug diagnosis; symbolic path explosion and supported-language limits must be explicit. |

### 5.10 Batch 07 — Sentinel-specific mapping audit

The Sentinel file is a mapping of 20 selected ideas to a 13-crate agent platform, not an independent source of research validation. Its main architectural mappings require these corrections:

| # | Sentinel mapping | v3 review |
|---:|---|---|
| 1 | “Causal KV Cache Eviction” applied to Episodic/Semantic memory eviction | **Reject the direct equivalence.** Model KV entries and application memory records have different data structures and semantics. External-memory eviction can use source freshness, retrieval utility, policy and retention rules; counterfactual model KL is not a direct substitute. |
| 2 | Monotonic CRDT Agent Memory | **Retain with semantic correction.** CRDT convergence is not truth, freshness or authority. Add correction, expiry, tombstones and provenance. |
| 3 | Temporal Causal Context Compression | **Narrow.** Use task relevance, chronology and explicit dependency metadata. “Causal importance” needs intervention or domain-grounded causal structure. |
| 4 | Intent-Typed Agent Checkpointing | **Adopt at application layer.** Checkpoints should include semantic goal, plan, completed actions, side effects, idempotency and recovery state. |
| 5 | Adversarial Context Minimization | **Research.** Preserve the raw source and require task-specific equivalence tests; bisection can fail when interactions are non-monotonic. |
| 6 | Context Topology Reordering | **Reject as general pre-dispatch transform.** It may alter order, temporal semantics, code layout, negation or references. Reorder only schema fields explicitly declared order-independent. |
| 7 | Hierarchical Context Encoding | **Research for external retrieval.** Token/phrase/paragraph multi-resolution indexes are plausible, but do not imply neural attention over every scale. |
| 8 | Semantic KV Clustering | **Restrict to external embedding/index retrieval.** Do not use semantic centroids as an assumed bounded-error approximation of neural KV tensors. |
| 9 | Multi-LLM Semantic Consensus | **Retain as a weak routing signal only.** Model errors are correlated; disagreement can trigger escalation but agreement does not prove truth. |
| 10 | Entropy-Adaptive Temperature | **Task-specific experiment.** Temperature modifies sampling entropy; high uncertainty does not automatically imply more randomness. |
| 11 | Confidence-Gated Greedy Commit | **Reject the claimed performance guarantee.** Greedy decoding changes sampling behavior and sampler cost is generally small relative to model compute. |
| 12 | Decode Path Prefix-Tree Prewarming | **Narrow to exact token-prefix caching.** Procedural action sequences are not necessarily token-prefixes of future model outputs. |
| 13 | Session-Typed Agent Protocols | **Adopt at application layer.** Typestate/session protocols can enforce legal tool/agent message transitions. |
| 14 | Capability-Native Agent Permissions | **Adopt security principle, correct implementation claim.** WASM linear-memory isolation alone does not make a host capability unforgeable; enforce authorization in trusted host code. |
| 15 | Intent-Typed Agent Checkpointing | **Adopt.** Include tool idempotency, durable external side effect status and recovery policy. |
| 16 | Deterministic Agent Replay | **Adopt with scope.** Replay recorded IO and state transitions; fresh external calls are not automatically deterministic. |
| 17 | Agentic Execution Diff | **Adopt as debugging.** DAG divergence can identify the first observed execution difference, not prove neural token causality. |
| 18 | Agent Execution Budget Primitive | **Adopt.** Enforce token/tool/time/network/memory budgets in trusted runtime code. |
| 19 | Recursive Symbolic Self-Model | **Narrow.** Keep a sourced capability registry; do not let performance heuristics become permanent facts or replace novel reasoning. |
| 20 | Causal Output Attribution via OTel | **Split into two claims.** OTel can trace application-level event/data lineage; causal influence of input tokens on output tokens requires model-level intervention studies. |

---

## 6. Architecture Decisions Made in v3

| Decision | Rationale | Validation required |
|---|---|---|
| Separate neural, serving and agent/application strata | Avoid conflating mechanisms with different state, guarantees and cost centers. | Interface and component-level tests. |
| Hybrid recurrent + attention backbone remains the central hypothesis | Publicly documented hybrid models show this is a productive design space. | Matched scale/schedule/operator ablations. |
| Do not hard-code 1:7 or 3:1 | Published schedules are model-specific. | Search schedule under equal budgets. |
| GDN-2, Mamba-2/3 and Muir are candidate recurrent operators | Diverse current designs provide testable sequence-mixing alternatives. | Replication and hardware profiling. |
| GQA is the initial attention-cache baseline; MLA/QSA are alternatives | Simpler baseline helps isolate cache and indexer trade-offs. | Compare latency, recall and cache footprint. |
| Standard residual is initial baseline; AttnRes and GR are separate experiments | Avoid confounded stacking of depth-aggregation mechanisms. | Independent and interaction ablations. |
| MoE is introduced after sequence backbone stability | Isolate routing and all-to-all failure modes. | Dense/MoE/latent-MoE matched tests. |
| Recurrent state, attention KV and external memory are separate | Different persistence, indexing and correctness semantics. | State isolation, reset, retrieval and privacy tests. |
| Bounded backtracking replaces strictly monotonic reasoning | Real problem-solving revises failed hypotheses. | Loop tests, progress metrics and budget enforcement. |
| Typed evidence and typed control signals replace one uncertainty vector | Preserve semantics, calibration and provenance. | Schema validation and calibration tests. |
| Grammar plus domain validation replaces “semantic filtering” by KV proxy | Syntax and semantic correctness are different. | Parser, schema, domain and execution tests. |
| Experiment-first promotion policy | No combined architecture result exists yet. | Reproducible published ablation record. |

---

## 7. Remaining Work Versus Work Completed

### 7.1 Completed in this document task

- Reviewed the entire baseline v2 document.
- Inventoried all files in the supplied ZIP and read the accompanying idea lists/mappings.
- Identified factual, citation, systems-boundary, algorithmic and evaluation errors.
- Surveyed major public sequence-model families and representative Transformer, SSM, recurrent, hybrid, MoE, memory, adaptive-compute, diffusion and multimodal work.
- Rewrote the architecture as v3 with explicit evidence statuses, references and a testable experimental program.
- Classified the 140 numbered ideas in batches 01–06 and 08, reviewed the 20 Sentinel mappings, and identified duplicate counting in the archive index.

### 7.2 Not performed (and why)

- **No model weights were trained.** The task supplied no compute allocation, target model scale, dataset/license plan or hardware environment. A written specification cannot honestly establish training gains.
- **No numerical performance comparison was run.** Architecture claims need controlled implementations and matched benchmarks; synthetic FLOP estimates would not validate actual speed.
- **No public or proprietary catalogue can be complete.** v3 scopes the survey to major public and documented families through the review date. Proprietary internals and unpublished research remain unknown.
- **No Sentinel or other external repository was modified.** The supplied ZIP provided conceptual mappings but did not authorize repository changes.

### 7.3 The remaining work is experimental, not another design recommendation

The specification's next step is unambiguous: select model scale, hardware and data budget; freeze the experiment contract; implement operator-level pilots; run the architecture and runtime ablations; and promote only measured improvements. More speculative components should not be added until the base experiment produces evidence.

---

## 8. Deliverable Integrity

The v3 `.md` and `.txt` files are intended to be byte-for-byte identical Markdown source documents. The audit is a separate file so that the architecture specification remains usable as a standalone technical design while the audit preserves rejected ideas and rationale.
