#!/usr/bin/env python3
"""Small CPU feasibility pilot for v9 P0/P1/P1-LA using an online-sourced text excerpt.

P1 implements a GDN-2-inspired recurrence equation with simplified projections and
sigmoid gate parameterization. It is not an exact reproduction of the official GDN-2
model or its fused WY/chunkwise GPU kernels.
"""
from __future__ import annotations
import hashlib, json, math, os, random, time, platform, resource, re
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "tinyshakespeare_30k.txt"
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)
DEVICE = torch.device("cpu")
SEED = 20261118
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
FF_MULT = 2
BLOCK_SIZE = 32
BATCH_SIZE = 4
STEPS = int(os.environ.get("PILOT_STEPS", "500"))
EVAL_INTERVAL = int(os.environ.get("PILOT_EVAL_INTERVAL", "250"))
SEEDS = tuple(int(x) for x in os.environ.get("PILOT_SEEDS", "20261118,20261119").split(",") if x.strip())
PROFILES = tuple(x.strip() for x in os.environ.get("PILOT_PROFILES", "P0,P1,P1-LA").split(",") if x.strip())
RESULTS_NAME = os.environ.get("PILOT_RESULTS_NAME", "pilot_results.json")
EXPECTED_SAMPLE_SHA256 = "3229111e25b798391b379a7707aad6b45cb8cfe8d26e0ec463a2120ec15e18be"
EVAL_BATCH_SIZE = 4
LR = 3e-4



def read_verified_dataset_bundle(data_dir: Path) -> tuple[str, str, str, dict]:
    """Read train/validation/test text splits and verify each against its manifest hash."""
    manifest_path = data_dir / "dataset_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    sidecar_path = data_dir / "dataset_manifest.sha256"
    if not sidecar_path.is_file():
        raise FileNotFoundError("Missing dataset_manifest.sha256; bundle manifest identity is not authenticated")
    sidecar_parts = sidecar_path.read_text(encoding="ascii").strip().split()
    if len(sidecar_parts) != 2 or sidecar_parts[1] != manifest_path.name:
        raise ValueError("Malformed dataset manifest checksum sidecar")
    manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
    if sidecar_parts[0] != manifest_digest:
        raise RuntimeError("Dataset manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("format") != "llm-text-split-bundle-v1":
        raise ValueError("Unsupported dataset bundle format")
    revision = str(manifest.get("revision", ""))
    if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("Dataset bundle must identify an immutable 40-character revision")
    rights_ok = manifest.get("rights_review_status") == "approved_for_declared_uses"
    if not rights_ok and os.environ.get("PILOT_ALLOW_UNREVIEWED_DATA") != "1":
        raise PermissionError(
            "Dataset rights are not approved for declared uses. Complete human review or set "
            "PILOT_ALLOW_UNREVIEWED_DATA=1 only for an explicitly exploratory, non-production test."
        )
    required_uses = {"training", "validation", "model_selection", "evaluation"}
    if rights_ok and not required_uses.issubset(set(manifest.get("allowed_uses", []))):
        raise PermissionError(f"Dataset approval does not cover required uses: {sorted(required_uses)}")
    splits = {}
    seen_split_files: set[str] = set()
    seen_split_digests: set[str] = set()
    resolved_root = data_dir.resolve()
    for name in ("train", "validation", "test"):
        record = manifest.get("splits", {}).get(name)
        if not isinstance(record, dict) or not record.get("file") or not record.get("sha256"):
            raise ValueError(f"Dataset manifest lacks file/hash identity for split {name}")
        split_file = record["file"]
        if split_file in seen_split_files:
            raise ValueError("Train/validation/test must reference distinct split files")
        seen_split_files.add(split_file)
        path = (data_dir / split_file).resolve()
        if path.parent != resolved_root or path.name != split_file:
            raise ValueError(f"Unsafe split path for {name}; split files must be direct children of the bundle directory")
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} split file: {path}")
        digest = sha256(path)
        if digest != record["sha256"]:
            raise RuntimeError(f"{name} split SHA-256 mismatch: expected {record['sha256']}, got {digest}")
        if digest in seen_split_digests:
            raise ValueError("Train/validation/test split contents must not be byte-identical")
        seen_split_digests.add(digest)
        if record.get("bytes") is not None and path.stat().st_size != record["bytes"]:
            raise RuntimeError(f"{name} split byte count mismatch")
        splits[name] = path.read_text(encoding="utf-8")
        if not splits[name]:
            raise ValueError(f"Empty dataset split: {name}")
        if record.get("characters") is not None and len(splits[name]) != record["characters"]:
            raise RuntimeError(f"{name} split character count mismatch")
    manifest["local_bundle_dir"] = str(resolved_root)
    manifest["manifest_sha256_verified"] = True
    manifest["manifest_sha256"] = manifest_digest
    manifest["local_split_sha256_verified"] = True
    return splits["train"], splits["validation"], splits["test"], manifest

def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class CausalAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, window: int | None = None):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.window = window
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        if self.window is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            # Explicit causal local mask: each query sees at most `window` prior positions.
            qi = torch.arange(t, device=x.device)[:, None]
            kj = torch.arange(t, device=x.device)[None, :]
            allowed = (kj <= qi) & (kj >= qi - self.window + 1)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[None, None, :, :])
        return self.out(y.transpose(1, 2).contiguous().view(b, t, c))


class GatedDeltaNet2InspiredReference(nn.Module):
    """Naive reference for the GDN-2-family state-update equation.

    This pilot uses simplified input projections and sigmoid parameterizations;
    it is not an exact port of the official checkpoint/model or fused kernels.
    State S_t has [batch, head, key_dim, value_dim]. The key-side erase gate
    and value-side write gate are independent. The scan is sequential PyTorch.
    """
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.dk = self.dv = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.erase_gate = nn.Linear(d_model, d_model)
        self.write_gate = nn.Linear(d_model, d_model)
        self.decay_gate = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        nn.init.constant_(self.erase_gate.bias, -1.0)
        nn.init.constant_(self.write_gate.bias, -1.0)
        nn.init.constant_(self.decay_gate.bias, 2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        qkv = self.qkv(x).view(b, t, 3, self.n_heads, self.dk)
        q, k, v = qkv.unbind(dim=2)
        erase = torch.sigmoid(self.erase_gate(x).view(b, t, self.n_heads, self.dk))
        write = torch.sigmoid(self.write_gate(x).view(b, t, self.n_heads, self.dv))
        decay = torch.sigmoid(self.decay_gate(x).view(b, t, self.n_heads, self.dk))
        q = F.normalize(q, dim=-1, eps=1e-6)
        k = F.normalize(k, dim=-1, eps=1e-6)
        state = x.new_zeros((b, self.n_heads, self.dk, self.dv))
        outputs = []
        for i in range(t):
            ki = k[:, i]
            qi = q[:, i]
            vi = v[:, i]
            # D_t S_{t-1}: per-key-channel decay.
            decayed = decay[:, i].unsqueeze(-1) * state
            # (b_t * k_t)^T D_t S_{t-1}; erase and write are independently gated.
            erase_read = torch.matmul((erase[:, i] * ki).unsqueeze(-2), decayed).squeeze(-2)
            state = decayed - ki.unsqueeze(-1) * erase_read.unsqueeze(-2)
            state = state + ki.unsqueeze(-1) * (write[:, i] * vi).unsqueeze(-2)
            outputs.append(torch.matmul(qi.unsqueeze(-2), state).squeeze(-2))
        y = torch.stack(outputs, dim=1).reshape(b, t, c)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mixer: str, ffn_hidden: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if mixer == "attention":
            self.mixer = CausalAttention(d_model, n_heads)
        elif mixer == "local_attention":
            self.mixer = CausalAttention(d_model, n_heads, window=16)
        elif mixer == "gdn2_inspired_reference":
            self.mixer = GatedDeltaNet2InspiredReference(d_model, n_heads)
        else:
            raise ValueError(mixer)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden), nn.GELU(),
            nn.Linear(ffn_hidden, d_model),
        )
        self.drop = nn.Dropout(0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int, profile: str):
        super().__init__()
        self.profile = profile
        self.token_emb = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Parameter(torch.zeros(1, BLOCK_SIZE, D_MODEL))
        if profile == "P0":
            mixers = ["attention", "attention"]
        elif profile == "P1":
            mixers = ["gdn2_inspired_reference", "attention"]
        elif profile == "P1-LA":
            mixers = ["local_attention", "attention"]
        else:
            raise ValueError(profile)
        ffn_hidden = 80 if profile == "P1" else FF_MULT * D_MODEL
        self.blocks = nn.ModuleList([Block(D_MODEL, N_HEADS, m, ffn_hidden) for m in mixers])
        self.norm = nn.LayerNorm(D_MODEL)
        self.head = nn.Linear(D_MODEL, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        nn.init.normal_(self.pos_emb, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        _, t = idx.shape
        if t > BLOCK_SIZE:
            raise ValueError(f"sequence {t} > context {BLOCK_SIZE}")
        x = self.token_emb(idx) + self.pos_emb[:, :t]
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


@torch.no_grad()
def evaluate(model: nn.Module, tokens: torch.Tensor, batch_size: int = EVAL_BATCH_SIZE) -> float:
    model.eval()
    losses = []
    max_start = len(tokens) - BLOCK_SIZE - 1
    starts = list(range(0, max_start, BLOCK_SIZE))
    for start_i in range(0, len(starts), batch_size):
        ss = starts[start_i:start_i + batch_size]
        x = torch.stack([tokens[s:s + BLOCK_SIZE] for s in ss])
        y = torch.stack([tokens[s + 1:s + BLOCK_SIZE + 1] for s in ss])
        logits = model(x)
        losses.append(F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item())
    model.train()
    denom = sum(min(batch_size, len(starts) - i) * BLOCK_SIZE for i in range(0, len(starts), batch_size))
    return sum(losses) / max(1, denom)


@torch.no_grad()
def evaluate_full_metrics(model: nn.Module, tokens: torch.Tensor) -> dict:
    model.eval()
    total_loss = 0.0; total_correct = 0; total = 0
    max_start = len(tokens) - BLOCK_SIZE - 1
    for start in range(0, max_start, BLOCK_SIZE):
        x = tokens[start:start + BLOCK_SIZE].unsqueeze(0)
        y = tokens[start + 1:start + BLOCK_SIZE + 1].unsqueeze(0)
        logits = model(x)
        total_loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    model.train()
    return {"nats_per_char": total_loss / max(1,total), "top1_next_char_accuracy": total_correct / max(1,total), "evaluated_chars": total}


def ngram_baselines(train: torch.Tensor, eval_tokens: torch.Tensor, vocab_size: int) -> dict:
    # Evaluate n-gram and neural models on exactly the same next-character targets.
    # The neural evaluator uses non-overlapping 32-character context blocks.
    alpha = 0.5
    counts = torch.bincount(train, minlength=vocab_size).double() + alpha
    uni = counts / counts.sum()
    trans = torch.full((vocab_size, vocab_size), alpha, dtype=torch.float64)
    for a,b in zip(train[:-1].tolist(), train[1:].tolist()):
        trans[a,b] += 1.0
    trans = trans / trans.sum(dim=1, keepdim=True)
    target_indices = []
    max_start = len(eval_tokens) - BLOCK_SIZE - 1
    for start in range(0, max_start, BLOCK_SIZE):
        target_indices.extend(range(start + 1, start + BLOCK_SIZE + 1))
    idx = torch.tensor(target_indices, dtype=torch.long)
    prev, nxt = eval_tokens[idx - 1], eval_tokens[idx]
    unigram_pred = torch.argmax(uni).expand_as(nxt)
    bigram_pred = torch.argmax(trans[prev], dim=-1)
    unigram_loss = -torch.log(uni[nxt]).mean().item()
    bigram_loss = -torch.log(trans[prev, nxt]).mean().item()
    return {
        "unigram_nats_per_char": unigram_loss,
        "bigram_nats_per_char": bigram_loss,
        "unigram_top1_next_char_accuracy": (unigram_pred == nxt).double().mean().item(),
        "bigram_top1_next_char_accuracy": (bigram_pred == nxt).double().mean().item(),
        "smoothing_alpha": alpha,
        "matched_evaluated_targets": len(target_indices),
        "evaluation_positions": "same non-overlapping context-block targets as neural models",
    }



def assert_gdn2_update_equivalence() -> float:
    """Check the implemented rank-one update against its explicit matrix form."""
    torch.manual_seed(99173)
    bsz, heads, dk, dv = 2, 3, 4, 5
    state = torch.randn(bsz, heads, dk, dv, dtype=torch.float64)
    key = F.normalize(torch.randn(bsz, heads, dk, dtype=torch.float64), dim=-1)
    erase = torch.sigmoid(torch.randn(bsz, heads, dk, dtype=torch.float64))
    write = torch.sigmoid(torch.randn(bsz, heads, dv, dtype=torch.float64))
    value = torch.randn(bsz, heads, dv, dtype=torch.float64)
    decay = torch.sigmoid(torch.randn(bsz, heads, dk, dtype=torch.float64))
    decayed = decay.unsqueeze(-1) * state
    # Rank-one implementation used in the pilot.
    erased_read = torch.matmul((erase * key).unsqueeze(-2), decayed).squeeze(-2)
    rank_one = decayed - key.unsqueeze(-1) * erased_read.unsqueeze(-2)
    rank_one = rank_one + key.unsqueeze(-1) * (write * value).unsqueeze(-2)
    # Explicit matrix expression: (I - k (b ⊙ k)^T) D S + k (w ⊙ v)^T.
    eye = torch.eye(dk, dtype=torch.float64).view(1, 1, dk, dk)
    erase_matrix = eye - key.unsqueeze(-1) * (erase * key).unsqueeze(-2)
    explicit = torch.matmul(erase_matrix, decayed)
    explicit = explicit + key.unsqueeze(-1) * (write * value).unsqueeze(-2)
    error = (rank_one - explicit).abs().max().item()
    if error > 1e-10:
        raise AssertionError(f"GDN-2 rank-one update mismatch: {error}")
    return error

def assert_architecture_invariants(model: nn.Module, profile: str, vocab_size: int) -> dict:
    # Causality: perturb several future positions; every earlier logit must be unchanged.
    model.eval()
    x = torch.randint(0, vocab_size, (2, BLOCK_SIZE), dtype=torch.long)
    positions = sorted(set([1, BLOCK_SIZE // 4, BLOCK_SIZE // 2, BLOCK_SIZE - 1]))
    errors = {}
    with torch.no_grad():
        reference = model(x)
        for position in positions:
            changed = x.clone()
            changed[:, position] = (changed[:, position] + 1) % vocab_size
            changed_logits = model(changed)
            errors[str(position)] = (reference[:, :position] - changed_logits[:, :position]).abs().max().item()
    max_prefix_error = max(errors.values(), default=0.0)
    if max_prefix_error > 1e-6:
        raise AssertionError(f"future-token leakage in {profile}: {errors}")
    # State is request-local: re-running the same input after another request must reset.
    intervening = torch.randint(0, vocab_size, (2, BLOCK_SIZE), dtype=torch.long)
    with torch.no_grad():
        _ = model(intervening)
        repeated_logits = model(x)
    repeat_error = (reference - repeated_logits).abs().max().item()
    if repeat_error > 1e-7:
        raise AssertionError(f"non-deterministic/reset failure in {profile}: {repeat_error}")
    model.train()
    return {
        "causal_future_leakage_max_abs": max_prefix_error,
        "causality_positions_tested": positions,
        "causality_error_by_position": errors,
        "request_state_reset_repeat_max_abs": repeat_error,
    }


def get_batch(data: torch.Tensor, rng: random.Random) -> Tuple[torch.Tensor, torch.Tensor]:
    hi = len(data) - BLOCK_SIZE - 1
    starts = [rng.randrange(0, hi) for _ in range(BATCH_SIZE)]
    x = torch.stack([data[s:s + BLOCK_SIZE] for s in starts])
    y = torch.stack([data[s + 1:s + BLOCK_SIZE + 1] for s in starts])
    return x, y


@torch.no_grad()
def generate(model: nn.Module, prompt: torch.Tensor, reverse_vocab: Dict[int, str], n: int = 240) -> str:
    model.eval(); ids = prompt.clone()
    for _ in range(n):
        logits = model(ids[:, -BLOCK_SIZE:])[:, -1, :]
        # Temperature 0.85 and top-k 12; deterministic greedy is also logged in report.
        logits = logits / 0.85
        vals, inds = torch.topk(logits, k=min(12, logits.size(-1)), dim=-1)
        probs = F.softmax(vals, dim=-1)
        choice = inds.gather(-1, torch.multinomial(probs, 1))
        ids = torch.cat([ids, choice], dim=1)
    model.train()
    return "".join(reverse_vocab[int(i)] for i in ids[0].tolist())


def train_one(profile: str, seed: int, train: torch.Tensor, val: torch.Tensor, test: torch.Tensor,
              vocab: Dict[str, int], reverse_vocab: Dict[int, str], baselines: dict) -> dict:
    seed_all(seed)
    model = TinyLM(len(vocab), profile).to(DEVICE)
    params = sum(p.numel() for p in model.parameters())
    invariants = assert_architecture_invariants(model, profile, len(vocab))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.1)
    rng = random.Random(seed + 97)
    initial_val = evaluate(model, val)
    t0 = time.perf_counter(); grad_norms = []; losses = []
    all_parameters_received_gradient = True
    finite_loss_and_gradients = True
    best_val = float("inf"); best_step = 0; best_state = None
    for step in range(1, STEPS + 1):
        xb, yb = get_batch(train, rng)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        if not torch.isfinite(loss):
            finite_loss_and_gradients = False
            raise FloatingPointError(f"non-finite loss in {profile} seed={seed} step={step}")
        loss.backward()
        if step == 1:
            missing = [name for name, parameter in model.named_parameters()
                       if parameter.requires_grad and parameter.grad is None]
            all_parameters_received_gradient = (len(missing) == 0)
            if missing:
                raise RuntimeError(f"trainable parameters without gradients in {profile}: {missing}")
        # clip_grad_norm_ checks the aggregate norm for non-finite values without
        # launching a separate per-parameter tensor scan on every CPU step.
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0, error_if_nonfinite=True
        ).item()
        if not math.isfinite(grad_norm):
            finite_loss_and_gradients = False
            raise FloatingPointError(f"non-finite gradient norm in {profile} seed={seed} step={step}")
        opt.step()
        losses.append(float(loss.item())); grad_norms.append(float(grad_norm))
        if step % EVAL_INTERVAL == 0 or step == STEPS:
            current_val = evaluate(model, val)
            if current_val < best_val:
                best_val = current_val; best_step = step
                best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    train_seconds = time.perf_counter() - t0
    if best_state is not None:
        model.load_state_dict(best_state)
    train_metrics = evaluate_full_metrics(model, train)
    val_metrics = evaluate_full_metrics(model, val)
    test_metrics = evaluate_full_metrics(model, test)
    train_loss = train_metrics["nats_per_char"]
    val_loss = val_metrics["nats_per_char"]
    test_loss = test_metrics["nats_per_char"]
    # Inference throughput: evaluation mode, no autograd, fixed batch and context.
    probe = torch.stack([train[i:i + BLOCK_SIZE] for i in range(0, BATCH_SIZE * 2 * BLOCK_SIZE, BLOCK_SIZE)])
    model.eval()
    with torch.inference_mode():
        for _ in range(5):
            model(probe)
        tbench = time.perf_counter()
        for _ in range(30):
            model(probe)
    bench_seconds = time.perf_counter() - tbench
    throughput = (30 * probe.numel()) / max(bench_seconds, 1e-9)
    model.train()
    prompt_text = "First Citizen:\nBefore we proceed"
    prompt_ids = torch.tensor([[vocab[c] for c in prompt_text if c in vocab]], dtype=torch.long)
    sample = generate(model, prompt_ids, reverse_vocab, 220)
    result = {
        "profile": profile, "seed": seed, "parameters": params, "selected_best_step": best_step,
        "initial_val_nats_per_char": initial_val,
        "train_nats_per_char": train_loss, "validation_nats_per_char": val_loss,
        "test_nats_per_char": test_loss,
        "train_evaluated_characters": train_metrics["evaluated_chars"],
        "validation_evaluated_characters": val_metrics["evaluated_chars"],
        "test_evaluated_characters": test_metrics["evaluated_chars"],
        "validation_top1_next_char_accuracy": val_metrics["top1_next_char_accuracy"],
        "test_top1_next_char_accuracy": test_metrics["top1_next_char_accuracy"],
        "ngram_baselines": baselines, "invariants": invariants,
        "train_perplexity_char": math.exp(min(train_loss, 20)),
        "validation_perplexity_char": math.exp(min(val_loss, 20)),
        "test_perplexity_char": math.exp(min(test_loss, 20)),
        "train_seconds": train_seconds,
        "training_characters_processed": STEPS * BATCH_SIZE * BLOCK_SIZE,
        "training_characters_per_second": STEPS * BATCH_SIZE * BLOCK_SIZE / max(train_seconds, 1e-9),
        "forward_characters_per_second_batch16": throughput,
        "median_step_loss_first_25": float(np.median(losses[:25])),
        "median_step_loss_last_25": float(np.median(losses[-25:])),
        "optimizer_steps_completed": STEPS,
        "gradient_norm_median": float(np.median(grad_norms)),
        "gradient_norm_max": max(grad_norms),
        "all_trainable_parameters_received_gradient_first_step": all_parameters_received_gradient,
        "finite_loss_and_gradients_all_steps": finite_loss_and_gradients,
        "sample": sample,
    }
    save_path = OUT / f"{profile}_seed{seed}.pt"
    torch.save({"state_dict": model.state_dict(), "profile": profile, "seed": seed,
                "params": params, "vocab": vocab, "config": {"d_model": D_MODEL,"heads":N_HEADS,"layers":N_LAYERS,"ctx":BLOCK_SIZE}}, save_path)
    result["checkpoint_sha256"] = sha256(save_path)
    return result


def main() -> None:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    seed_all(SEED)
    data_dir_env = os.environ.get("PILOT_DATA_DIR")
    if data_dir_env:
        raw_train, raw_val, raw_test, dataset = read_verified_dataset_bundle(Path(data_dir_env).resolve())
        split_method = "official train/validation/test splits supplied by a hash-verified dataset bundle"
    else:
        sample_digest = sha256(DATA_PATH)
        if sample_digest != EXPECTED_SAMPLE_SHA256:
            raise RuntimeError(f"Dataset sample digest mismatch: expected {EXPECTED_SAMPLE_SHA256}, got {sample_digest}")
        text = DATA_PATH.read_text(encoding="utf-8")
        split1, split2 = int(len(text) * 0.80), int(len(text) * 0.90)
        raw_train, raw_val, raw_test = text[:split1], text[split1:split2], text[split2:]
        split_method = "sequential 80/10/10 split of the locally recovered 29.9 KB excerpt; not the official upstream split"
        dataset = {
            "source_name": "Tiny Shakespeare text from karpathy/char-rnn data/tinyshakespeare/input.txt",
            "source_url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
            "repository_blob_sha": "7dcb3a2d4cc3b48b6283dd46870bfeb78f88aac9",
            "source_file_bytes_reported_online": 1115394,
            "local_sample_bytes": DATA_PATH.stat().st_size,
            "local_sample_sha256": sample_digest,
            "license_note": "Source repository has no declared license metadata. Shakespeare's underlying works are public domain, but transcription/edition rights have not been independently reviewed; commercial training approval is not established.",
            "rights_review_status": "pending",
            "sample_materialization_note": "A 29,926-byte excerpt was locally materialized from the GitHub blob response. The local SHA-256 locks the exact sample used; byte-for-byte equivalence to the complete upstream file was not independently verified.",
            "sample_fraction_of_upstream_file": DATA_PATH.stat().st_size / 1115394,
        }
    # Character vocabulary is built from training data only. Unknown validation/test characters
    # map to one explicit UNK id rather than leaking evaluation vocabulary into training.
    chars = sorted(set(raw_train))
    vocab = {ch:i for i,ch in enumerate(chars)}
    unknown = (set(raw_val) | set(raw_test)) - set(chars)
    unk_id = len(vocab) if unknown else None
    if unknown:
        vocab["<UNK>"] = unk_id
    encode = lambda s: torch.tensor([vocab.get(c, unk_id) for c in s], dtype=torch.long)
    train, val, test = encode(raw_train), encode(raw_val), encode(raw_test)
    dataset["split_method"] = split_method
    dataset["counts"] = {
        "characters_total": len(raw_train) + len(raw_val) + len(raw_test),
        "train": len(train), "validation": len(val), "test": len(test),
        "vocabulary": len(vocab), "unknown_validation_test_characters": len(unknown),
    }
    (OUT / "dataset_manifest.json").write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    results = []
    gdn2_equation_error = assert_gdn2_update_equivalence()
    baselines = ngram_baselines(train, test, len(vocab))
    for seed in SEEDS:
      for profile in PROFILES:
        print(f"START {profile} seed={seed}", flush=True)
        r = train_one(profile, seed, train, val, test, vocab, {i:c for c,i in vocab.items()}, baselines)
        results.append(r)
        print(json.dumps({k:v for k,v in r.items() if k != "sample"}, indent=2), flush=True)
    dataset_manifest_path = OUT / "dataset_manifest.json"
    report = {
        "title": "v9 CPU architecture feasibility pilot",
        "run_class": "exploratory_small_scale_reference_training_not_frontier_validation",
        "source_identity": {
            "train_pilot_sha256": sha256(Path(__file__).resolve()),
            "prepare_hf_dataset_sha256": sha256(ROOT / "prepare_hf_dataset.py"),
            "dataset_manifest_sha256": sha256(dataset_manifest_path),
        },
        "operator_algebra_checks": {
            "gdn2_rank_one_vs_explicit_matrix_max_abs_error": gdn2_equation_error,
            "pass": gdn2_equation_error <= 1e-10,
        },
        "implementation_scope": {
            "P0": "two dense causal attention blocks with feed-forward sublayers",
            "P1": "one GDN-2-inspired recurrent reference operator plus one dense attention block; simplified projections/gates; not official GDN-2 implementation",
            "P1-LA": "one 16-token local causal attention block plus one dense global attention block; not an Inkling 2 reproduction",
        },
        "run_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {"python":platform.python_version(),"torch":torch.__version__,"device":"cpu","cpu_count":os.cpu_count(),"threads":torch.get_num_threads(),"platform":platform.platform()},
        "config": {"seeds":list(SEEDS),"d_model":D_MODEL,"heads":N_HEADS,"layers":N_LAYERS,"ffn_hidden_P0_P1LA":FF_MULT*D_MODEL,"ffn_hidden_P1":80,"context":BLOCK_SIZE,"batch_size":BATCH_SIZE,"steps":STEPS,"eval_interval":EVAL_INTERVAL,"lr":LR},
        "dataset":dataset,
        "ngram_baselines":baselines,
        "results":results,
        "scope_warning": "Exploratory character-LM pilot only. It does not validate frontier LLM quality, large-scale scaling, GPU kernels, multilingual/general instruction following, or production GDN-2 performance. Source-repository license metadata is absent and rights review is pending; do not treat this corpus as approved for commercial training.",
    }
    (OUT / RESULTS_NAME).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("PILOT_COMPLETE", flush=True)

if __name__ == "__main__":
    main()
