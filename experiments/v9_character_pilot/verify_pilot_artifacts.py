#!/usr/bin/env python3
"""Verify the v9 CPU pilot data, metrics, model checkpoints and invariants."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
PROFILES = ("P0", "P1", "P1-LA")
SEEDS = (20261118, 20261119)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str, checks: list[str]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append(message)


def main() -> None:
    checks: list[str] = []
    manifest = json.loads((RESULTS / "dataset_manifest.json").read_text(encoding="utf-8"))
    results = json.loads((RESULTS / "pilot_results.json").read_text(encoding="utf-8"))
    data_path = ROOT / "tinyshakespeare_30k.txt"
    require(data_path.is_file(), "training sample exists", checks)
    require(sha256(data_path) == manifest["local_sample_sha256"], "training sample digest matches dataset manifest", checks)
    require(manifest["rights_review_status"] == "pending", "source rights remain explicitly pending", checks)
    require(results["dataset"]["local_sample_sha256"] == manifest["local_sample_sha256"], "merged results use the same sample digest", checks)
    require(results["config"]["seeds"] == list(SEEDS), "two declared random seeds are present", checks)
    require(len(results["results"]) == len(SEEDS) * len(PROFILES), "six full model/seed runs are recorded", checks)
    require(results["operator_algebra_checks"]["pass"] is True, "GDN-2 rank-one update matches explicit matrix form", checks)
    require(results["operator_algebra_checks"]["gdn2_rank_one_vs_explicit_matrix_max_abs_error"] < 1e-10, "GDN-2 algebra error is below tolerance", checks)
    require(results["ngram_baselines"]["matched_evaluated_targets"] == 2976, "n-gram baseline target count is fixed", checks)

    mapping = {(row["profile"], row["seed"]): row for row in results["results"]}
    require(len(mapping) == len(results["results"]), "no duplicate profile/seed result rows", checks)
    for seed in SEEDS:
        require(all((profile, seed) in mapping for profile in PROFILES), f"all three profiles exist for seed {seed}", checks)
    for (profile, seed), row in mapping.items():
        require(row["optimizer_steps_completed"] == results["config"]["steps"], f"{profile}/{seed}: completed optimizer-step budget", checks)
        require(row["test_evaluated_characters"] == results["ngram_baselines"]["matched_evaluated_targets"], f"{profile}/{seed}: same held-out positions as n-gram baseline", checks)
        require(row["all_trainable_parameters_received_gradient_first_step"], f"{profile}/{seed}: every trainable parameter receives a gradient", checks)
        require(row["finite_loss_and_gradients_all_steps"], f"{profile}/{seed}: finite loss/gradient checks pass", checks)
        require(row["invariants"]["causality_positions_tested"] == [1, 8, 16, 31], f"{profile}/{seed}: four causality perturbation positions tested", checks)
        require(max(row["invariants"]["causality_error_by_position"].values()) == 0.0, f"{profile}/{seed}: no future-token leakage detected", checks)
        require(row["invariants"]["request_state_reset_repeat_max_abs"] == 0.0, f"{profile}/{seed}: request-local state resets exactly", checks)
        for metric in ("train_nats_per_char", "validation_nats_per_char", "test_nats_per_char", "test_top1_next_char_accuracy"):
            require(math.isfinite(row[metric]), f"{profile}/{seed}: {metric} is finite", checks)
        ckpt = RESULTS / f"{profile}_seed{seed}.pt"
        require(ckpt.is_file(), f"{profile}/{seed}: checkpoint exists", checks)
        require(sha256(ckpt) == row["checkpoint_sha256"], f"{profile}/{seed}: checkpoint digest matches result", checks)

    # An independently repeated seed-20261118 run must match loss, accuracy and checkpoint hashes.
    repeat = json.loads((RESULTS / "repeat_20261118_current.json").read_text(encoding="utf-8"))
    repeat_map = {(r["profile"], r["seed"]): r for r in repeat["results"]}
    for profile in PROFILES:
        left = mapping[(profile, 20261118)]
        right = repeat_map[(profile, 20261118)]
        for key in ("train_nats_per_char", "validation_nats_per_char", "test_nats_per_char", "test_top1_next_char_accuracy", "checkpoint_sha256"):
            require(left[key] == right[key], f"repeatability: {profile} {key} is bit-identical", checks)

    # A later repeat was run with source identity captured at launch. It is required
    # to match the completed seed-20261118 metrics/checkpoints and to identify the
    # exact trainer, exporter and dataset manifest bytes used in that repeat.
    source_locked = json.loads((RESULTS / "source_locked_repeat_20261118.json").read_text(encoding="utf-8"))
    source_identity = source_locked.get("source_identity", {})
    require(set(source_identity) == {"train_pilot_sha256", "prepare_hf_dataset_sha256", "dataset_manifest_sha256"},
            "source-locked repeat records all three source identities", checks)
    source_paths = {
        "train_pilot_sha256": ROOT / "train_pilot.py",
        "prepare_hf_dataset_sha256": ROOT / "prepare_hf_dataset.py",
        "dataset_manifest_sha256": RESULTS / "dataset_manifest.json",
    }
    for key, path in source_paths.items():
        require(len(source_identity.get(key, "")) == 64, f"source-locked repeat {key} is a SHA-256 digest", checks)
        require(source_identity.get(key) == sha256(path), f"source-locked repeat {key} matches current artifact", checks)
    source_map = {(r["profile"], r["seed"]): r for r in source_locked["results"]}
    require(len(source_map) == len(PROFILES), "source-locked repeat contains exactly three unique profile runs", checks)
    for profile in PROFILES:
        left = mapping[(profile, 20261118)]
        right = source_map[(profile, 20261118)]
        for key in ("train_nats_per_char", "validation_nats_per_char", "test_nats_per_char", "test_top1_next_char_accuracy", "checkpoint_sha256"):
            require(left[key] == right[key], f"source-locked repeat: {profile} {key} matches archived seed run", checks)

    # Verify the separate-split dataset loader using a local test bundle; this is a loader smoke test,
    # not an online WikiText training run.
    bundle_dir = RESULTS / "local_split_bundle_smoke"
    bundle = json.loads((bundle_dir / "dataset_manifest.json").read_text(encoding="utf-8"))
    for split, item in bundle["splits"].items():
        path = bundle_dir / item["file"]
        require(sha256(path) == item["sha256"], f"split bundle {split} file hash validates", checks)
    sidecar = (bundle_dir / "dataset_manifest.sha256").read_text(encoding="ascii").strip().split()
    require(len(sidecar) == 2 and sidecar[0] == sha256(bundle_dir / "dataset_manifest.json"), "split bundle manifest checksum sidecar validates", checks)
    smoke = json.loads((RESULTS / "bundle_loader_hardened_smoke.json").read_text(encoding="utf-8"))
    require(smoke["dataset"]["manifest_sha256_verified"] is True, "trainer accepted hash-verified dataset manifest", checks)
    require(smoke["dataset"]["local_split_sha256_verified"] is True, "trainer accepted hash-verified separate splits", checks)
    require(smoke["results"][0]["optimizer_steps_completed"] == 1, "separate-split loader completed a one-step training smoke test", checks)
    smoke_checkpoint = RESULTS / "P0_seed903341.pt"
    require(smoke_checkpoint.is_file(), "separate-split smoke checkpoint exists", checks)
    require(sha256(smoke_checkpoint) == smoke["results"][0]["checkpoint_sha256"], "separate-split smoke checkpoint digest matches", checks)

    report = {
        "status": "passed",
        "check_count": len(checks),
        "checks": checks,
        "scope": "Artifact integrity and small-scale training invariants only; not model quality or online dataset training validation.",
    }
    (RESULTS / "pilot_verification_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": report["check_count"]}, indent=2))


if __name__ == "__main__":
    main()
