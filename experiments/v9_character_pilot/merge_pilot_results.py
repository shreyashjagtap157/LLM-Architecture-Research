#!/usr/bin/env python3
"""Merge independent one-seed pilot result files after identity and artifact checks."""
from __future__ import annotations
import argparse, hashlib, json, math, statistics
from pathlib import Path

PROFILE_ORDER = ("P0", "P1", "P1-LA")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def mean(xs):
    return statistics.mean(xs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("result_files", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--repeat-file", type=Path, default=None, help="Optional exact repeat run for determinism checks.")
    args = p.parse_args()

    reports = [json.loads(x.read_text(encoding="utf-8")) for x in args.result_files]
    if len(reports) < 2:
        raise SystemExit("At least two seed-specific result files are required")
    base = reports[0]
    expected_common = {k: base["config"][k] for k in base["config"] if k != "seeds"}
    expected_dataset_sha = base["dataset"].get("local_sample_sha256")
    source_identities = [report.get("source_identity") for report in reports]
    if any(source_identities) and (not all(isinstance(item, dict) for item in source_identities) or any(item != source_identities[0] for item in source_identities[1:])):
        raise ValueError("Seed runs have mixed or mismatched source/data manifest identities")
    expected_source_identity = source_identities[0] if source_identities else None
    results = []
    seeds = []
    for report in reports:
        current_common = {k: report["config"][k] for k in report["config"] if k != "seeds"}
        if current_common != expected_common:
            raise ValueError("Seed runs have mismatched model/training configurations")
        if report["dataset"].get("local_sample_sha256") != expected_dataset_sha:
            raise ValueError("Seed runs use different data sample digests")
        if report.get("source_identity") != expected_source_identity:
            raise ValueError("Seed runs use different or incomplete source/data manifest identity")
        if report["ngram_baselines"] != base["ngram_baselines"]:
            raise ValueError("Seed runs use different n-gram baseline definitions/results")
        if report["operator_algebra_checks"] != base["operator_algebra_checks"]:
            raise ValueError("Seed runs disagree on the GDN-2 equation check")
        run_seeds = report["config"].get("seeds", [])
        if len(run_seeds) != 1:
            raise ValueError("Each input must contain exactly one seed")
        seed = run_seeds[0]
        if seed in seeds:
            raise ValueError(f"Duplicate seed {seed}")
        seeds.append(seed)
        profile_names = [r["profile"] for r in report["results"]]
        if profile_names != list(PROFILE_ORDER):
            raise ValueError(f"Seed {seed} must contain profiles in order {PROFILE_ORDER}, got {profile_names}")
        results.extend(report["results"])

    seeds = sorted(seeds)
    for row in results:
        if row["optimizer_steps_completed"] != base["config"]["steps"]:
            raise ValueError(f"Incomplete optimizer steps for {row['profile']} seed={row['seed']}")
        if row["test_evaluated_characters"] != base["ngram_baselines"]["matched_evaluated_targets"]:
            raise ValueError("Neural model and n-gram baseline did not evaluate the same test targets")
        if not row["all_trainable_parameters_received_gradient_first_step"]:
            raise ValueError("At least one trainable model parameter did not receive a gradient")
        if not row["finite_loss_and_gradients_all_steps"]:
            raise ValueError("Non-finite loss/gradient result")
        if max(row["invariants"]["causality_error_by_position"].values()) > 1e-6:
            raise ValueError("Causality invariant failed")
        if row["invariants"]["request_state_reset_repeat_max_abs"] > 1e-7:
            raise ValueError("Request-state reset invariant failed")
        checkpoint = args.checkpoint_dir / f"{row['profile']}_seed{row['seed']}.pt"
        if not checkpoint.is_file() or digest(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"Checkpoint missing or digest mismatch: {checkpoint}")

    aggregate = {}
    for profile in PROFILE_ORDER:
        rows = [r for r in results if r["profile"] == profile]
        aggregate[profile] = {
            "runs": len(rows),
            "parameters": rows[0]["parameters"],
            "test_nats_per_char_mean": mean([r["test_nats_per_char"] for r in rows]),
            "test_nats_per_char_sample_sd": statistics.stdev([r["test_nats_per_char"] for r in rows]),
            "test_top1_accuracy_mean": mean([r["test_top1_next_char_accuracy"] for r in rows]),
            "training_seconds_mean": mean([r["train_seconds"] for r in rows]),
            "inference_characters_per_second_mean": mean([r["forward_characters_per_second_batch16"] for r in rows]),
        }
    base_nll = aggregate["P0"]["test_nats_per_char_mean"]
    p1_nll = aggregate["P1"]["test_nats_per_char_mean"]
    aggregate["comparisons"] = {
        "P1_vs_P0_test_nll_relative_change_percent": 100.0 * (p1_nll / base_nll - 1.0),
        "P1_vs_P0_test_top1_absolute_percentage_point_change": 100.0 * (aggregate["P1"]["test_top1_accuracy_mean"] - aggregate["P0"]["test_top1_accuracy_mean"]),
        "P1_vs_P0_training_time_ratio": aggregate["P1"]["training_seconds_mean"] / aggregate["P0"]["training_seconds_mean"],
        "P0_vs_P1_inference_throughput_ratio": aggregate["P0"]["inference_characters_per_second_mean"] / aggregate["P1"]["inference_characters_per_second_mean"],
        "P1_vs_bigram_test_nll_relative_excess_percent": 100.0 * (p1_nll / base["ngram_baselines"]["bigram_nats_per_char"] - 1.0),
    }

    out = dict(base)
    out["run_timestamp_utc"] = max(r["run_timestamp_utc"] for r in reports)
    out["config"] = dict(base["config"])
    out["config"]["seeds"] = seeds
    out["results"] = sorted(results, key=lambda r: (r["seed"], PROFILE_ORDER.index(r["profile"])))
    out["aggregate"] = aggregate
    out["merge_audit"] = {
        "input_files": [str(x) for x in args.result_files],
        "seed_runs": seeds,
        "all_model_checkpoint_hashes_verified": True,
        "same_dataset_config_and_baselines": True,
        "matched_neural_ngram_test_targets": True,
    }
    if args.repeat_file is not None:
        repeated = json.loads(args.repeat_file.read_text(encoding="utf-8"))
        repeat_rows = {(r["profile"], r["seed"]): r for r in repeated["results"]}
        compared_seed = repeated["config"]["seeds"][0]
        compared_fields = ("train_nats_per_char", "validation_nats_per_char", "test_nats_per_char", "test_top1_next_char_accuracy", "checkpoint_sha256")
        exact = True
        for profile in PROFILE_ORDER:
            original = next(r for r in results if r["profile"] == profile and r["seed"] == compared_seed)
            duplicate = repeat_rows[(profile, compared_seed)]
            exact = exact and all(original[field] == duplicate[field] for field in compared_fields)
        if not exact:
            raise ValueError("Repeat run metrics/checkpoint digests are not bit-identical")
        out["reproducibility"] = {
            "repeat_seed": compared_seed,
            "compared_profiles": list(PROFILE_ORDER),
            "compared_fields": list(compared_fields),
            "all_compared_fields_bit_identical": True,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "seeds": seeds, "runs": len(results), "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
