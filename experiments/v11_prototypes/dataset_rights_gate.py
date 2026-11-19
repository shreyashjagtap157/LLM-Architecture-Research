#!/usr/bin/env python3
"""Fail-closed checker for a human-reviewed dataset rights and privacy manifest.

This tool validates the completeness and internal consistency of supplied
attestations. It does not inspect source websites, verify legal authority,
interpret licences, certify privacy compliance, or provide legal advice.
A PASS means only that the manifest records an approval for the requested use.
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Sequence

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ALLOWED_USES = {
    "research_training",
    "commercial_training",
    "evaluation",
    "model_distribution",
    "hosted_service",
    "public_data_redistribution",
}


class ManifestError(ValueError):
    """Raised when a rights manifest is incomplete or contradictory."""


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ManifestError(f"{path}: expected object")
    return value


def _require_text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{path}: required non-empty string")
    return value.strip()


def _require_sha256(value: Any, path: str) -> str:
    text = _require_text(value, path)
    if not SHA256_RE.fullmatch(text):
        raise ManifestError(f"{path}: expected lowercase 64-character SHA-256")
    return text


def _require_date(value: Any, path: str) -> str:
    text = _require_text(value, path)
    try:
        date.fromisoformat(text)
    except ValueError as exc:
        raise ManifestError(f"{path}: expected ISO date YYYY-MM-DD") from exc
    if len(text) != 10:
        raise ManifestError(f"{path}: expected ISO date YYYY-MM-DD")
    return text


def _require_allowed_uses(value: Any, path: str, requested_use: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ManifestError(f"{path}: expected non-empty list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        use = _require_text(item, f"{path}[{index}]")
        if use not in ALLOWED_USES:
            raise ManifestError(f"{path}[{index}]: unsupported use '{use}'")
        if use in normalized:
            raise ManifestError(f"{path}: duplicate use '{use}'")
        normalized.append(use)
    if requested_use not in normalized:
        raise ManifestError(f"{path}: requested use '{requested_use}' is not approved")
    return normalized


def validate_manifest(manifest: Any, requested_use: str) -> dict[str, Any]:
    """Validate the human attestation manifest for one specific intended use."""
    if requested_use not in ALLOWED_USES:
        raise ManifestError(f"use: unsupported requested use '{requested_use}'")
    root = _require_mapping(manifest, "manifest")
    if root.get("schema_version") != "1.0":
        raise ManifestError("schema_version: expected '1.0'")

    dataset = _require_mapping(root.get("dataset"), "dataset")
    dataset_id = _require_text(dataset.get("repo_id"), "dataset.repo_id")
    _require_text(dataset.get("configuration"), "dataset.configuration")
    revision = _require_text(dataset.get("revision"), "dataset.revision")
    if not COMMIT_RE.fullmatch(revision):
        raise ManifestError("dataset.revision: expected full 40-character immutable commit SHA")
    _require_text(dataset.get("source_url"), "dataset.source_url")
    _require_sha256(dataset.get("dataset_card_sha256"), "dataset.dataset_card_sha256")
    _require_sha256(dataset.get("terms_snapshot_sha256"), "dataset.terms_snapshot_sha256")

    rights = _require_mapping(root.get("rights_review"), "rights_review")
    if rights.get("status") != "approved":
        raise ManifestError("rights_review.status: must be 'approved'; pending/rejected manifests fail closed")
    _require_text(rights.get("reviewer_id"), "rights_review.reviewer_id")
    _require_text(rights.get("reviewer_role"), "rights_review.reviewer_role")
    _require_date(rights.get("reviewed_at"), "rights_review.reviewed_at")
    _require_text(rights.get("jurisdiction_scope"), "rights_review.jurisdiction_scope")
    _require_text(rights.get("decision_note"), "rights_review.decision_note")
    _require_sha256(rights.get("evidence_bundle_sha256"), "rights_review.evidence_bundle_sha256")
    _require_allowed_uses(rights.get("approved_uses"), "rights_review.approved_uses", requested_use)

    sources = root.get("source_inventory")
    if not isinstance(sources, list) or not sources:
        raise ManifestError("source_inventory: expected non-empty list of included/excluded upstream sources")
    included_sources = 0
    source_ids: set[str] = set()
    for index, raw_source in enumerate(sources):
        path = f"source_inventory[{index}]"
        source = _require_mapping(raw_source, path)
        source_id = _require_text(source.get("source_id"), f"{path}.source_id")
        if source_id in source_ids:
            raise ManifestError(f"{path}.source_id: duplicate source ID '{source_id}'")
        source_ids.add(source_id)
        if not isinstance(source.get("included"), bool):
            raise ManifestError(f"{path}.included: expected boolean")
        if not source["included"]:
            continue
        included_sources += 1
        _require_text(source.get("upstream_uri"), f"{path}.upstream_uri")
        upstream_revision = _require_text(source.get("immutable_revision"), f"{path}.immutable_revision")
        if not (COMMIT_RE.fullmatch(upstream_revision) or SHA256_RE.fullmatch(upstream_revision)):
            raise ManifestError(f"{path}.immutable_revision: expected full 40-character commit SHA or 64-character content SHA-256")
        _require_text(source.get("license_id"), f"{path}.license_id")
        _require_sha256(source.get("license_evidence_sha256"), f"{path}.license_evidence_sha256")
        if source.get("rights_status") != "approved":
            raise ManifestError(f"{path}.rights_status: included source must be approved")
        _require_allowed_uses(source.get("approved_uses"), f"{path}.approved_uses", requested_use)
    if included_sources == 0:
        raise ManifestError("source_inventory: at least one included upstream source is required")

    privacy = _require_mapping(root.get("privacy_review"), "privacy_review")
    if privacy.get("status") != "approved":
        raise ManifestError("privacy_review.status: must be 'approved'; pending/rejected manifests fail closed")
    _require_text(privacy.get("reviewer_id"), "privacy_review.reviewer_id")
    _require_date(privacy.get("reviewed_at"), "privacy_review.reviewed_at")
    _require_sha256(privacy.get("report_sha256"), "privacy_review.report_sha256")
    _require_text(privacy.get("retention_and_removal_plan"), "privacy_review.retention_and_removal_plan")

    return {
        "status": "ATTESTATIONS_PRESENT_FOR_REQUESTED_USE",
        "dataset_id": dataset_id,
        "revision": revision,
        "requested_use": requested_use,
        "included_source_count": included_sources,
        "rights_reviewer_id": rights["reviewer_id"],
        "privacy_reviewer_id": privacy["reviewer_id"],
        "disclaimer": (
            "The checker validated manifest structure and recorded attestations only. "
            "It did not verify legal accuracy, source authority, licence interpretation, "
            "privacy compliance or actual dataset contents. This is not legal clearance."
        ),
    }


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ManifestError(f"manifest_file: cannot read file ({exc.__class__.__name__})") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"manifest_file: invalid JSON at line {exc.lineno}, column {exc.colno}") from exc
    return _require_mapping(data, "manifest")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="human-reviewed JSON rights manifest")
    parser.add_argument("--use", required=True, choices=sorted(ALLOWED_USES), help="specific intended use to authorize")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON decision report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        report = validate_manifest(manifest, args.use)
    except ManifestError as exc:
        parser.error(str(exc))
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
