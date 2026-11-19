from __future__ import annotations

import copy
import json

import pytest

from experiments.v11_prototypes.dataset_rights_gate import ManifestError, main, validate_manifest

SHA = "a" * 64
REV = "b" * 40


@pytest.fixture
def approved_manifest() -> dict:
    return {
        "schema_version": "1.0",
        "dataset": {
            "repo_id": "example/research-corpus",
            "configuration": "default",
            "revision": REV,
            "source_url": "https://example.invalid/dataset",
            "dataset_card_sha256": SHA,
            "terms_snapshot_sha256": SHA,
        },
        "rights_review": {
            "status": "approved",
            "reviewer_id": "reviewer-001",
            "reviewer_role": "data-rights-reviewer",
            "reviewed_at": "2026-11-19",
            "jurisdiction_scope": "declared project jurisdictions",
            "decision_note": "Source inventory and intended use reviewed.",
            "evidence_bundle_sha256": SHA,
            "approved_uses": ["research_training", "evaluation"],
        },
        "source_inventory": [
            {
                "source_id": "SRC-001",
                "included": True,
                "upstream_uri": "https://example.invalid/source",
                "immutable_revision": REV,
                "license_id": "Example-License-1.0",
                "license_evidence_sha256": SHA,
                "rights_status": "approved",
                "approved_uses": ["research_training", "evaluation"],
            },
            {"source_id": "SRC-EXCLUDED", "included": False},
        ],
        "privacy_review": {
            "status": "approved",
            "reviewer_id": "privacy-reviewer-001",
            "reviewed_at": "2026-11-19",
            "report_sha256": SHA,
            "retention_and_removal_plan": "Documented removal and opt-out procedure.",
        },
    }


def test_approved_manifest_passes_only_for_requested_use(approved_manifest: dict) -> None:
    result = validate_manifest(approved_manifest, "research_training")
    assert result["status"] == "ATTESTATIONS_PRESENT_FOR_REQUESTED_USE"
    assert result["included_source_count"] == 1
    assert "not legal clearance" in result["disclaimer"]


def test_pending_template_fails_closed(approved_manifest: dict) -> None:
    manifest = copy.deepcopy(approved_manifest)
    manifest["rights_review"]["status"] = "pending"
    with pytest.raises(ManifestError, match="must be 'approved'"):
        validate_manifest(manifest, "research_training")


def test_requesting_unapproved_use_is_rejected(approved_manifest: dict) -> None:
    with pytest.raises(ManifestError, match="not approved"):
        validate_manifest(approved_manifest, "commercial_training")


def test_included_source_must_have_approved_rights(approved_manifest: dict) -> None:
    manifest = copy.deepcopy(approved_manifest)
    manifest["source_inventory"][0]["rights_status"] = "pending"
    with pytest.raises(ManifestError, match="included source must be approved"):
        validate_manifest(manifest, "research_training")


def test_included_source_requires_its_own_licence_evidence(approved_manifest: dict) -> None:
    manifest = copy.deepcopy(approved_manifest)
    manifest["source_inventory"][0]["license_evidence_sha256"] = ""
    with pytest.raises(ManifestError, match="license_evidence_sha256"):
        validate_manifest(manifest, "research_training")


def test_no_included_sources_is_rejected(approved_manifest: dict) -> None:
    manifest = copy.deepcopy(approved_manifest)
    manifest["source_inventory"] = [{"source_id": "EXCLUDED", "included": False}]
    with pytest.raises(ManifestError, match="at least one included"):
        validate_manifest(manifest, "research_training")


def test_privacy_review_must_be_approved(approved_manifest: dict) -> None:
    manifest = copy.deepcopy(approved_manifest)
    manifest["privacy_review"]["status"] = "pending"
    with pytest.raises(ManifestError, match="privacy_review.status"):
        validate_manifest(manifest, "research_training")


def test_immutable_dataset_revision_must_be_full_sha(approved_manifest: dict) -> None:
    manifest = copy.deepcopy(approved_manifest)
    manifest["dataset"]["revision"] = "main"
    with pytest.raises(ManifestError, match="full 40-character"):
        validate_manifest(manifest, "research_training")


def test_source_list_cannot_duplicate_source_ids(approved_manifest: dict) -> None:
    manifest = copy.deepcopy(approved_manifest)
    manifest["source_inventory"].append(copy.deepcopy(manifest["source_inventory"][0]))
    with pytest.raises(ManifestError, match="duplicate source ID"):
        validate_manifest(manifest, "research_training")


def test_cli_emits_attestation_not_legal_clearance(approved_manifest: dict, tmp_path, capsys) -> None:
    manifest_path = tmp_path / "approved.json"
    manifest_path.write_text(json.dumps(approved_manifest), encoding="utf-8")
    assert main(["--manifest", str(manifest_path), "--use", "research_training"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["status"] == "ATTESTATIONS_PRESENT_FOR_REQUESTED_USE"
    assert "did not verify legal accuracy" in report["disclaimer"]
