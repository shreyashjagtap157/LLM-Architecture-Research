#!/usr/bin/env python3
"""Offline regression tests for the immutable, hash-verified dataset bundle path."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from prepare_hf_dataset import export_split_bundle
from train_pilot import read_verified_dataset_bundle


REVISION = "a" * 40
SPLITS = {
    "train": ["alpha", "beta", "gamma"],
    "validation": ["delta"],
    "test": ["epsilon"],
}


class DatasetPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.metadata = {
            "dataset_id": "fixture/public-text",
            "configuration": "default",
            "revision": REVISION,
            "text_field": "text",
            "source_uri": "https://example.invalid/datasets/fixture/public-text",
            "rights_reviewed": False,
        }
        self.manifest = export_split_bundle(SPLITS, self.root, self.metadata)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_export_writes_manifest_digest_sidecar(self) -> None:
        self.assertTrue((self.root / "dataset_manifest.json").is_file())
        self.assertTrue((self.root / "dataset_manifest.sha256").is_file())
        disk_manifest = json.loads((self.root / "dataset_manifest.json").read_text())
        self.assertNotIn("manifest_sha256", disk_manifest)
        self.assertEqual(len(self.manifest["manifest_sha256"]), 64)

    def test_default_approved_uses_include_model_selection(self) -> None:
        approved = self.root / "approved"
        metadata = dict(self.metadata, rights_reviewed=True, license_statement="Reviewed test fixture only")
        result = export_split_bundle(SPLITS, approved, metadata)
        self.assertIn("model_selection", result["allowed_uses"])

    def test_unreviewed_bundle_fails_closed_by_default(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PILOT_ALLOW_UNREVIEWED_DATA", None)
            with self.assertRaises(PermissionError):
                read_verified_dataset_bundle(self.root)

    def test_explicit_exploratory_override_loads_unreviewed_bundle(self) -> None:
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            train, validation, test, manifest = read_verified_dataset_bundle(self.root)
        self.assertEqual(train, "alpha\nbeta\ngamma")
        self.assertEqual(validation, "delta")
        self.assertEqual(test, "epsilon")
        self.assertTrue(manifest["manifest_sha256_verified"])
        self.assertTrue(manifest["local_split_sha256_verified"])

    def test_split_tampering_is_rejected(self) -> None:
        (self.root / "train.txt").write_text("tampered", encoding="utf-8")
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(RuntimeError, "train split SHA-256 mismatch"):
                read_verified_dataset_bundle(self.root)

    def test_manifest_tampering_is_rejected(self) -> None:
        path = self.root / "dataset_manifest.json"
        path.write_text(path.read_text() + " ", encoding="utf-8")
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(RuntimeError, "manifest SHA-256 mismatch"):
                read_verified_dataset_bundle(self.root)

    def test_path_traversal_is_rejected(self) -> None:
        path = self.root / "dataset_manifest.json"
        data = json.loads(path.read_text())
        data["splits"]["train"]["file"] = "../outside.txt"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "dataset_manifest.sha256").write_text(f"{digest}  dataset_manifest.json\n", encoding="ascii")
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(ValueError, "Unsafe split path"):
                read_verified_dataset_bundle(self.root)

    def test_missing_manifest_sidecar_is_rejected(self) -> None:
        (self.root / "dataset_manifest.sha256").unlink()
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(FileNotFoundError, "dataset_manifest.sha256"):
                read_verified_dataset_bundle(self.root)

    def test_mutable_or_missing_revision_is_rejected(self) -> None:
        path = self.root / "dataset_manifest.json"
        data = json.loads(path.read_text())
        data["revision"] = "main"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        import hashlib
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (self.root / "dataset_manifest.sha256").write_text(f"{digest}  dataset_manifest.json\n", encoding="ascii")
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(ValueError, "immutable 40-character revision"):
                read_verified_dataset_bundle(self.root)

    def test_streaming_iterables_export_without_list_materialization(self) -> None:
        streaming = self.root / "streaming"
        result = export_split_bundle(
            {name: iter(rows) for name, rows in SPLITS.items()},
            streaming,
            dict(self.metadata, streaming_export=True),
        )
        self.assertTrue(json.loads((streaming / "dataset_manifest.json").read_text())["streaming_export"])
        self.assertEqual(result["splits"]["train"]["rows"], 3)
        self.assertEqual((streaming / "train.txt").read_text(), "alpha\nbeta\ngamma")

    def test_online_cli_rejects_mutable_revision_before_network_access(self) -> None:
        script = Path(__file__).with_name("prepare_hf_dataset.py")
        result = subprocess.run(
            [sys.executable, str(script), "--revision", "main", "--output-dir", str(self.root / "bad-revision")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("40-character immutable commit SHA", result.stderr)

    def test_online_cli_requires_review_statement_when_approving_rights(self) -> None:
        script = Path(__file__).with_name("prepare_hf_dataset.py")
        result = subprocess.run(
            [sys.executable, str(script), "--revision", REVISION, "--output-dir", str(self.root / "no-license"), "--rights-reviewed"],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("--rights-reviewed requires --license-statement", result.stderr)

    def test_approved_bundle_must_include_required_uses(self) -> None:
        approved = self.root / "insufficient-uses"
        export_split_bundle(
            SPLITS, approved,
            dict(self.metadata, rights_reviewed=True, license_statement="Reviewed fixture", allowed_uses=["training"]),
        )
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PILOT_ALLOW_UNREVIEWED_DATA", None)
            with self.assertRaisesRegex(PermissionError, "approval does not cover required uses"):
                read_verified_dataset_bundle(approved)

    def _rewrite_manifest_with_digest(self, directory: Path) -> None:
        import hashlib
        manifest_path = directory / "dataset_manifest.json"
        manifest_path.write_text(json.dumps(json.loads(manifest_path.read_text()), indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        (directory / "dataset_manifest.sha256").write_text(f"{digest}  dataset_manifest.json\n", encoding="ascii")

    def test_duplicate_split_filename_is_rejected(self) -> None:
        path = self.root / "dataset_manifest.json"
        data = json.loads(path.read_text())
        data["splits"]["validation"]["file"] = "train.txt"
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self._rewrite_manifest_with_digest(self.root)
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(ValueError, "distinct split files"):
                read_verified_dataset_bundle(self.root)

    def test_byte_identical_split_contents_are_rejected(self) -> None:
        import hashlib
        train_content = (self.root / "train.txt").read_text(encoding="utf-8")
        val_path = self.root / "validation.txt"
        val_path.write_text(train_content, encoding="utf-8")
        data_path = self.root / "dataset_manifest.json"
        data = json.loads(data_path.read_text())
        data["splits"]["validation"].update({
            "bytes": val_path.stat().st_size,
            "characters": len(train_content),
            "sha256": hashlib.sha256(val_path.read_bytes()).hexdigest(),
        })
        data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self._rewrite_manifest_with_digest(self.root)
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                read_verified_dataset_bundle(self.root)

    def test_split_character_count_metadata_is_verified(self) -> None:
        path = self.root / "dataset_manifest.json"
        data = json.loads(path.read_text())
        data["splits"]["train"]["characters"] += 1
        path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        self._rewrite_manifest_with_digest(self.root)
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaisesRegex(RuntimeError, "character count mismatch"):
                read_verified_dataset_bundle(self.root)

    def test_missing_split_rejected(self) -> None:
        (self.root / "validation.txt").unlink()
        with patch.dict(os.environ, {"PILOT_ALLOW_UNREVIEWED_DATA": "1"}):
            with self.assertRaises(FileNotFoundError):
                read_verified_dataset_bundle(self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
