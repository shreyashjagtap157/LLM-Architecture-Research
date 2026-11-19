#!/usr/bin/env python3
"""Export an immutable Hugging Face text dataset into hash-verified train/val/test files.

Network execution was not possible in the current pilot environment. This loader is provided
for a network-enabled environment and deliberately leaves dataset rights pending unless the
operator explicitly supplies a reviewed license statement and --rights-reviewed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_split_bundle(
    splits: dict[str, Iterable[str]],
    output_dir: Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Stream train/validation/test records to disk and write immutable hashes."""
    required = ("train", "validation", "test")
    if any(name not in splits for name in required):
        raise ValueError("train, validation and test splits must all be provided")
    output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest: dict[str, Any] = {}
    for name in required:
        path = output_dir / f"{name}.txt"
        row_count = 0
        character_count = 0
        with path.open("w", encoding="utf-8", newline="") as handle:
            for item in splits[name]:
                if not isinstance(item, str):
                    raise TypeError(f"All {name} text records must be strings")
                if row_count:
                    handle.write("\n")
                    character_count += 1
                handle.write(item)
                character_count += len(item)
                row_count += 1
        if row_count == 0 or path.stat().st_size == 0:
            raise ValueError(f"{name} split must contain at least one non-empty text record")
        split_manifest[name] = {
            "file": path.name,
            "rows": row_count,
            "characters": character_count,
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    rights_reviewed = bool(metadata.get("rights_reviewed", False))
    license_statement = metadata.get("license_statement")
    if rights_reviewed and not license_statement:
        raise ValueError("rights_reviewed=true requires a reviewed license_statement")
    manifest = {
        "format": "llm-text-split-bundle-v1",
        "dataset_id": metadata.get("dataset_id", "unspecified"),
        "configuration": metadata.get("configuration", "unspecified"),
        "revision": metadata.get("revision", "unspecified"),
        "text_field": metadata.get("text_field", "text"),
        "source_uri": metadata.get("source_uri"),
        "library": metadata.get("library", "datasets"),
        "library_version": metadata.get("library_version"),
        "streaming_export": bool(metadata.get("streaming_export", False)),
        "rights_review_status": "approved_for_declared_uses" if rights_reviewed else "pending",
        "license_statement": license_statement,
        "allowed_uses": metadata.get("allowed_uses", ["training", "validation", "model_selection", "evaluation"]) if rights_reviewed else [],
        "review_note": metadata.get("review_note", "License and permitted uses require human review; the exporter does not make legal determinations."),
        "splits": split_manifest,
    }
    manifest_path = output_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest_digest = sha256(manifest_path)
    # Keep the digest in a sidecar to avoid a circular self-hash inside the manifest.
    sidecar_path = output_dir / "dataset_manifest.sha256"
    sidecar_path.write_text(f"{manifest_digest}  {manifest_path.name}\n", encoding="ascii")
    manifest["manifest_sha256"] = manifest_digest
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--config", default="wikitext-2-raw-v1")
    parser.add_argument("--revision", required=True, help="Immutable dataset commit SHA (40 hex characters).")
    parser.add_argument("--text-field", default="text")
    parser.add_argument("--streaming", action="store_true", help="Stream dataset rows instead of materializing each split in RAM.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--license-statement", default=None)
    parser.add_argument("--rights-reviewed", action="store_true", help="Operator attests that dataset rights have been reviewed.")
    parser.add_argument("--review-note", default="Manual review required for the dataset license, source terms and all declared uses.")
    parser.add_argument(
        "--allowed-use", action="append", choices=("training", "validation", "model_selection", "evaluation", "reward_model_training"),
        help="Allowed use asserted by the operator after review; repeat as needed.",
    )
    args = parser.parse_args()

    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be a full 40-character immutable commit SHA")
    if args.rights_reviewed and not args.license_statement:
        parser.error("--rights-reviewed requires --license-statement")

    try:
        import datasets
    except ImportError as exc:
        raise SystemExit("Install dataset_requirements.txt before using this online loader") from exc

    corpus = datasets.load_dataset(
        args.dataset,
        args.config,
        revision=args.revision,
        trust_remote_code=False,
        streaming=args.streaming,
    )
    exported: dict[str, Iterable[str]] = {}
    for split in ("train", "validation", "test"):
        if split not in corpus:
            raise ValueError(f"Dataset revision does not provide required split: {split}")
        split_dataset = corpus[split]
        column_names = getattr(split_dataset, "column_names", None)
        if column_names is None:
            features = getattr(split_dataset, "features", None)
            column_names = list(features.keys()) if features is not None else []
        if args.text_field not in column_names:
            raise ValueError(f"Missing text field {args.text_field!r} in split {split}")
        def iter_text(dataset_split=split_dataset):
            for row in dataset_split:
                value = row.get(args.text_field)
                yield str(value or "")
        exported[split] = iter_text()
    manifest = export_split_bundle(
        exported,
        args.output_dir,
        {
            "dataset_id": args.dataset,
            "configuration": args.config,
            "revision": args.revision,
            "text_field": args.text_field,
            "source_uri": f"https://huggingface.co/datasets/{args.dataset}",
            "library": "datasets",
            "library_version": datasets.__version__,
            "streaming_export": args.streaming,
            "rights_reviewed": args.rights_reviewed,
            "license_statement": args.license_statement,
            "allowed_uses": args.allowed_use or ["training", "validation", "model_selection", "evaluation"],
            "review_note": args.review_note,
        },
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
