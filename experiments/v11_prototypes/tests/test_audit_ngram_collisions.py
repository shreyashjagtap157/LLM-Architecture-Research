from __future__ import annotations

import json

import pytest

from experiments.v11_prototypes.audit_ngram_collisions import (
    audit_ngram_collisions,
    main,
    read_token_ids,
)


def test_no_collision_when_unique_ngrams_fit_separate_buckets() -> None:
    report = audit_ngram_collisions(
        [1, 2, 1, 2, 3], table_size=10_000, orders=(2,), d_model=8
    )
    row = report["per_order"][0]
    assert row["complete_ngram_observations"] == 4
    assert row["unique_ngram_types"] == 3
    assert row["occupied_buckets"] == 3
    assert row["unique_type_collision_excess"] == 0
    assert row["occurrence_collision_exposure"] == 0.0


def test_single_bucket_exposes_all_distinct_ngram_types_as_collisions() -> None:
    report = audit_ngram_collisions(
        [1, 2, 1, 2], table_size=1, orders=(2,), d_model=4
    )
    row = report["per_order"][0]
    assert row["unique_ngram_types"] == 2
    assert row["occupied_buckets"] == 1
    assert row["unique_type_collision_excess"] == 1
    assert row["unique_types_in_collided_buckets"] == 2
    assert row["occurrence_collision_exposure"] == 1.0


def test_storage_estimates_are_explicit_and_scale_with_table_size() -> None:
    report = audit_ngram_collisions(
        [1, 2, 3, 4], table_size=100, orders=(2, 3), d_model=16,
        dtype_bytes=2, optimizer_state_bytes_per_param=4,
    )
    assert report["embedding_table_parameter_count"] == 2 * 100 * 16
    assert report["estimated_weight_bytes"] == 6_400
    assert report["estimated_gradient_bytes"] == 6_400
    assert report["estimated_adam_moment_bytes"] == 25_600
    assert report["estimated_training_storage_bytes"] == 38_400


def test_complete_ngrams_only_excludes_sequences_shorter_than_order() -> None:
    report = audit_ngram_collisions(
        [1, 2], table_size=16, orders=(2, 3), d_model=4
    )
    assert report["per_order"][0]["complete_ngram_observations"] == 1
    assert report["per_order"][1]["complete_ngram_observations"] == 0


def test_token_file_parser_checks_integer_ids_and_optional_limit(tmp_path) -> None:
    data = tmp_path / "ids.txt"
    data.write_text("1 2 3\n4 5 6\n", encoding="utf-8")
    assert read_token_ids(data, vocab_size=8, max_tokens=4) == [1, 2, 3, 4]
    data.write_text("1 two 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid integer"):
        read_token_ids(data, vocab_size=8)
    data.write_text("1 8 3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="outside"):
        read_token_ids(data, vocab_size=8)


def test_cli_writes_json_without_emitting_token_ids(tmp_path, capsys) -> None:
    data = tmp_path / "ids.txt"
    data.write_text("1 2 1 2\n", encoding="utf-8")
    output = tmp_path / "reports" / "audit.json"
    result = main([
        "--token-ids", str(data), "--vocab-size", "4", "--table-size", "8",
        "--orders", "2", "3", "--d-model", "8", "--output", str(output),
    ])
    assert result == 0
    assert capsys.readouterr().out == ""
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["token_count"] == 4
    assert report["per_order"][0]["complete_ngram_observations"] == 3
    assert "1 2 1 2" not in output.read_text(encoding="utf-8")
