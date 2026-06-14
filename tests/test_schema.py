"""
test_schema.py — Unit tests for schema change detection in ingestion.py.

Simulates two batches with differing schemas and asserts that the correct
feature_added / feature_removed flags are raised.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.ingestion import _compare_schemas


def test_feature_added():
    """New feature appears in current schema — should be detected as added."""
    previous = ["feature_0", "feature_1", "label"]
    current  = ["feature_0", "feature_1", "feature_2", "label"]

    added, removed = _compare_schemas(previous, current)

    assert "feature_2" in added, "Expected feature_2 to be detected as added"
    assert len(removed) == 0, "Expected no removed features"


def test_feature_removed():
    """Existing feature disappears from schema — should be detected as removed."""
    previous = ["feature_0", "feature_1", "label"]
    current  = ["feature_0", "label"]

    added, removed = _compare_schemas(previous, current)

    assert "feature_1" in removed, "Expected feature_1 to be detected as removed"
    assert len(added) == 0, "Expected no added features"


def test_feature_added_and_removed():
    """One feature added and one removed simultaneously."""
    previous = ["feature_0", "feature_1", "label"]
    current  = ["feature_0", "feature_2", "label"]

    added, removed = _compare_schemas(previous, current)

    assert "feature_2" in added,   "Expected feature_2 to be added"
    assert "feature_1" in removed, "Expected feature_1 to be removed"


def test_no_schema_change():
    """Identical schemas — no changes should be detected."""
    schema = ["feature_0", "feature_1", "label"]

    added, removed = _compare_schemas(schema, schema)

    assert len(added)   == 0, "Expected no added features"
    assert len(removed) == 0, "Expected no removed features"


def test_first_batch_no_baseline():
    """First batch has no previous schema — nothing should be flagged."""
    added, removed = _compare_schemas(None, ["feature_0", "feature_1", "label"])

    assert len(added)   == 0
    assert len(removed) == 0