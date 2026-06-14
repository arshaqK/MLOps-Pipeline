"""
test_drift.py — Unit tests for distribution drift detection.

Provides a reference distribution and a clearly shifted distribution
and asserts that drift is correctly flagged.
"""

import sys
import os
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingestion.drift_detector import DriftDetector


def _make_df(mean, std, n=100, cols=("feature_0", "feature_1")):
    """Helper to create a DataFrame with given mean and std per column."""
    data = {col: np.random.normal(mean, std, n) for col in cols}
    return pd.DataFrame(data)


def test_drift_detected_on_large_shift():
    """
    Clearly shifted distribution should be flagged as drifted.
    Baseline mean=0, batch mean=10 — z-score will be very high.
    """
    detector = DriftDetector(threshold=0.5, min_baseline_batches=1)

    # Feed one warm-up batch to build baseline (mean~0)
    baseline_df = _make_df(mean=0, std=1)
    detector.check(baseline_df)

    # Now feed a clearly shifted batch (mean~10)
    shifted_df = _make_df(mean=10, std=1)
    drifted = detector.check(shifted_df)

    assert len(drifted) > 0, "Expected drift to be detected on large distribution shift"


def test_no_drift_on_similar_distribution():
    """
    Similar distributions should not be flagged as drifted.
    Both batches have mean~0, std~1.
    """
    detector = DriftDetector(threshold=2.0, min_baseline_batches=1)

    baseline_df = _make_df(mean=0, std=1)
    detector.check(baseline_df)

    similar_df = _make_df(mean=0.1, std=1)
    drifted = detector.check(similar_df)

    assert len(drifted) == 0, "Expected no drift on similar distributions"


def test_warmup_batches_no_drift():
    """
    During warm-up phase, no drift should be flagged regardless of distribution.
    """
    detector = DriftDetector(threshold=0.5, min_baseline_batches=3)

    # First 3 batches are warm-up — no drift should fire
    for _ in range(3):
        shifted_df = _make_df(mean=100, std=1)
        drifted = detector.check(shifted_df)
        assert len(drifted) == 0, "Expected no drift during warm-up phase"


def test_drift_reset():
    """After reset, detector should behave as if freshly initialized."""
    detector = DriftDetector(threshold=0.5, min_baseline_batches=1)

    baseline_df = _make_df(mean=0, std=1)
    detector.check(baseline_df)

    detector.reset()

    # After reset, first batch is warm-up again — no drift
    shifted_df = _make_df(mean=10, std=1)
    drifted = detector.check(shifted_df)
    assert len(drifted) == 0, "Expected no drift after reset during warm-up"