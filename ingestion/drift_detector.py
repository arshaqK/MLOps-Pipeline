"""
drift_detector.py — Distribution drift detection for numeric features.

Compares per-feature statistics (mean, std) of each incoming batch against
a rolling baseline.  A z-score-style formula measures how far the current
batch mean has moved relative to the baseline standard deviation; if it
exceeds a configurable threshold the feature is flagged as drifted.

Design decisions:
  • Rolling baseline update — after each non-drifted batch the baseline
    is updated with a small learning rate so gradual, legitimate shifts
    don't permanently anchor the detector to stale statistics.
  • Std-floor guard — a minimum std of 1e-6 prevents division-by-zero on
    constant or near-constant features.
  • Only numeric columns are checked; categorical/ID columns are ignored
    automatically via pandas select_dtypes.
  • The detector is stateless across process restarts; a future improvement
    would persist baseline stats to disk so they survive pod restarts.
"""

import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

log = logging.getLogger(__name__)


@dataclass
class FeatureStats:
    """Stores the rolling baseline statistics for a single feature."""
    mean: float
    std: float


class DriftDetector:
    """
    Detects distribution drift by comparing batch statistics to a baseline.

    Parameters
    ----------
    threshold : float
        z-score threshold above which a feature is considered drifted.
        Default 0.5 is intentionally sensitive for demo purposes;
        production systems typically use 2.0–3.0.
    baseline_lr : float
        Learning rate for updating the baseline after each non-drifted batch.
        A value of 0.1 means the baseline slowly tracks legitimate data evolution.
    min_baseline_batches : int
        Number of batches to consume before drift detection starts.
        Ensures the baseline is stable before comparisons begin.
    """

    def __init__(
        self,
        threshold: float = 0.5,
        baseline_lr: float = 0.1,
        min_baseline_batches: int = 3,
    ) -> None:
        self.threshold            = threshold
        self.baseline_lr          = baseline_lr
        self.min_baseline_batches = min_baseline_batches

        self._baseline: dict[str, FeatureStats] = {}  # feature → baseline stats
        self._batches_seen: int = 0                   # warm-up counter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, df: pd.DataFrame) -> list[str]:
        """
        Evaluate a new batch for distribution drift.

        Returns a list of feature names that have drifted beyond the
        configured threshold.  An empty list means no drift detected.

        Parameters
        ----------
        df : pd.DataFrame
            The incoming batch.  Only numeric columns are evaluated.
        """
        # Extract only numeric columns — non-numeric features are out of scope
        numeric_df = df.select_dtypes(include=[np.number])

        if numeric_df.empty:
            log.debug("No numeric features in batch; skipping drift check.")
            return []

        self._batches_seen += 1

        # Warm-up phase: build the baseline without firing any alerts
        if self._batches_seen <= self.min_baseline_batches:
            self._update_baseline(numeric_df, force=True)
            log.debug(
                "Warm-up batch %d/%d — building baseline, no drift checks yet.",
                self._batches_seen, self.min_baseline_batches,
            )
            return []

        drifted: list[str] = []

        for col in numeric_df.columns:
            batch_mean = float(numeric_df[col].mean())
            batch_std  = float(numeric_df[col].std(ddof=1)) if len(numeric_df) > 1 else 0.0

            if col not in self._baseline:
                # New feature has no baseline yet — register it and move on
                self._baseline[col] = FeatureStats(mean=batch_mean, std=max(batch_std, 1e-6))
                log.debug("New feature '%s' added to drift baseline.", col)
                continue

            baseline = self._baseline[col]

            # z-score: how many baseline-std-deviations is the batch mean away?
            z_score = abs(batch_mean - baseline.mean) / max(baseline.std, 1e-6)

            log.debug(
                "Feature '%s': batch_mean=%.4f baseline_mean=%.4f "
                "baseline_std=%.4f z=%.4f (threshold=%.2f)",
                col, batch_mean, baseline.mean, baseline.std, z_score, self.threshold,
            )

            if z_score > self.threshold:
                drifted.append(col)
                log.warning(
                    "DRIFT — '%s' z=%.3f > %.3f  "
                    "(batch_mean=%.4f, baseline_mean=%.4f, baseline_std=%.4f)",
                    col, z_score, self.threshold,
                    batch_mean, baseline.mean, baseline.std,
                )
            else:
                # Non-drifted feature: gently update baseline toward current batch
                self._update_feature_baseline(col, batch_mean, batch_std)

        return drifted

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_baseline(self, df: pd.DataFrame, force: bool = False) -> None:
        """Bulk-update or initialise baseline stats from a DataFrame."""
        for col in df.columns:
            batch_mean = float(df[col].mean())
            batch_std  = float(df[col].std(ddof=1)) if len(df) > 1 else 0.0

            if force or col not in self._baseline:
                # Hard-set during warm-up or for unseen features
                self._baseline[col] = FeatureStats(
                    mean=batch_mean,
                    std=max(batch_std, 1e-6),
                )
            else:
                self._update_feature_baseline(col, batch_mean, batch_std)

    def _update_feature_baseline(
        self, col: str, batch_mean: float, batch_std: float
    ) -> None:
        """
        Exponential moving average update for a single feature's baseline.
        Keeps the baseline tracking slow, legitimate distribution evolution.
        """
        lr = self.baseline_lr
        old = self._baseline[col]
        self._baseline[col] = FeatureStats(
            mean=old.mean * (1 - lr) + batch_mean * lr,
            std=max(old.std  * (1 - lr) + batch_std  * lr, 1e-6),
        )

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def get_baseline_summary(self) -> dict[str, dict]:
        """Return the current baseline stats — useful for logging / debugging."""
        return {
            col: {"mean": round(s.mean, 6), "std": round(s.std, 6)}
            for col, s in self._baseline.items()
        }

    def reset(self) -> None:
        """Reset detector state — useful between test runs or after retraining."""
        self._baseline.clear()
        self._batches_seen = 0
        log.info("DriftDetector baseline reset.")