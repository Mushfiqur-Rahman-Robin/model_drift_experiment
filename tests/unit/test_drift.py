import pytest
import numpy as np
from src.drift.detector import DriftDetector

@pytest.mark.unit
class TestDriftDetector:
    def test_no_drift(self):
        detector = DriftDetector(threshold=0.05)
        # Same distribution
        ref = np.random.normal(0, 1, (100, 10))
        curr = np.random.normal(0, 1, (100, 10))

        result = detector.detect_drift(ref, curr)
        # Should not detect drift (mostly)
        # Note: Statistical tests can have false positives, but with same seed/dist it should be fine mostly.
        # We check if drift_ratio is low.
        assert result["drift_ratio"] < 0.5

    def test_drift_detected(self):
        detector = DriftDetector(threshold=0.05)
        # Different distribution
        ref = np.random.normal(0, 1, (100, 10))
        curr = np.random.normal(2, 1, (100, 10)) # Shift mean

        result = detector.detect_drift(ref, curr)
        assert result["drift_detected"] is True
        assert result["drift_ratio"] > 0.5

    def test_1d_input(self):
        detector = DriftDetector()
        # 1D arrays should be reshaped to (N, 1)
        ref = np.random.normal(0, 1, (100,))
        curr = np.random.normal(0, 1, (100,))

        result = detector.detect_drift(ref, curr)
        assert "drift_detected" in result
        assert result["details"]["total_features_checked"] == 1
