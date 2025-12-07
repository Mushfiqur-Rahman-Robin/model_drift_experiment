import pytest
import numpy as np
from src.data.generator import DriftGenerator

@pytest.mark.unit
class TestDriftGenerator:
    def test_drift_images(self):
        # Test with 0 severity (no change)
        images = np.zeros((10, 28, 28), dtype=np.float32)
        drifted = DriftGenerator.drift_images(images, severity=0.0)
        assert np.allclose(images, drifted)

        # Test with high severity
        drifted_high = DriftGenerator.drift_images(images, severity=1.0)
        assert not np.allclose(images, drifted_high)
        assert drifted_high.min() >= 0.0
        assert drifted_high.max() <= 1.0

    def test_drift_text(self):
        texts = ["hello world", "test sentence"]

        # Test with 0 severity
        drifted = DriftGenerator.drift_text(texts, severity=0.0)
        assert drifted == texts

        # Test with 1.0 severity (always drift)
        # We need to mock random to ensure deterministic behavior for coverage if needed,
        # but for now let's just check that it changes.
        drifted_high = DriftGenerator.drift_text(texts, severity=1.0)
        assert len(drifted_high) == len(texts)
        assert drifted_high != texts

        # Check that vocab words are inserted
        vocab = ["drift", "noise", "error", "unknown", "###", "!!!"]
        found_vocab = False
        for text in drifted_high:
            for word in text.split():
                if word in vocab:
                    found_vocab = True
                    break
        assert found_vocab
