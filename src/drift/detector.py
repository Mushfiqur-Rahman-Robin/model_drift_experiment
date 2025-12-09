import logging
from typing import Any

import numpy as np
from scipy.stats import ks_2samp

logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects drift between reference data and current data.
    Uses Kolmogorov-Smirnov test on feature embeddings.
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold

    def detect_drift(
        self, reference_embeddings: np.ndarray, current_embeddings: np.ndarray
    ) -> dict[str, Any]:
        """
        Compare reference embeddings with current embeddings.
        Returns a dictionary with drift status and statistics.
        """
        logger.debug(f"Starting drift detection with threshold: {self.threshold}")
        # Ensure 2D arrays
        if reference_embeddings.ndim == 1:
            reference_embeddings = reference_embeddings.reshape(-1, 1)
        if current_embeddings.ndim == 1:
            current_embeddings = current_embeddings.reshape(-1, 1)

        n_features = reference_embeddings.shape[1]
        drift_detected = False
        p_values = []

        # Perform KS test for each feature dimension
        # For high-dimensional data (like BERT embeddings), we might want to reduce dimensions first (PCA/UMAP)
        # or just average the p-values / check if any feature drifted significantly.
        # For this experiment, we'll check if a significant portion of features have drifted.

        drifted_features_count = 0

        # Limit features to check to avoid performance issues if embeddings are huge
        features_to_check = min(n_features, 50)

        for i in range(features_to_check):
            ref_feat = reference_embeddings[:, i]
            curr_feat = current_embeddings[:, i]

            _stat, p_value = ks_2samp(ref_feat, curr_feat)
            p_values.append(p_value)

            if p_value < self.threshold:
                drifted_features_count += 1

        # Heuristic: if more than 10% of checked features show drift, flag it.
        drift_ratio = drifted_features_count / features_to_check
        drift_detected = drift_ratio > 0.1

        if drift_detected:
            logger.warning(
                f"Drift detected in {drifted_features_count}/{features_to_check} features (ratio: {drift_ratio:.2f}) "
                f"with average p-value: {np.mean(p_values):.4f}"
            )
        else:
            logger.info(
                f"No significant drift detected. {drifted_features_count}/{features_to_check} features drifted "
                f"(ratio: {drift_ratio:.2f}), average p-value: {np.mean(p_values):.4f}"
            )
        return {
            "drift_detected": drift_detected,
            "drift_ratio": drift_ratio,
            "avg_p_value": np.mean(p_values),
            "details": {
                "drifted_features": drifted_features_count,
                "total_features_checked": features_to_check,
            },
        }
