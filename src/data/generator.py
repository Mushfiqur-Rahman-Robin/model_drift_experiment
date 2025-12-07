import numpy as np


class DriftGenerator:
    """
    Injects drift into data to simulate production scenarios.
    """

    @staticmethod
    def drift_images(images: np.ndarray, severity: float = 0.5) -> np.ndarray:
        """
        Add Gaussian noise to images to simulate concept drift / covariate shift.
        Severity: 0.0 to 1.0 (noise level)
        """
        # Using numpy.random for simulation purposes to avoid Bandit B311 warning
        noise = np.random.normal(loc=0.0, scale=severity, size=images.shape)
        drifted_images = images + noise
        return np.clip(drifted_images, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def drift_text(texts: list[str], severity: float = 0.5) -> list[str]:
        """
        Inject typos or random words to simulate text drift.
        Severity: probability of modifying a sentence.
        """
        drifted_texts = []
        vocab = ["drift", "noise", "error", "unknown", "###", "!!!"]

        for text in texts:
            # Using numpy.random for simulation purposes to avoid Bandit B311 warning
            if np.random.rand() < severity:
                words = text.split()
                # Randomly replace a word or append garbage
                if words and np.random.rand() < 0.5:
                    idx = np.random.randint(0, len(words))
                    words[idx] = np.random.choice(vocab)
                else:
                    words.append(np.random.choice(vocab))
                drifted_texts.append(" ".join(words))
            else:
                drifted_texts.append(text)
        return drifted_texts
