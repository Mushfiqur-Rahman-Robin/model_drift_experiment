# Drift Detection & Application Workflow

## The Core Idea: What is Drift?
"Model Drift" (or Data Drift) happens when the data your model sees in production starts to look different from the data it was trained on. This usually causes the model's performance to drop.

In this application, we **simulate** this drift intentionally to demonstrate how to detect it using the **Kolmogorov-Smirnov (KS) Test**.

## Workflow (Under the Hood)

When you hit the `/detect-drift/{model_type}` endpoint, the following process occurs:

### 1. Establish a Baseline (Reference Data)
*   The app generates a batch of clean, synthetic data (e.g., perfect MNIST digits or clean sentences).
*   It passes this data through the model (CNN or BERT) to get **Embeddings**.
    *   *Embeddings* are the model's internal representation of the data—a list of numbers that capture the "meaning" of the input.
*   This "Reference Embedding" is stored in memory.

### 2. Simulate "Production" Data (Current Data)
*   The app generates a *new* batch of data.
*   It applies the **Drift Generator** to this data based on the `severity` (0.0 to 1.0).
    *   **For Images (CNN)**: Adds random Gaussian noise.
    *   **For Text (BERT)**: Randomly swaps words with garbage tokens.
*   This "Drifted Data" is passed through the model to get "Current Embeddings".

### 3. The Comparison (Drift Detection)
*   The `DriftDetector` takes the **Reference Embeddings** and **Current Embeddings**.
*   It runs the **KS Test** on each feature dimension.
    *   *Null Hypothesis*: The feature values in Reference and Current come from the same distribution.
    *   *P-value < 0.05*: We reject the null hypothesis -> **Drift Detected!**
*   If more than 10% of the checked features show significant drift, the whole batch is flagged as drifted.

## Key Components

*   **`src/api/routes.py`**: The controller that orchestrates the flow.
*   **`src/models/`**: Contains the CNN and BERT model wrappers used to extract embeddings.
*   **`src/data/generator.py`**: The "Chaos Monkey" that injects noise to simulate drift.
*   **`src/drift/detector.py`**: The "Judge" that uses Scipy's `ks_2samp` to statistically validate drift.
