# System Architecture

## Overview

The Model Drift Experiment system is designed as a modular microservice that exposes ML models and drift detection capabilities via a REST API.

## Components

### 1. API Layer (`src/api`)
- **Framework**: FastAPI.
- **Responsibility**: Handles HTTP requests, input validation (Pydantic), and routing.
- **Endpoints**:
    - `/predict`: Routes requests to the appropriate model for inference.
    - `/detect-drift`: Orchestrates the drift detection workflow.
    - `/cleanup`: Manages resource lifecycle.

### 2. Model Layer (`src/models`)
- **Design Pattern**: Strategy Pattern (via `BaseModel`).
- **Implementations**:
    - `CNNModel`: PyTorch-based CNN for image data.
    - `BERTModel`: HuggingFace Transformers wrapper for text.
- **Interface**:
    - `load()`: Lazy loading of weights.
    - `predict()`: Inference logic.
    - `get_embeddings()`: Feature extraction for drift analysis.

### 3. Drift Detection Layer (`src/drift`)
- **Algorithm**: Kolmogorov-Smirnov (KS) Test.
- **Logic**:
    1. Extract embeddings from **Reference Data** (clean/training distribution).
    2. Extract embeddings from **Current Data** (production/inference distribution).
    3. Compare distributions feature-wise.
    4. Aggregate p-values to determine if significant drift occurred.

### 4. Data Layer (`src/data`)
- **Loaders**: Generate synthetic data (MNIST-like arrays, random sentences) to ensure the project is self-contained.
- **Generators**: Inject specific types of noise (Gaussian for images, typos for text) to simulate drift with controllable severity.

## Data Flow

### Drift Detection Flow
1. **User** sends `POST /detect-drift/cnn` with `severity=0.5`.
2. **API** calls `DataLoader` to get reference images.
3. **API** calls `DataLoader` to get current images.
4. **API** calls `DriftGenerator` to apply noise (severity 0.5) to current images.
5. **Model** extracts embeddings for both sets.
6. **DriftDetector** compares embeddings and returns statistics.
7. **API** returns JSON response to user.

## Deployment

- **Containerization**: Dockerfile provided for building the image.
- **CI/CD**: GitHub Actions (implied) for running tests on push.
