# Model Drift Experiment

A comprehensive codebase demonstrating model drift detection for Computer Vision (CNN) and Natural Language Processing (BERT) models. This project is designed for advanced learning, featuring a modular architecture, synthetic data generation, and a FastAPI-based service.

## Features

- **Multi-Modal Support**:
    - **CNN**: A simple Convolutional Neural Network for image classification (simulated MNIST).
    - **BERT**: A wrapper around HuggingFace's BERT for text analysis.
- **Drift Detection**:
    - Statistical monitoring of feature embeddings using the Kolmogorov-Smirnov (KS) test.
    - Detects distribution shifts in both image and text data.
- **Synthetic Data & Drift**:
    - Built-in data generators to simulate production environments.
    - Injectable drift (Gaussian noise for images, typos/vocab shifts for text).
- **Modern Stack**:
    - **FastAPI**: High-performance API.
    - **PyTorch & Transformers**: State-of-the-art ML libraries.
    - **Docker Ready**: Containerization support (see Dockerfile).

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd model-drift-experiment
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   # Or manually:
   pip install fastapi uvicorn torch transformers scikit-learn numpy pandas pillow httpx scipy
   ```

## Usage

### Running the API

Start the FastAPI server:
```bash
uvicorn src.api.main:app --reload
```

The API will be available at `http://localhost:8000`.
Documentation (Swagger UI) is at `http://localhost:8000/docs`.

### API Endpoints

- **`POST /api/v1/predict/{model_type}`**: Make predictions.
    - `model_type`: `cnn` or `bert`.
    - Body: `{"simulate": true}` (uses synthetic data).

- **`POST /api/v1/detect-drift/{model_type}`**: Analyze drift.
    - Body: `{"severity": 0.5}` (0.0 to 1.0).
    - Returns drift statistics and whether drift was detected.

- **`POST /api/v1/cleanup`**: Clear loaded models from memory.

## Testing

Run the test suite:
```bash
python -m pytest tests/
```

## Project Structure

```
model-drift-experiment/
├── src/
│   ├── api/            # FastAPI application
│   ├── data/           # Data loaders and generators
│   ├── drift/          # Drift detection logic
│   ├── models/         # Model implementations (CNN, BERT)
│   └── config.py       # Configuration
├── tests/              # Unit and integration tests
├── requirements.txt    # Dependencies
└── README.md           # This file
```

## Advanced Learning

This codebase demonstrates:
1. **Abstract Base Classes**: Using `abc` for consistent model interfaces.
2. **Dependency Injection**: Passing models to the API.
3. **Statistical Testing**: Using KS-test for drift detection on high-dimensional embeddings.
4. **Synthetic Simulation**: Generating data on-the-fly to test robustness.

## License

MIT
