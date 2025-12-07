# Welcome to Model Drift Experiment

This documentation provides a deep dive into the **Model Drift Experiment** codebase, a project designed to demonstrate advanced concepts in machine learning monitoring and engineering.

## Project Goals

1.  **Educational Resource**: Serve as a reference for building production-grade ML systems.
2.  **Drift Detection**: Demonstrate how to detect concept drift in both image and text data.
3.  **Modern Engineering**: Showcase best practices using FastAPI, Docker, and CI/CD.

## Quick Start

To get started with the project, you can run the following command:

```bash
docker compose up --build
```

This will start the API server at `http://localhost:8000`.

## Key Features

- **Multi-Modal**: Supports both Computer Vision (CNN) and NLP (BERT) models.
- **Synthetic Data**: Includes generators for creating synthetic datasets and simulating drift.
- **Statistical Monitoring**: Uses the Kolmogorov-Smirnov test for robust drift detection.
