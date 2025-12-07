from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticBaseModel

from ..data import DataLoader, DriftGenerator
from ..drift import DriftDetector
from ..models import BERTModel, CNNModel

router = APIRouter()

# Global state to hold loaded models and reference data
# In a real app, this might be in a dependency or a singleton manager
models = {"cnn": CNNModel(), "bert": BERTModel()}

# Store reference embeddings for drift detection
reference_data = {"cnn": None, "bert": None}


class PredictionRequest(PydanticBaseModel):
    data: list[Any] | None = None  # For custom data
    simulate: bool = True  # If true, use synthetic data


class DriftRequest(PydanticBaseModel):
    severity: float = 0.5  # Drift severity


@router.post("/predict/{model_type}")
async def predict(model_type: str, request: PredictionRequest):
    """
    Make predictions using the specified model.
    """
    if model_type not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[model_type]

    if request.simulate:
        if model_type == "cnn":
            data, _ = DataLoader.get_mnist_data(n_samples=10)
            # Convert to list for JSON response (just a sample)
            prediction = model.predict(data)
        else:  # model_type == "bert"
            data = DataLoader.get_text_data(n_samples=5)
            # BERT wrapper returns embeddings as "prediction" for this demo
            embeddings = model.predict(data)
            prediction = {"shape": embeddings.shape}
    else:
        # Handle custom data input (omitted for brevity in this demo, assuming simulation focus)
        raise HTTPException(
            status_code=501, detail="Custom data input not fully implemented for demo"
        )

    return {"model": model_type, "prediction": prediction}


@router.post("/detect-drift/{model_type}")
async def detect_drift(model_type: str, request: DriftRequest):
    """
    Run drift detection analysis.
    1. Load reference data (clean).
    2. Generate current data (drifted based on severity).
    3. Compare using DriftDetector.
    """
    if model_type not in models:
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[model_type]
    detector = DriftDetector()

    # 1. Get Reference Data (Clean)
    if reference_data[model_type] is None:
        if model_type == "cnn":
            ref_images, _ = DataLoader.get_mnist_data(n_samples=50)
            reference_data[model_type] = model.get_embeddings(ref_images)
        else:  # model_type == "bert"
            ref_texts = DataLoader.get_text_data(n_samples=20)
            reference_data[model_type] = model.get_embeddings(ref_texts)

    ref_embeddings = reference_data[model_type]

    # 2. Generate Current Data (Drifted)
    if model_type == "cnn":
        curr_images, _ = DataLoader.get_mnist_data(n_samples=50)
        drifted_images = DriftGenerator.drift_images(
            curr_images, severity=request.severity
        )
        curr_embeddings = model.get_embeddings(drifted_images)
    else:  # model_type == "bert"
        curr_texts = DataLoader.get_text_data(n_samples=20)
        drifted_texts = DriftGenerator.drift_text(curr_texts, severity=request.severity)
        curr_embeddings = model.get_embeddings(drifted_texts)

    # 3. Detect Drift
    result = detector.detect_drift(ref_embeddings, curr_embeddings)

    return {
        "model": model_type,
        "drift_severity_simulated": request.severity,
        "analysis": result,
    }


@router.post("/cleanup")
async def cleanup():
    """
    Clear models from memory.
    """
    for name, model in models.items():
        model.clear()
        reference_data[name] = None

    # Force garbage collection if needed, but python usually handles it
    import gc

    gc.collect()

    from ..config import config

    if config.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "cleaned", "message": "Models unloaded and cache cleared."}
