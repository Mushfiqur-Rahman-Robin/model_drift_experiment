# src/api/routes.py
import logging
from typing import Any

import torch
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticBaseModel

from ..config import config
from ..data import DataLoader, DriftGenerator
from ..drift import DriftDetector
from ..models import BERTModel, CNNModel

router = APIRouter()

# Global state to hold loaded models and reference data
models = {"cnn": CNNModel(), "bert": BERTModel()}

# Get the logger instance for this module
logger = logging.getLogger(__name__)

# Store reference embeddings for drift detection
reference_data = {"cnn": None, "bert": None}


class PredictionRequest(PydanticBaseModel):
    data: list[Any] | None = None
    simulate: bool = True


class DriftRequest(PydanticBaseModel):
    severity: float = 0.5


@router.post("/predict/{model_type}")
async def predict(model_type: str, request: PredictionRequest):
    """
    Make predictions using the specified model.
    
    Parameters:
    - model_type: 'cnn' or 'bert'
    - request.simulate: true for synthetic data, false for custom data
    """
    logger.info(
        f"Received prediction request for model: {model_type}, simulate: {request.simulate}"
    )
    if model_type not in models:
        logger.warning(f"Prediction request for unknown model type: {model_type}")
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[model_type]

    if request.simulate:
        if model_type == "cnn":
            logger.debug("Generating synthetic CNN data for prediction.")
            data, _ = DataLoader.get_mnist_data(n_samples=10)
            prediction = model.predict(data)
            logger.info(f"CNN model predicted {len(prediction)} samples.")
        else:  # model_type == "bert"
            logger.debug("Generating synthetic BERT data for prediction.")
            data = DataLoader.get_text_data(n_samples=5)
            embeddings = model.predict(data)
            prediction = {"shape": embeddings.shape}
            logger.info(
                f"BERT model generated embeddings with shape {embeddings.shape}."
            )
    else:
        logger.error(f"Custom data input not fully implemented for model: {model_type}")
        raise HTTPException(
            status_code=501, detail="Custom data input not fully implemented for demo"
        )

    return {"model": model_type, "prediction": prediction}


@router.post("/detect-drift/{model_type}")
async def detect_drift(model_type: str, request: DriftRequest):
    """
    Run drift detection analysis.
    
    Parameters:
    - model_type: 'cnn' or 'bert'
    - request.severity: 0.0 to 1.0 (drift level)
    
    Process:
    1. Load reference data (clean)
    2. Generate current data (drifted based on severity)
    3. Compare using statistical KS test
    """
    logger.info(
        f"Received drift detection request for model: {model_type}, severity: {request.severity}"
    )
    if model_type not in models:
        logger.warning(f"Drift detection request for unknown model type: {model_type}")
        raise HTTPException(status_code=404, detail="Model not found")

    model = models[model_type]
    detector = DriftDetector(threshold=config.DRIFT_THRESHOLD)

    # 1. Get Reference Data (Clean)
    if reference_data[model_type] is None:
        logger.info(f"Loading reference data for {model_type} for the first time.")
        if model_type == "cnn":
            ref_images, _ = DataLoader.get_mnist_data(n_samples=50)
            reference_data[model_type] = model.get_embeddings(ref_images)
            logger.debug(
                f"CNN reference embeddings shape: {reference_data[model_type].shape}"
            )
        else:  # model_type == "bert"
            ref_texts = DataLoader.get_text_data(n_samples=20)
            reference_data[model_type] = model.get_embeddings(ref_texts)
            logger.debug(
                f"BERT reference embeddings shape: {reference_data[model_type].shape}"
            )
    else:
        logger.debug(f"Using cached reference data for {model_type}.")

    ref_embeddings = reference_data[model_type]

    # 2. Generate Current Data (Drifted)
    if model_type == "cnn":
        logger.debug(f"Generating current CNN data with severity {request.severity}.")
        curr_images, _ = DataLoader.get_mnist_data(n_samples=50)
        drifted_images = DriftGenerator.drift_images(
            curr_images, severity=request.severity
        )
        curr_embeddings = model.get_embeddings(drifted_images)
        logger.debug(f"CNN current embeddings shape: {curr_embeddings.shape}")
    else:  # model_type == "bert"
        logger.debug(f"Generating current BERT data with severity {request.severity}.")
        curr_texts = DataLoader.get_text_data(n_samples=20)
        drifted_texts = DriftGenerator.drift_text(curr_texts, severity=request.severity)
        curr_embeddings = model.get_embeddings(drifted_texts)
        logger.debug(f"BERT current embeddings shape: {curr_embeddings.shape}")

    # 3. Detect Drift
    result = detector.detect_drift(ref_embeddings, curr_embeddings)
    if result["drift_detected"]:
        logger.warning(
            f"Drift DETECTED for {model_type}! Ratio: {result['drift_ratio']:.2f}, Avg P-value: {result['avg_p_value']:.4f}"
        )
    else:
        logger.info(
            f"No significant drift detected for {model_type}. Ratio: {result['drift_ratio']:.2f}, Avg P-value: {result['avg_p_value']:.4f}"
        )

    return {
        "model": model_type,
        "drift_severity_simulated": request.severity,
        "analysis": result,
    }


@router.post("/cleanup")
async def cleanup():
    """
    Clear models from memory and caches.
    Useful for freeing resources between experiments.
    """
    logger.info("Received cleanup request. Clearing models and caches.")
    for name, model in models.items():
        model.clear()
        reference_data[name] = None
        logger.info(f"Cleared model '{name}' and its reference data.")

    import gc

    gc.collect()
    logger.debug("Garbage collection run.")

    if config.DEVICE == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("CUDA cache emptied.")

    return {"status": "cleaned", "message": "Models unloaded and cache cleared."}
