# API Reference

The API is built using **FastAPI**, which provides automatic interactive documentation at `/docs`.

## Endpoints

### Prediction

`POST /api/v1/predict/{model_type}`

Make a prediction using the specified model.

**Parameters**:
- `model_type` (path): `cnn` or `bert`.
- `simulate` (body): `true` to use synthetic data.

### Drift Detection

`POST /api/v1/detect-drift/{model_type}`

Analyze drift for the specified model.

**Parameters**:
- `model_type` (path): `cnn` or `bert`.
- `severity` (body): Float between 0.0 and 1.0 indicating the level of simulated drift.

### Cleanup

`POST /api/v1/cleanup`

Unload all models from memory and clear caches.
