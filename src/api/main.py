from fastapi import FastAPI

from ..config import config
from .routes import router

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
)

app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Model Drift Experiment API. Visit /docs for documentation."
    }


@app.on_event("startup")
async def startup_event():
    config.ensure_dirs()
