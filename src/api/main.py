# src/api/main.py
import logging
import time
from uuid import uuid4

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from ..config import config, request_id_ctx, setup_logging
from .routes import router

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
)

logger = logging.getLogger(__name__)


@app.middleware("http")
async def add_request_id_and_log_requests(request: Request, call_next):
    request_id = str(uuid4())
    token = request_id_ctx.set(request_id)

    start_time = time.monotonic()
    response = None
    status_code = 500

    logger.info(f"Request started: {request.method} {request.url.path}")
    try:
        response = await call_next(request)
        status_code = response.status_code
    except HTTPException as http_exc:
        response = JSONResponse(
            status_code=http_exc.status_code, content={"detail": http_exc.detail}
        )
        status_code = http_exc.status_code
        logger.warning(
            f"Handled HTTPException: {request.method} {request.url.path} - Status: {status_code} - Detail: {http_exc.detail}"
        )
    except Exception as e:
        response = JSONResponse(
            status_code=500, content={"detail": "Internal Server Error"}
        )
        status_code = 500
        logger.error(
            f"Unhandled exception during request: {request.method} {request.url.path} with error: {e}",
            exc_info=True,
        )
    finally:
        process_time = time.monotonic() - start_time
        if response:
            response.headers["X-Request-ID"] = request_id
            logger.info(
                f"Request finished: {request.method} {request.url.path} - Status: {status_code} - Time: {process_time:.4f}s"
            )
        else:
            logger.critical(
                f"Middleware finished processing but 'response' object was None for {request.method} {request.url.path}. Time: {process_time:.4f}s"
            )

        request_id_ctx.reset(token)

    if response is None:
        logger.critical(
            f"Middleware returning a fallback 500 response. Original response was None for {request.method} {request.url.path}."
        )
        return JSONResponse(
            status_code=500, content={"detail": "Unhandled Middleware Error Fallback"}
        )
    return response


app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {
        "message": "Welcome to the Model Drift Experiment API. Visit /docs for documentation."
    }


@app.on_event("startup")
async def startup_event():
    """Startup event: ensure directories and set up logging."""
    config.ensure_dirs()
    setup_logging()
    logger.info("Application startup complete.")
