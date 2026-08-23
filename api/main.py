"""
main.py

FastAPI application entrypoint for HealthAudit-AI.

Run locally with:
    uvicorn api.main:app --reload

Then visit http://127.0.0.1:8000/docs for the interactive Swagger UI,
auto-generated from the Pydantic schemas in api/schemas.py.
"""

from fastapi import FastAPI

from api.routers import predict, recourse
from src.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="HealthAudit-AI",
    description="Explainable AI system for healthcare claim decision support.",
    version="0.1.0",
)

app.include_router(predict.router, tags=["Prediction"])
app.include_router(recourse.router, tags=["Recourse"])


@app.get("/health", tags=["System"])
def health_check() -> dict:
    """Basic liveness check -- useful once deployed, to confirm the service is up."""
    return {"status": "ok"}


@app.on_event("startup")
def on_startup() -> None:
    logger.info("HealthAudit-AI API starting up...")