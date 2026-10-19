
"""
main.py

FastAPI application entrypoint for HealthAudit-AI.

Run locally with:
    uvicorn api.main:app --reload

Then visit:
    http://127.0.0.1:8000/docs

for the interactive Swagger UI.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import predict, recourse, anchor
from src.utils.logger import get_logger


# -------------------------------------------------------------------
# Logger
# -------------------------------------------------------------------

logger = get_logger(__name__)


# -------------------------------------------------------------------
# FastAPI application
# -------------------------------------------------------------------

app = FastAPI(
    title="HealthAudit-AI",
    description=(
        "Explainable AI system for healthcare "
        "claim decision support."
    ),
    version="0.1.0",
)


# -------------------------------------------------------------------
# CORS
# -------------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://healthaudit-ai.vercel.app",
        "https://healthaudit-relwqjjt6-nandita-gaurs-projects.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Routers
# -------------------------------------------------------------------

app.include_router(
    predict.router,
    tags=["Prediction"],
)

app.include_router(
    recourse.router,
    tags=["Recourse"],
)

app.include_router(anchor.router)


# -------------------------------------------------------------------
# Health check
# -------------------------------------------------------------------

@app.get("/health", tags=["System"])
def health_check() -> dict:
    """Basic liveness check."""
    return {"status": "ok"}


# -------------------------------------------------------------------
# Startup
# -------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    """Log API startup."""
    logger.info("HealthAudit-AI API starting up...")
