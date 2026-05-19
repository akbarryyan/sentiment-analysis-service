from fastapi import FastAPI

from app.api.predict import router as predict_router
from app.config import get_settings
from app.schemas import HealthResponse
from app.services.model_loader import is_model_ready

settings = get_settings()

app = FastAPI(
    title="Sentiment Analysis Service",
    version="0.1.0",
    description="API analisis sentimen feedback siswa berbasis FastAPI.",
)

app.include_router(predict_router)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health_check() -> HealthResponse:
    return HealthResponse(
        status="ok",
        modelVersion=settings.model_version,
        modelReady=is_model_ready(),
    )
