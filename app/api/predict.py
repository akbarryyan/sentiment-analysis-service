from fastapi import APIRouter

from app.schemas import PredictRequest, PredictResponse
from app.services.inference import predict_sentiment

router = APIRouter(tags=["prediction"])


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    return predict_sentiment(payload)
