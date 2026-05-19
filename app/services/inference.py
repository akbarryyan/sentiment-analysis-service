from typing import Any

from app.config import get_settings
from app.schemas import PredictRequest, PredictResponse, SentimentLabel
from app.services.model_loader import is_model_ready, load_model
from app.services.preprocessing import preprocess_text


def _coerce_label(value: Any) -> SentimentLabel:
    normalized = str(value).strip().upper()

    if normalized == SentimentLabel.POSITIF.value:
        return SentimentLabel.POSITIF
    if normalized == SentimentLabel.NEGATIF.value:
        return SentimentLabel.NEGATIF

    return SentimentLabel.NETRAL


def predict_sentiment(payload: PredictRequest) -> PredictResponse:
    settings = get_settings()
    preprocessed_text = preprocess_text(payload.comment)
    model_ready = is_model_ready()

    if not model_ready:
        return PredictResponse(
            label=SentimentLabel.NETRAL,
            confidence=0.0,
            preprocessedText=preprocessed_text,
            modelVersion=settings.model_version,
            modelReady=False,
        )

    model = load_model()
    predicted_label = model.predict([preprocessed_text])[0]
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([preprocessed_text])[0]
        confidence = float(max(probabilities))

    return PredictResponse(
        label=_coerce_label(predicted_label),
        confidence=confidence,
        preprocessedText=preprocessed_text,
        modelVersion=settings.model_version,
        modelReady=True,
    )
