from fastapi import HTTPException

from app.config import get_settings
from app.schemas import AutoMethod, PredictRequest, PredictResponse, SentimentLabel
from app.services.model_loader import is_model_ready, load_model
from app.services.preprocessing import preprocess_text


def _coerce_label(value: object) -> SentimentLabel:
    normalized = str(value).strip().upper()

    if normalized == SentimentLabel.POSITIF.value:
        return SentimentLabel.POSITIF
    if normalized == SentimentLabel.NEGATIF.value:
        return SentimentLabel.NEGATIF

    return SentimentLabel.NETRAL


def predict_sentiment(payload: PredictRequest) -> PredictResponse:
    settings = get_settings()

    if not is_model_ready():
        raise HTTPException(
            status_code=503,
            detail="Model belum siap. Jalankan train_model.py terlebih dahulu.",
        )

    preprocessed_text = preprocess_text(payload.comment)
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
        autoMethod=AutoMethod.NAIVE_BAYES,
    )
