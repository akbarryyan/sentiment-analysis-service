from typing import Any

from app.config import get_settings
from app.schemas import AutoMethod, PredictRequest, PredictResponse, SentimentLabel
from app.services.model_loader import is_model_ready, load_model
from app.services.preprocessing import preprocess_text

STRONG_NEGATIVE_PHRASES = {
    "guru jarang masuk",
    "jarang masuk",
    "sering tidak masuk",
    "sering telat",
    "sering terlambat",
    "kurang jelas",
    "sulit dipahami",
    "tidak jelas",
    "tidak membantu",
    "membosankan",
}

STRONG_POSITIVE_PHRASES = {
    "mudah dipahami",
    "sangat jelas",
    "sangat membantu",
    "menyenangkan",
    "sangat baik",
}

POSITIVE_TERMS = {
    "bagus",
    "baik",
    "jelas",
    "mudah",
    "paham",
    "rapi",
    "sesuai",
    "semangat",
    "bantu",
    "mantap",
    "menarik",
    "lengkap",
    "nyaman",
    "variatif",
    "tantang",
}

NEGATIVE_TERMS = {
    "buruk",
    "bingung",
    "kurang",
    "sulit",
    "bosan",
    "jelek",
    "rumit",
    "salah",
    "lambat",
    "berat",
    "bingungkan",
    "tidak",
    "membingungkan",
    "susah",
    "jarang",
    "telat",
    "terlambat",
    "membosan",
}


def _coerce_label(value: Any) -> SentimentLabel:
    normalized = str(value).strip().upper()

    if normalized == SentimentLabel.POSITIF.value:
        return SentimentLabel.POSITIF
    if normalized == SentimentLabel.NEGATIF.value:
        return SentimentLabel.NEGATIF

    return SentimentLabel.NETRAL


def _predict_with_lexicon(raw_text: str, preprocessed_text: str) -> tuple[SentimentLabel, float]:
    lowered_text = raw_text.casefold().strip()

    if any(phrase in lowered_text for phrase in STRONG_NEGATIVE_PHRASES):
        return SentimentLabel.NEGATIF, 0.92

    if any(phrase in lowered_text for phrase in STRONG_POSITIVE_PHRASES):
        return SentimentLabel.POSITIF, 0.92

    tokens = [token for token in preprocessed_text.split(" ") if token]

    if not tokens:
        return SentimentLabel.NETRAL, 0.0

    positive_hits = sum(1 for token in tokens if token in POSITIVE_TERMS)
    negative_hits = sum(1 for token in tokens if token in NEGATIVE_TERMS)
    total_hits = positive_hits + negative_hits

    if total_hits == 0 or positive_hits == negative_hits:
        return SentimentLabel.NETRAL, 0.0

    confidence = float(total_hits / max(len(tokens), 1))
    confidence = min(confidence, 1.0)

    if positive_hits > negative_hits:
        return SentimentLabel.POSITIF, confidence

    return SentimentLabel.NEGATIF, confidence


def _should_override_model(
    model_label: SentimentLabel,
    model_confidence: float,
    lexicon_label: SentimentLabel,
    lexicon_confidence: float,
) -> bool:
    if lexicon_label == SentimentLabel.NETRAL:
        return False

    if model_label == lexicon_label:
        return False

    if lexicon_confidence >= 0.9:
        return True

    return model_confidence < 0.65


def predict_sentiment(payload: PredictRequest) -> PredictResponse:
    settings = get_settings()
    preprocessed_text = preprocess_text(payload.comment)
    model_ready = is_model_ready()
    lexicon_label, lexicon_confidence = _predict_with_lexicon(
        payload.comment,
        preprocessed_text,
    )

    if not model_ready:
        return PredictResponse(
            label=lexicon_label,
            confidence=lexicon_confidence,
            preprocessedText=preprocessed_text,
            modelVersion=settings.model_version,
            modelReady=False,
            autoMethod=AutoMethod.LEXICON,
        )

    model = load_model()
    predicted_label = model.predict([preprocessed_text])[0]
    confidence = 0.0

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba([preprocessed_text])[0]
        confidence = float(max(probabilities))

    model_label = _coerce_label(predicted_label)

    if _should_override_model(
        model_label=model_label,
        model_confidence=confidence,
        lexicon_label=lexicon_label,
        lexicon_confidence=lexicon_confidence,
    ):
        return PredictResponse(
            label=lexicon_label,
            confidence=lexicon_confidence,
            preprocessedText=preprocessed_text,
            modelVersion=settings.model_version,
            modelReady=True,
            autoMethod=AutoMethod.LEXICON,
        )

    return PredictResponse(
        label=model_label,
        confidence=confidence,
        preprocessedText=preprocessed_text,
        modelVersion=settings.model_version,
        modelReady=True,
        autoMethod=AutoMethod.NAIVE_BAYES,
    )
