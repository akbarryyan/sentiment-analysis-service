from typing import Any

from app.config import get_settings
from app.schemas import AutoMethod, PredictRequest, PredictResponse, SentimentLabel
from app.services.model_loader import is_model_ready, load_model
from app.services.preprocessing import preprocess_text

STRONG_NEGATIVE_PHRASES = {
    # Kehadiran & kedisiplinan guru
    "guru jarang masuk",
    "jarang masuk",
    "sering tidak masuk",
    "sering absen",
    "sering telat",
    "sering terlambat",
    "gurunya tidak hadir",
    "gurunya jarang hadir",
    # Sikap & perilaku guru
    "tidak mau menjelaskan ulang",
    "tidak sabar menjelaskan",
    "guru tidak menjelaskan",
    "gurunya tidak sabar",
    "guru tidak menjawab",
    "tidak peduli",
    # Kejelasan materi / penyampaian
    "kurang jelas",
    "tidak jelas",
    "kurang dipahami",
    "kurang dimengerti",
    "kurang paham",
    "tidak dipahami",
    "tidak dimengerti",
    "tidak mudah dipahami",
    "tidak mudah dimengerti",
    "belum mudah dipahami",
    "belum mudah dimengerti",
    "sulit dipahami",
    "susah dipahami",
    "sulit dimengerti",
    "susah dimengerti",
    "susah dipelajari",
    "sulit dipelajari",
    "masih membingungkan",
    "sangat membingungkan",
    "bikin bingung",
    "membuat bingung",
    "tidak ada contoh",
    "contoh tidak jelas",
    "contoh kurang",
    "penjelasan terlalu singkat",
    "penjelasan tidak detail",
    "tidak detail",
    # Kualitas penyampaian
    "tidak menarik",
    "kurang menarik",
    "tidak membantu",
    "kurang membantu",
    "kurang lengkap",
    "tidak lengkap",
    "kurang terstruktur",
    "tidak terstruktur",
    "terlalu cepat",
    "terlalu cepat dijelaskan",
    "terlalu lambat",
    "terlalu singkat",
    "terlalu monoton",
    "kurang interaktif",
    "tidak interaktif",
    "membosankan",
    "sangat membosankan",
    "kurang bervariasi",
    "tidak bervariasi",
    # Waktu
    "waktu tidak cukup",
    "waktu kurang",
    "waktunya kurang",
    # Soal / evaluasi
    "soal terlalu sulit",
    "soal susah",
    "soal tidak sesuai",
    "soal tidak sesuai pelajaran",
    "soal tidak relevan",
    "soal membingungkan",
    "soal tidak jelas",
    "soal terlalu banyak",
    "soal tidak adil",
    "kisi-kisi tidak sesuai",
    "tidak ada kisi-kisi",
    "tidak sesuai materi",
    "tidak sesuai pelajaran",
    "tidak relevan",
    "kurang sesuai",
    # Efektivitas
    "tidak efektif",
    "kurang efektif",
    "buang-buang waktu",
    "tidak ada manfaat",
    "kurang bermanfaat",
    "tidak bermanfaat",
    # Ketidakpahaman total
    "tidak paham sama sekali",
    "sama sekali tidak mengerti",
    "benar-benar tidak paham",
    "benar-benar tidak mengerti",
    "tidak mengerti sama sekali",
    # Permintaan perbaikan
    "perlu dijelaskan lebih perlahan",
    "perlu diperjelas",
    "perlu lebih jelas",
    "perlu lebih interaktif",
    "perlu diperbaiki",
    "harus diperbaiki",
    "perlu ditingkatkan",
    "perlu lebih lengkap",
    # Negatif umum
    "tidak bagus",
    "kurang bagus",
    "tidak baik",
    "kurang baik",
    "sangat buruk",
    "tidak memuaskan",
    "kurang memuaskan",
    "sangat mengecewakan",
    "mengecewakan",
}

# Kata yang jika muncul tepat sebelum frasa positif membalik maknanya
_POSITIVE_PHRASE_NEGATORS = {"tidak", "tak", "belum", "bukan", "kurang", "sulit", "susah"}

STRONG_POSITIVE_PHRASES = {
    # Kejelasan & kemudahan
    "mudah dipahami",
    "mudah dimengerti",
    "mudah dipelajari",
    "mudah diikuti",
    "sangat jelas",
    "sudah jelas",
    "cukup jelas",
    "sangat mudah dipahami",
    "sangat mudah dimengerti",
    # Hasil belajar
    "mudah diingat",
    "langsung dipahami",
    "langsung dimengerti",
    "mudah diterapkan",
    "sangat paham",
    "jadi lebih paham",
    "lebih mudah dipahami",
    # Manfaat & kualitas
    "sangat membantu",
    "sangat bermanfaat",
    "sangat berguna",
    "sangat relevan",
    "sesuai materi",
    "sesuai kurikulum",
    "sesuai pelajaran",
    "sangat sesuai",
    # Contoh & latihan
    "contoh sangat jelas",
    "banyak contoh",
    "contoh relevan",
    "latihan sangat berguna",
    "latihan sangat membantu",
    # Penyampaian
    "menyenangkan",
    "sangat menyenangkan",
    "sangat menarik",
    "sangat interaktif",
    "sangat variatif",
    "enak dijelaskan",
    "enak dipelajari",
    "asik dipelajari",
    "bikin semangat belajar",
    "membuat termotivasi",
    "sangat antusias",
    # Soal positif
    "soal sesuai pelajaran",
    "soal sesuai materi",
    "soal sangat jelas",
    "soal sangat adil",
    "soal sangat bagus",
    # Penilaian umum
    "sangat baik",
    "sangat bagus",
    "luar biasa",
    "sangat memuaskan",
    "sudah bagus",
    "sudah baik",
    "cukup baik",
    "cukup bagus",
    "sangat lengkap",
    "sudah lengkap",
    "sangat terstruktur",
    # Apresiasi guru
    "guru sangat baik",
    "guru sangat jelas",
    "guru sangat sabar",
    "guru sangat kompeten",
    "guru sangat profesional",
    "gurunya sabar",
    "gurunya menyenangkan",
    "gurunya baik",
    "gurunya bagus",
    "gurunya rajin",
    "gurunya kompeten",
    "selalu tepat waktu",
    "penjelasan guru sangat jelas",
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

# Kata yang membalik makna token positif di belakangnya
# Contoh: "kurang paham" → negatif, "perlu jelas" → negatif
NEGATION_MODIFIERS = {
    "kurang",
    "tidak",
    "tak",
    "belum",
    "bukan",
    "sulit",
    "susah",
    "perlu",
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

    for phrase in STRONG_POSITIVE_PHRASES:
        idx = lowered_text.find(phrase)
        if idx == -1:
            continue
        prefix_words = lowered_text[:idx].split()
        if prefix_words and prefix_words[-1] in _POSITIVE_PHRASE_NEGATORS:
            continue
        return SentimentLabel.POSITIF, 0.92

    tokens = [token for token in preprocessed_text.split(" ") if token]

    if not tokens:
        return SentimentLabel.NETRAL, 0.0

    positive_hits = 0
    negative_hits = 0

    for i, token in enumerate(tokens):
        prev_token = tokens[i - 1] if i > 0 else None
        negated = prev_token in NEGATION_MODIFIERS

        if token in POSITIVE_TERMS:
            # "kurang paham", "perlu jelas" → makna negatif
            if negated:
                negative_hits += 1
            else:
                positive_hits += 1
        elif token in NEGATIVE_TERMS:
            negative_hits += 1

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
