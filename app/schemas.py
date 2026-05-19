from enum import Enum

from pydantic import BaseModel, Field


class LearningAspect(str, Enum):
    MATERI = "MATERI"
    PENYAMPAIAN = "PENYAMPAIAN"
    SOAL = "SOAL"


class SentimentLabel(str, Enum):
    POSITIF = "POSITIF"
    NEGATIF = "NEGATIF"
    NETRAL = "NETRAL"


class PredictRequest(BaseModel):
    comment: str = Field(min_length=3, max_length=5000)
    aspect: LearningAspect
    subject: str | None = Field(default=None, max_length=255)


class PredictResponse(BaseModel):
    label: SentimentLabel
    confidence: float
    preprocessedText: str
    modelVersion: str
    modelReady: bool


class HealthResponse(BaseModel):
    status: str
    modelVersion: str
    modelReady: bool
