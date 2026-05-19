import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.config import get_settings
from app.services.preprocessing import preprocess_text

ALLOWED_LABELS = {"POSITIF", "NEGATIF", "NETRAL"}
REQUIRED_COLUMNS = {"comment", "aspect", "subject", "label"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train model Naive Bayes untuk analisis sentimen feedback siswa.",
    )
    parser.add_argument(
        "--input",
        default="data/raw/feedback_sentiment_labeled.csv",
        help="Path dataset CSV berlabel.",
    )
    parser.add_argument(
        "--output",
        default="app/ml/pipeline.joblib",
        help="Path output model joblib.",
    )
    parser.add_argument(
        "--metadata-output",
        default="app/ml/metadata.json",
        help="Path output metadata training.",
    )
    parser.add_argument(
        "--processed-output",
        default="data/processed/feedback_sentiment_preprocessed.csv",
        help="Path output dataset hasil preprocessing.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data uji jika dataset cukup untuk split.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state untuk split data.",
    )
    return parser.parse_args()


def load_and_validate_dataset(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di '{csv_path}'. Siapkan CSV training terlebih dahulu.",
        )

    df = pd.read_csv(csv_path)
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset belum memiliki kolom wajib: {missing_text}.")

    normalized = df.copy()
    normalized["comment"] = normalized["comment"].fillna("").astype(str).str.strip()
    normalized["aspect"] = normalized["aspect"].fillna("").astype(str).str.strip().str.upper()
    normalized["subject"] = normalized["subject"].fillna("").astype(str).str.strip()
    normalized["label"] = normalized["label"].fillna("").astype(str).str.strip().str.upper()

    normalized = normalized[normalized["comment"] != ""].copy()

    invalid_labels = sorted(set(normalized["label"]) - ALLOWED_LABELS)
    if invalid_labels:
        invalid_text = ", ".join(invalid_labels)
        raise ValueError(
            f"Dataset mengandung label tidak valid: {invalid_text}. "
            "Gunakan hanya POSITIF, NEGATIF, atau NETRAL.",
        )

    if normalized.empty:
        raise ValueError("Dataset kosong setelah pembersihan komentar.")

    return normalized


def can_use_stratified_split(df: pd.DataFrame, test_size: float) -> bool:
    label_counts = df["label"].value_counts()
    minimum_required = max(2, int(round(1 / max(test_size, 0.01))))

    return len(df) >= 9 and (label_counts >= 2).all() and label_counts.min() >= minimum_required


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            ("classifier", MultinomialNB()),
        ],
    )


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    settings = get_settings()

    dataset_path = Path(args.input)
    model_output_path = Path(args.output)
    metadata_output_path = Path(args.metadata_output)
    processed_output_path = Path(args.processed_output)

    dataset = load_and_validate_dataset(dataset_path)
    dataset["preprocessed_text"] = dataset["comment"].apply(preprocess_text)
    dataset = dataset[dataset["preprocessed_text"] != ""].copy()

    if dataset.empty:
        raise ValueError("Semua teks kosong setelah preprocessing. Dataset perlu diperiksa.")

    ensure_parent_dir(processed_output_path)
    dataset.to_csv(processed_output_path, index=False)

    x = dataset["preprocessed_text"]
    y = dataset["label"]

    pipeline = build_pipeline()
    evaluation: dict[str, Any]

    if can_use_stratified_split(dataset, args.test_size):
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        pipeline.fit(x_train, y_train)
        y_pred = pipeline.predict(x_test)
        accuracy = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        evaluation = {
            "mode": "train_test_split",
            "testSize": args.test_size,
            "accuracy": accuracy,
            "classificationReport": report,
            "trainRows": int(len(x_train)),
            "testRows": int(len(x_test)),
        }
    else:
        pipeline.fit(x, y)
        evaluation = {
            "mode": "train_only",
            "testSize": 0.0,
            "accuracy": None,
            "classificationReport": None,
            "trainRows": int(len(x)),
            "testRows": 0,
            "note": (
                "Dataset belum cukup untuk train/test split terstratifikasi. "
                "Tambahkan data per label agar evaluasi lebih valid."
            ),
        }

    ensure_parent_dir(model_output_path)
    joblib.dump(pipeline, model_output_path)

    class_distribution = {
        label: int(count)
        for label, count in dataset["label"].value_counts().sort_index().items()
    }

    metadata = {
        "modelVersion": settings.model_version,
        "algorithm": "MultinomialNB",
        "vectorizer": "TfidfVectorizer",
        "datasetPath": str(dataset_path),
        "processedDatasetPath": str(processed_output_path),
        "modelPath": str(model_output_path),
        "rows": int(len(dataset)),
        "classDistribution": class_distribution,
        "evaluation": evaluation,
    }

    ensure_parent_dir(metadata_output_path)
    metadata_output_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print("Training selesai.")
    print(f"Dataset        : {dataset_path}")
    print(f"Baris terpakai : {len(dataset)}")
    print(f"Model output   : {model_output_path}")
    print(f"Metadata       : {metadata_output_path}")
    print(f"Mode evaluasi  : {evaluation['mode']}")
    if evaluation["accuracy"] is not None:
        print(f"Akurasi        : {evaluation['accuracy']:.4f}")
    else:
        print("Akurasi        : belum tersedia")


if __name__ == "__main__":
    main()
