import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.services.preprocessing import preprocess_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluasi model Naive Bayes dan tampilkan confusion matrix.",
    )
    parser.add_argument(
        "--input",
        default="data/processed/feedback_sentiment_preprocessed.csv",
        help="Path dataset CSV hasil preprocessing.",
    )
    parser.add_argument(
        "--model",
        default="app/ml/pipeline.joblib",
        help="Path model pipeline joblib.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proporsi data uji (samakan dengan saat training).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random state untuk split (samakan dengan saat training).",
    )
    parser.add_argument(
        "--output",
        default="app/ml/evaluation.json",
        help="Path output JSON hasil evaluasi lengkap.",
    )
    return parser.parse_args()


def _print_confusion_matrix(cm: list[list[int]], labels: list[str]) -> None:
    col_width = max(len(label) for label in labels) + 2
    header = " " * (col_width + 3) + "".join(label.rjust(col_width) for label in labels)
    separator = " " * (col_width + 3) + "-" * (col_width * len(labels))
    print(header)
    print(separator)
    for i, actual in enumerate(labels):
        row = actual.rjust(col_width) + "  |"
        for j in range(len(labels)):
            row += str(cm[i][j]).rjust(col_width)
        print(row)


def _can_split(y: pd.Series, test_size: float) -> bool:
    label_counts = y.value_counts()
    min_required = max(2, int(round(1 / max(test_size, 0.01))))
    return len(y) >= 9 and (label_counts >= 2).all() and int(label_counts.min()) >= min_required


def main() -> None:
    args = parse_args()
    dataset_path = Path(args.input)
    model_path = Path(args.model)
    output_path = Path(args.output)

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: '{dataset_path}'. "
            "Jalankan train_model.py terlebih dahulu agar file preprocessed tersedia.",
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model tidak ditemukan: '{model_path}'. "
            "Jalankan train_model.py terlebih dahulu.",
        )

    df = pd.read_csv(dataset_path)

    if "preprocessed_text" in df.columns:
        df = df[df["preprocessed_text"].notna() & (df["preprocessed_text"].str.strip() != "")].copy()
        x = df["preprocessed_text"].astype(str)
    else:
        df["preprocessed_text"] = df["comment"].fillna("").astype(str).apply(preprocess_text)
        df = df[df["preprocessed_text"].str.strip() != ""].copy()
        x = df["preprocessed_text"]

    y = df["label"].astype(str).str.strip().str.upper()

    if _can_split(y, args.test_size):
        _, x_eval, _, y_eval = train_test_split(
            x,
            y,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y,
        )
        eval_mode = "test_split"
    else:
        x_eval, y_eval = x, y
        eval_mode = "full_dataset"
        print(
            "Peringatan: dataset belum cukup untuk train/test split. "
            "Evaluasi dilakukan pada seluruh data.\n",
        )

    model = joblib.load(model_path)
    y_pred = model.predict(x_eval)

    labels = sorted(set(y_eval.tolist()) | set(y_pred.tolist()))
    cm = confusion_matrix(y_eval, y_pred, labels=labels)
    cm_list: list[list[int]] = cm.tolist()
    accuracy = float(accuracy_score(y_eval, y_pred))
    report_str = classification_report(y_eval, y_pred, labels=labels, zero_division=0)
    report_dict = classification_report(y_eval, y_pred, labels=labels, output_dict=True, zero_division=0)

    print("\n" + "=" * 50)
    print("EVALUASI MODEL ANALISIS SENTIMEN")
    print("=" * 50)
    print(f"\nModel     : {model_path}")
    print(f"Dataset   : {dataset_path}")
    print(f"Mode      : {eval_mode}")
    print(f"Data eval : {len(y_eval)} baris")
    print(f"Accuracy  : {accuracy:.4f}  ({accuracy * 100:.2f}%)")

    print("\n--- Confusion Matrix ---")
    print("(Baris = Label Aktual, Kolom = Label Prediksi)\n")
    _print_confusion_matrix(cm_list, labels)

    print("\n--- Per-Kelas (TP / FP / FN) ---")
    header_tp = f"{'Label':<10}  {'TP':>6}  {'FP':>6}  {'FN':>6}"
    print(header_tp)
    print("-" * len(header_tp))
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum()) - tp
        fn = int(cm[i, :].sum()) - tp
        print(f"{label:<10}  {tp:>6}  {fp:>6}  {fn:>6}")

    print("\n--- Classification Report ---")
    print(report_str)

    result = {
        "modelPath": str(model_path),
        "datasetPath": str(dataset_path),
        "evalMode": eval_mode,
        "evalRows": int(len(y_eval)),
        "accuracy": accuracy,
        "labels": labels,
        "confusionMatrix": cm_list,
        "classificationReport": report_dict,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"Hasil evaluasi disimpan di: {output_path}")


if __name__ == "__main__":
    main()
