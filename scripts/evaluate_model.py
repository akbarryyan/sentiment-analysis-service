from pathlib import Path


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    print("Evaluation script placeholder.")
    print(f"Service directory: {base_dir}")
    print("Untuk saat ini, hasil evaluasi dasar disimpan di app/ml/metadata.json setelah training.")
    print("Langkah berikutnya: tambahkan confusion matrix dan visualisasi jika diperlukan.")


if __name__ == "__main__":
    main()
