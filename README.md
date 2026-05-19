# Sentiment Analysis Service

Service ini menangani analisis sentimen feedback siswa secara terpisah dari aplikasi Next.js.

## Stack

- Python
- FastAPI
- scikit-learn
- Sastrawi

## Struktur

```text
sentiment-analysis-service/
├── app/
│   ├── api/
│   ├── ml/
│   ├── services/
│   └── utils/
├── data/
├── notebooks/
├── scripts/
└── tests/
```

## Setup

```bash
cd sentiment-analysis-service
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Dataset Training

Format dataset awal ada di:

`data/raw/feedback_sentiment_labeled.csv`

Kolom wajib:

- `comment`
- `aspect`
- `subject`
- `label`

Nilai label yang valid:

- `POSITIF`
- `NEGATIF`
- `NETRAL`

Contoh:

```csv
comment,aspect,subject,label
"Materinya jelas dan mudah dipahami",MATERI,Agama,POSITIF
"Soalnya terlalu sulit",SOAL,Agama,NEGATIF
"Penyampaiannya biasa saja",PENYAMPAIAN,Agama,NETRAL
```

Catatan:

- Dataset contoh yang disertakan sekarang masih sangat kecil dan hanya untuk bootstrap awal.
- Untuk hasil TA yang layak, kamu perlu menambah data berlabel dalam jumlah jauh lebih besar dan lebih seimbang.

## Training Model

Jalankan training:

```bash
python scripts/train_model.py
```

Output yang dihasilkan:

- `app/ml/pipeline.joblib`
- `app/ml/metadata.json`
- `data/processed/feedback_sentiment_preprocessed.csv`

Jika dataset cukup besar dan seimbang, script akan otomatis memakai `train_test_split` untuk evaluasi akurasi.
Kalau dataset masih kecil, model tetap dilatih tetapi metadata akan menandai bahwa evaluasi belum valid.

## Endpoint

### `GET /health`

Memeriksa status service dan metadata model aktif.

### `POST /predict`

Menerima komentar feedback siswa dan mengembalikan hasil prediksi sentimen.

Contoh request:

```json
{
  "comment": "Materinya cukup jelas dan mudah dipahami",
  "aspect": "MATERI",
  "subject": "Agama"
}
```

Contoh response saat model belum tersedia:

```json
{
  "label": "NETRAL",
  "confidence": 0.0,
  "preprocessedText": "materi cukup jelas mudah paham",
  "modelVersion": "nb-v1",
  "modelReady": false
}
```

## Catatan

- Saat ini scaffold sudah siap dijalankan.
- Endpoint `predict` akan memakai fallback jika file model belum tersedia.
- Setelah `pipeline.joblib` terbentuk, endpoint `predict` akan otomatis memakai model tersebut.
- Langkah berikutnya yang disarankan adalah memperbesar dataset, memvalidasi kualitas label, lalu integrasi ke Next.js.
