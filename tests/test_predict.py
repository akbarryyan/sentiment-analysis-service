from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "modelVersion" in payload


def test_predict_endpoint_returns_expected_shape() -> None:
    response = client.post(
        "/predict",
        json={
            "comment": "Materinya cukup jelas dan mudah dipahami",
            "aspect": "MATERI",
            "subject": "Agama",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["label"] in {"POSITIF", "NEGATIF", "NETRAL"}
    assert "confidence" in payload
    assert "preprocessedText" in payload
    assert "modelVersion" in payload
    assert "modelReady" in payload
    assert payload["autoMethod"] in {"LEXICON", "NAIVE_BAYES"}
