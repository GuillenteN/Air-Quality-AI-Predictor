from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ok"
    assert data["model_exists"] is True
    assert data["scaler_exists"] is True
    assert data["features_exists"] is True
    assert data["data_exists"] is True


def test_stations_endpoint():
    response = client.get("/stations")

    assert response.status_code == 200

    data = response.json()

    assert "stations" in data
    assert isinstance(data["stations"], list)
    assert len(data["stations"]) > 0


def test_predict_endpoint():
    response = client.post(
        "/predict",
        json={"station": "Burgos"}
    )

    assert response.status_code == 200

    data = response.json()

    assert data["station"] == "Burgos"
    assert "current_aqi" in data
    assert "predicted_aqi_next_hour" in data
    assert "alert" in data

    assert isinstance(data["predicted_aqi_next_hour"], float)
    assert data["alert"]["level"] in ["buena", "moderada", "mala"]


def test_predict_all_endpoint():
    response = client.get("/predict_all")

    assert response.status_code == 200

    data = response.json()

    assert "total_stations" in data
    assert "predictions" in data
    assert data["total_stations"] > 0
    assert isinstance(data["predictions"], list)

    first_prediction = data["predictions"][0]

    assert "station" in first_prediction
    assert "current_aqi" in first_prediction
    assert "predicted_aqi_next_hour" in first_prediction
    assert "alert" in first_prediction