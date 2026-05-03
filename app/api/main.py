from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = BASE_DIR / "models" / "xgb_future_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "future_scaler.pkl"
FEATURES_PATH = BASE_DIR / "models" / "future_features.pkl"
DATA_PATH = BASE_DIR / "data" / "processed" / "air_weather_features.csv"


app = FastAPI(
    title="Air Quality Predictive API",
    description="API para predecir AQI futuro y generar alertas ambientales.",
    version="1.0.0",
)


class PredictionInput(BaseModel):
    station: str


def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"No existe el modelo: {MODEL_PATH}")
    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"No existe el scaler: {SCALER_PATH}")
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"No existe features: {FEATURES_PATH}")

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)

    return model, scaler, features


def load_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"No existe el dataset: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values(["station", "time"])

    return df


def get_alert_level(aqi: float) -> Dict[str, str]:
    if aqi <= 50:
        return {
            "level": "buena",
            "color": "green",
            "message": "Calidad del aire buena. No se detecta riesgo relevante.",
        }
    elif aqi <= 100:
        return {
            "level": "moderada",
            "color": "yellow",
            "message": "Calidad del aire moderada. Se recomienda seguimiento.",
        }
    else:
        return {
            "level": "mala",
            "color": "red",
            "message": "Alerta: calidad del aire mala. Se recomienda reducir exposición.",
        }


@app.get("/")
def root():
    return {
        "message": "Air Quality Predictive API funcionando correctamente",
        "endpoints": ["/health", "/stations", "/predict"],
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_exists": MODEL_PATH.exists(),
        "scaler_exists": SCALER_PATH.exists(),
        "features_exists": FEATURES_PATH.exists(),
        "data_exists": DATA_PATH.exists(),
    }


@app.get("/stations")
def stations():
    df = load_data()
    return {
        "stations": sorted(df["station"].unique().tolist())
    }


@app.post("/predict")
def predict(input_data: PredictionInput) -> Dict[str, Any]:
    try:
        model, scaler, features = load_artifacts()
        df = load_data()

        df_station = df[df["station"] == input_data.station].copy()

        if df_station.empty:
            raise HTTPException(
                status_code=404,
                detail=f"No hay datos para la estación: {input_data.station}",
            )

        latest = df_station.sort_values("time").iloc[-1]

        X_input = latest[features].to_frame().T
        X_scaled = scaler.transform(X_input)

        predicted_aqi = float(model.predict(X_scaled)[0])
        current_aqi = float(latest["aqi"])

        alert = get_alert_level(predicted_aqi)

        return {
            "station": input_data.station,
            "timestamp": str(latest["time"]),
            "current_aqi": round(current_aqi, 2),
            "predicted_aqi_next_hour": round(predicted_aqi, 2),
            "alert": alert,
            "input_summary": {
                "pm10": float(latest["pm10"]),
                "pm2_5": float(latest["pm2_5"]),
                "no2": float(latest["no2"]),
                "ozone": float(latest["ozone"]),
                "temperature": float(latest["temperature"]),
                "precipitation": float(latest["precipitation"]),
            },
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/predict_all")
def predict_all() -> Dict[str, Any]:
    try:
        model, scaler, features = load_artifacts()
        df = load_data()

        latest_by_station = (
            df.sort_values("time")
            .groupby("station")
            .tail(1)
            .reset_index(drop=True)
        )

        results = []

        for _, latest in latest_by_station.iterrows():
            X_input = latest[features].to_frame().T
            X_scaled = scaler.transform(X_input)

            predicted_aqi = float(model.predict(X_scaled)[0])
            current_aqi = float(latest["aqi"])
            alert = get_alert_level(predicted_aqi)

            results.append({
                "station": latest["station"],
                "timestamp": str(latest["time"]),
                "current_aqi": round(current_aqi, 2),
                "predicted_aqi_next_hour": round(predicted_aqi, 2),
                "alert": alert,
            })

        return {
            "total_stations": len(results),
            "predictions": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))