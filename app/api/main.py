from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="API Predicción NO2")

model = joblib.load("models/xgboost_no2.pkl")
features = joblib.load("models/features.pkl")

class PredictionInput(BaseModel):
    no2: float
    hour: int
    day_of_week: int
    month: int
    is_weekend: int
    hour_sin: float
    hour_cos: float
    no2_lag_1: float
    no2_lag_24: float

@app.get("/")
def home():
    return {"message": "API de predicción de NO2 funcionando"}

@app.post("/predict")
def predict(data: PredictionInput):
    input_df = pd.DataFrame([data.dict()])

    print("INPUT COLUMNS:", input_df.columns.tolist())
    print("FEATURES:", features)

    input_df = input_df[features]

    prediction = model.predict(input_df)[0]
    alert = "Alta contaminación" if prediction > 100 else "Normal"

    return {
        "prediction_no2_24h": float(prediction),
        "alert": alert
    }