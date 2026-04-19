from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
load_dotenv()

# -------------------------
# App Setup
# -------------------------
app = FastAPI(title="Accident Risk Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Load Data + Models
# -------------------------

df = pd.read_csv("data/indian_roads_dataset.csv")

model = joblib.load("models/rf_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# -------------------------
# Prediction Request Schema
# -------------------------

class PredictionRequest(BaseModel):
    city: str
    weather: str
    visibility: str
    road_type: str
    traffic_density: str
    day_of_week: str
    festival: str
    cause: str

    hour: int
    temperature: float
    lanes: int

    vehicles_involved: int
    casualties: int

    is_weekend: bool
    is_peak_hour: bool
    traffic_signal: bool

    risk_score: float


# -------------------------
# Helpers
# -------------------------

def encode(value, column):
    if value in df[column].unique():
        return list(df[column].unique()).index(value)
    return 0


# -------------------------
# Predict Endpoint
# -------------------------

@app.post("/predict")
def predict(data: PredictionRequest):

    features = [
        data.hour,
        int(data.is_weekend),
        int(data.is_peak_hour),
        data.temperature,
        data.lanes,
        int(data.traffic_signal),
        data.vehicles_involved,
        data.casualties,
        data.risk_score,
        1,
        encode(data.weather, "weather"),
        encode(data.road_type, "road_type"),
        encode(data.visibility, "visibility"),
        encode(data.traffic_density, "traffic_density"),
        encode(data.festival, "festival"),
        encode(data.cause, "cause"),
        encode(data.day_of_week, "day_of_week"),
        encode(data.city, "city"),
        0,
    ]

    X = np.array(features).reshape(1, -1)
    X = scaler.transform(X)

    prediction = model.predict(X)[0]
    prob = model.predict_proba(X).max()

    labels = ["LOW", "MEDIUM", "HIGH"]

    return {
        "risk_level": labels[prediction],
        "probability": round(float(prob), 2),
    }


# -------------------------
# Stats Endpoint
# -------------------------

@app.get("/stats")
def stats():

    total = len(df)

    fatal_rate = (
        (df["accident_severity"] == "fatal").sum()
        / total
        * 100
    )

    avg_risk = df["risk_score"].mean()

    hourly = (
        df.groupby("hour")["risk_score"]
        .mean()
        .to_dict()
    )

    city = (
        df.groupby("city")["risk_score"]
        .mean()
        .to_dict()
    )

    weather = (
        df.groupby("weather")["risk_score"]
        .mean()
        .to_dict()
    )

    return {
        "total_records": total,
        "fatal_rate": round(fatal_rate, 1),
        "avg_risk": round(avg_risk, 2),
        "hourly_risk": hourly,
        "city_risk": city,
        "weather_risk": weather,
    }


# -------------------------
# City Ranking
# -------------------------

@app.get("/city-ranking")
def city_ranking():

    grouped = df.groupby("city")

    result = []

    for city, data in grouped:

        total = len(data)

        fatal = (
            data["accident_severity"] == "fatal"
        ).sum()

        avg_risk = data["risk_score"].mean()

        result.append(
            {
                "city": city,
                "total_accidents": int(total),
                "fatal_accidents": int(fatal),
                "avg_risk": round(avg_risk, 2),
            }
        )

    result.sort(
        key=lambda x: x["avg_risk"],
        reverse=True,
    )

    return result


# -------------------------
# Model Performance
# -------------------------

@app.get("/model-performance")
def model_performance():

    return [
        {
            "model": "Random Forest",
            "accuracy": 69.2,
            "precision": 71.0,
            "recall": 69.0,
            "f1": 70.0,
        },
        {
            "model": "XGBoost",
            "accuracy": 65.6,
            "precision": 67.0,
            "recall": 66.0,
            "f1": 66.0,
        },
        {
            "model": "KNN",
            "accuracy": 52.7,
            "precision": 54.0,
            "recall": 53.0,
            "f1": 53.0,
        },
        {
            "model": "Logistic Regression",
            "accuracy": 49.7,
            "precision": 51.0,
            "recall": 50.0,
            "f1": 50.0,
        },
    ]


# -------------------------
# Map Endpoint
# -------------------------

from fastapi.responses import FileResponse

@app.get("/map")
def get_map():

    file_path = "models/hotspot_map.html"

    if os.path.exists(file_path):

        return FileResponse(
            file_path,
            media_type="text/html"
        )

    return {"error": "Map not found"}

from google import genai
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)
@app.post("/chat")
def chat(data: dict):

    user_message = data.get("message", "")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=user_message
        )

        return {
            "reply": response.text
        }

    except Exception as e:

        return {
            "reply": f"Error: {str(e)}"
        }