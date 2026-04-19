from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
load_dotenv()

# ── App setup ─────────────────────────────────────────────────────
app = FastAPI(title="Accident Risk Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load models & data ────────────────────────────────────────────
df = pd.read_csv("data/indian_roads_dataset.csv")
df['festival'] = df['festival'].fillna('none')

rf     = joblib.load("models/rf_model.pkl")
knn    = joblib.load("models/knn_model.pkl")
lr     = joblib.load("models/lr_model.pkl")
xgb    = joblib.load("models/xgb_model.pkl")
scaler = joblib.load("models/scaler.pkl")

# ── Confirmed encoding maps ───────────────────────────────────────
city_map = {
    'Bangalore': 0, 'Chandigarh': 1, 'Chennai': 2,
    'Delhi': 3, 'Hyderabad': 4, 'Kolkata': 5,
    'Mumbai': 6, 'Pune': 7
}
weather_map  = {'clear': 0, 'fog': 1, 'rain': 2}
road_map     = {'highway': 0, 'rural': 1, 'urban': 2}
vis_map      = {'high': 0, 'low': 1, 'medium': 2}
traffic_map  = {'high': 0, 'low': 1, 'medium': 2}
day_map      = {
    'Friday': 0, 'Monday': 1, 'Saturday': 2,
    'Sunday': 3, 'Thursday': 4, 'Tuesday': 5, 'Wednesday': 6
}
festival_map = {'Diwali': 0, 'Eid': 1, 'Holi': 2, 'New Year': 3, 'none': 4}
cause_map    = {
    'distraction': 0, 'drunk driving': 1,
    'overspeeding': 2, 'poor road': 3, 'weather': 4
}

def safe_encode(value, mapping):
    if value in mapping:
        return mapping[value]
    if value.lower() in mapping:
        return mapping[value.lower()]
    if value.title() in mapping:
        return mapping[value.title()]
    if value in ('None', 'none'):
        return mapping.get('none', 4)
    return 0

def compute_risk_score(weather, visibility, hour, is_peak_hour,
                       is_weekend, traffic_density, cause,
                       festival, road_type, temperature,
                       vehicles_involved, casualties):
    """
    Compute risk score from inputs instead of taking it as user input.
    This ensures different inputs produce meaningfully different predictions.
    """
    score = 0.3  # base score

    # ── Weather (high impact) ─────────────────────────────────────
    weather_scores = {'fog': 0.25, 'rain': 0.15, 'clear': 0.0}
    score += weather_scores.get(weather.lower(), 0.0)

    # ── Visibility (high impact) ──────────────────────────────────
    vis_scores = {'low': 0.20, 'medium': 0.08, 'high': 0.0}
    score += vis_scores.get(visibility.lower(), 0.0)

    # ── Time of day ───────────────────────────────────────────────
    # Late night / early morning is dangerous
    if 22 <= hour <= 23 or 0 <= hour <= 4:
        score += 0.18
    elif 5 <= hour <= 7:
        score += 0.08
    elif is_peak_hour:
        score += 0.10

    # ── Weekend ───────────────────────────────────────────────────
    if is_weekend:
        score += 0.06

    # ── Traffic density ───────────────────────────────────────────
    traffic_scores = {'high': 0.12, 'medium': 0.06, 'low': 0.0}
    score += traffic_scores.get(traffic_density.lower(), 0.0)

    # ── Cause (high impact) ───────────────────────────────────────
    cause_scores = {
        'drunk driving': 0.25,
        'overspeeding': 0.18,
        'distraction': 0.12,
        'poor road': 0.10,
        'weather': 0.08
    }
    score += cause_scores.get(cause.lower(), 0.0)

    # ── Festival ──────────────────────────────────────────────────
    if festival.lower() not in ('none', ''):
        score += 0.10

    # ── Road type ─────────────────────────────────────────────────
    road_scores = {'highway': 0.10, 'urban': 0.05, 'rural': 0.08}
    score += road_scores.get(road_type.lower(), 0.0)

    # ── Temperature extremes ──────────────────────────────────────
    if temperature >= 42 or temperature <= 8:
        score += 0.05

    # ── Incident severity indicators ─────────────────────────────
    if vehicles_involved >= 4:
        score += 0.08
    elif vehicles_involved >= 2:
        score += 0.04

    if casualties >= 3:
        score += 0.08
    elif casualties >= 1:
        score += 0.04

    # ── Clamp to valid range ──────────────────────────────────────
    return round(min(max(score, 0.05), 1.0), 3)

# ── Request schemas ───────────────────────────────────────────────
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
    model_choice: str = "Random Forest"

class ChatRequest(BaseModel):
    message: str

# ── Predict endpoint ──────────────────────────────────────────────
@app.post("/predict")
def predict(data: PredictionRequest):

    # Step 1 — compute risk score from inputs
    computed_risk = compute_risk_score(
        weather          = data.weather,
        visibility       = data.visibility,
        hour             = data.hour,
        is_peak_hour     = data.is_peak_hour,
        is_weekend       = data.is_weekend,
        traffic_density  = data.traffic_density,
        cause            = data.cause,
        festival         = data.festival,
        road_type        = data.road_type,
        temperature      = data.temperature,
        vehicles_involved= data.vehicles_involved,
        casualties       = data.casualties
    )

    # Step 2 — build feature array in exact order as modelsall.py FEATURES
    features = np.array([[
        data.hour,                                       # hour
        int(data.is_weekend),                            # is_weekend
        int(data.is_peak_hour),                          # is_peak_hour
        data.temperature,                                # temperature
        data.lanes,                                      # lanes
        int(data.traffic_signal),                        # traffic_signal
        data.vehicles_involved,                          # vehicles_involved
        data.casualties,                                 # casualties
        computed_risk,                                   # risk_score (computed)
        6,                                               # month (mid-year)
        safe_encode(data.weather, weather_map),          # weather_enc
        safe_encode(data.road_type, road_map),           # road_type_enc
        safe_encode(data.visibility, vis_map),           # visibility_enc
        safe_encode(data.traffic_density, traffic_map),  # traffic_density_enc
        safe_encode(data.festival, festival_map),        # festival_enc
        safe_encode(data.cause, cause_map),              # cause_enc
        safe_encode(data.day_of_week, day_map),          # day_of_week_enc
        safe_encode(data.city, city_map),                # city_enc
        0                                                # state_enc
    ]])

    # Step 3 — predict with selected model
    # Tree models (RF, XGB) — no scaling needed
    # Linear models (KNN, LR) — scaling needed
    if data.model_choice == 'Random Forest':
        pred  = rf.predict(features)[0]
        proba = rf.predict_proba(features)[0]

    elif data.model_choice == 'XGBoost':
        pred  = xgb.predict(features)[0]
        proba = xgb.predict_proba(features)[0]

    elif data.model_choice == 'KNN':
        features_sc = scaler.transform(features)
        pred  = knn.predict(features_sc)[0]
        proba = knn.predict_proba(features_sc)[0]

    else:  # Logistic Regression
        features_sc = scaler.transform(features)
        pred  = lr.predict(features_sc)[0]
        proba = lr.predict_proba(features_sc)[0]

    # Step 4 — convert severity probs to risk level
    # proba = [minor_prob, major_prob, fatal_prob]
    risk_val = float(proba[0] * 20 + proba[1] * 60 + proba[2] * 100)

    # Also blend with computed_risk for more sensitivity
    blended_risk = (risk_val * 0.6) + (computed_risk * 100 * 0.4)

    if blended_risk >= 58:
        risk_level = "HIGH"
    elif blended_risk >= 32:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    severity_labels = {0: 'Minor', 1: 'Major', 2: 'Fatal'}

    return {
        "risk_level":          risk_level,
        "risk_score":          round(blended_risk, 1),
        "computed_risk":       computed_risk,
        "predicted_severity":  severity_labels[int(pred)],
        "probabilities": {
            "minor": round(float(proba[0]) , 1),
            "major": round(float(proba[1]) , 1),
            "fatal": round(float(proba[2]), 1),
        },
        "model_used":   data.model_choice,
        "confidence":   round(float(max(proba)), 1)
    }

# ── Stats endpoint ────────────────────────────────────────────────
@app.get("/stats")
def stats():
    total = len(df)
    return {
        "total_records":  total,
        "fatal_count":    int((df["accident_severity"] == "fatal").sum()),
        "fatal_rate":     round((df["accident_severity"] == "fatal").mean() * 100, 1),
        "avg_risk":       round(df["risk_score"].mean(), 3),
        "cities":         int(df["city"].nunique()),
        "hourly_risk": {
            int(k): round(v, 3)
            for k, v in df.groupby("hour")["risk_score"].mean().to_dict().items()
        },
        "city_risk":    df.groupby("city")["risk_score"].mean().round(3).sort_values(ascending=False).to_dict(),
        "weather_risk": df.groupby("weather")["risk_score"].mean().round(3).to_dict(),
        "road_risk":    df.groupby("road_type")["risk_score"].mean().round(3).to_dict(),
        "severity_dist":df["accident_severity"].value_counts().to_dict(),
        "day_risk":     df.groupby("day_of_week")["risk_score"].mean().round(3).to_dict(),
    }

# ── City ranking ──────────────────────────────────────────────────
@app.get("/city-ranking")
def city_ranking():
    city_stats = df.groupby("city").agg(
        avg_risk         =("risk_score", "mean"),
        total_accidents  =("risk_score", "count"),
        fatal_accidents  =("accident_severity", lambda x: (x == "fatal").sum())
    ).reset_index()
    city_stats["fatal_rate"] = (
        city_stats["fatal_accidents"] / city_stats["total_accidents"] * 100
    ).round(1)
    city_stats["avg_risk"] = city_stats["avg_risk"].round(3)
    return city_stats.sort_values("avg_risk", ascending=False).to_dict(orient="records")

# ── Model performance ─────────────────────────────────────────────
@app.get("/model-performance")
def model_performance():
    return [
        {"model": "Random Forest",       "accuracy": 69.2, "precision": 71.0, "recall": 69.0, "f1": 70.0},
        {"model": "XGBoost",             "accuracy": 65.6, "precision": 67.0, "recall": 66.0, "f1": 66.0},
        {"model": "KNN",                 "accuracy": 52.7, "precision": 54.0, "recall": 53.0, "f1": 53.0},
        {"model": "Logistic Regression", "accuracy": 49.7, "precision": 51.0, "recall": 50.0, "f1": 50.0},
    ]

# ── Feature importance ────────────────────────────────────────────
@app.get("/feature-importance")
def feature_importance():
    try:
        fi = pd.read_csv("models/feature_importance.csv")
        return fi.head(10).to_dict(orient="records")
    except:
        features = [
            "hour", "is_weekend", "is_peak_hour", "temperature", "lanes",
            "traffic_signal", "vehicles_involved", "casualties", "risk_score",
            "month", "weather_enc", "road_type_enc", "visibility_enc",
            "traffic_density_enc", "festival_enc", "cause_enc",
            "day_of_week_enc", "city_enc", "state_enc"
        ]
        fi = sorted(
            zip(features, rf.feature_importances_),
            key=lambda x: x[1], reverse=True
        )
        return [
            {"feature": f, "importance": round(float(i), 4)}
            for f, i in fi[:10]
        ]

# ── Map endpoint ──────────────────────────────────────────────────
@app.get("/map")
def get_map():
    file_path = "models/hotspot_map.html"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="text/html")
    return {"error": "Map not found. Run map_generator.py first."}

# ── Chat endpoint ─────────────────────────────────────────────────
try:
    from google import genai as google_genai
    google_client = google_genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False

SYSTEM_PROMPT = """You are an expert road safety analyst AI for an Indian Road
Accident Risk Analyzer. Dataset has 20,000 records from 8 Indian cities
(Mumbai, Delhi, Pune, Chennai, Bangalore, Hyderabad, Kolkata, Chandigarh).
Best model: Random Forest with 69.2% accuracy.
Fatal accidents most common in fog and low visibility.
Peak hours 8-10 AM and 5-8 PM have highest risk.
Festival seasons (Diwali, Holi, Eid) show elevated accident rates.
Main causes: overspeeding, drunk driving, distraction, poor road, weather.
Answer concisely and helpfully about road safety topics only."""

@app.post("/chat")
def chat(data: ChatRequest):
    if not GEMINI_AVAILABLE:
        return {"reply": "Chat feature is currently unavailable."}
    try:
        prompt = f"{SYSTEM_PROMPT}\n\nUser: {data.message}\nAssistant:"
        response = google_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt
        )
        return {"reply": response.text}
    except Exception as e:
        return {"reply": f"Sorry, could not process that. Error: {str(e)}"}

# ── Debug endpoint ────────────────────────────────────────────────
@app.post("/predict-debug")
def predict_debug(data: PredictionRequest):
    computed_risk = compute_risk_score(
        data.weather, data.visibility, data.hour,
        data.is_peak_hour, data.is_weekend, data.traffic_density,
        data.cause, data.festival, data.road_type,
        data.temperature, data.vehicles_involved, data.casualties
    )
    features = [
        data.hour, int(data.is_weekend), int(data.is_peak_hour),
        data.temperature, data.lanes, int(data.traffic_signal),
        data.vehicles_involved, data.casualties, computed_risk, 6,
        safe_encode(data.weather, weather_map),
        safe_encode(data.road_type, road_map),
        safe_encode(data.visibility, vis_map),
        safe_encode(data.traffic_density, traffic_map),
        safe_encode(data.festival, festival_map),
        safe_encode(data.cause, cause_map),
        safe_encode(data.day_of_week, day_map),
        safe_encode(data.city, city_map),
        0
    ]
    return {
        "computed_risk_score": computed_risk,
        "features_sent": features,
        "feature_names": [
            "hour", "is_weekend", "is_peak_hour", "temperature", "lanes",
            "traffic_signal", "vehicles_involved", "casualties", "risk_score",
            "month", "weather_enc", "road_type_enc", "visibility_enc",
            "traffic_density_enc", "festival_enc", "cause_enc",
            "day_of_week_enc", "city_enc", "state_enc"
        ]
    }

# ── Root ──────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "status": "Road Accident Risk API is running",
        "endpoints": [
            "GET  /stats",
            "GET  /city-ranking",
            "GET  /model-performance",
            "GET  /feature-importance",
            "GET  /map",
            "POST /predict",
            "POST /predict-debug",
            "POST /chat",
        ]
    }