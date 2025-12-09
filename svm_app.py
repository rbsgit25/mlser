from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from math import radians, sin, cos, asin, sqrt
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

MODEL_PATH = "fare_predictor_svr_pipeline.joblib"

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Haversine Distance
def haversine_km(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c  # KM

# Data Model
class TripRequest(BaseModel):
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    hour: int
    dayofweek: int
    surge_multiplier: float = 1.0
    pickup_zone: str = "ZoneA"

# Load the ML model
@app.on_event("startup")
def load_model():
    global model
    try:
        model = joblib.load(MODEL_PATH)
        print("Model loaded successfully")
    except Exception as e:
        print("Failed to load model:", e)
        model = None

@app.post("/predict")
def predict(trip: TripRequest):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Compute distance
    dist = haversine_km(
        trip.pickup_lat, trip.pickup_lon,
        trip.dropoff_lat, trip.dropoff_lon
    )

    # Estimated duration (minutes)
    est_duration = max(2.0, dist / (25.0 / 60.0))

    # Prepare single-row dataframe
    X = [{
        "distance_km": dist,
        "duration_min": est_duration,
        "hour": trip.hour,
        "dayofweek": trip.dayofweek,
        "surge_multiplier": trip.surge_multiplier,
        "pickup_zone": trip.pickup_zone
    }]

    df = pd.DataFrame(X)

    # Predict
    pred = model.predict(df)[0]

    return {
        "predicted_fare": float(pred),
        "predicted_fare_rounded": round(float(pred), 2),
        "features_used": X[0]
    }
