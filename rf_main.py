from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

MODEL_PATH = "pipeline.joblib"


# -------------------------
# Load and Save Model
# -------------------------

def load_model():
    """Load trained pipeline and feature names."""
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError("Model file not found. Train the model first.")

    data = joblib.load(MODEL_PATH)

    pipeline = data["pipeline"]
    features = data["features"]

    return pipeline, features


def save_model(pipeline, features):
    """Save trained model."""
    joblib.dump({"pipeline": pipeline, "features": features}, MODEL_PATH)


# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow Angular/React/Vue/etc
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model on startup
try:
    pipeline, FEATURES = load_model()
except Exception as e:
    pipeline = None
    FEATURES = []
    print("WARNING: Model not loaded:", e)


# -------------------------
# Request Model
# -------------------------

class PredictRequest(BaseModel):
    area_sqft: float
    num_floors: int
    num_rooms: int
    material_quality: str
    location_factor: float
    labor_cost_per_sqft: float
    material_cost_per_sqft: float
    project_duration_months: float


# -------------------------
# Routes
# -------------------------

@app.get("/")
def root():
    return {"status": "ok", "model_loaded": pipeline is not None}


@app.post("/rfpredict")
async def predict(payload: PredictRequest):
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        row = [{
            'area_sqft': payload.area_sqft,
            'num_floors': payload.num_floors,
            'num_rooms': payload.num_rooms,
            'material_quality': payload.material_quality,
            'location_factor': payload.location_factor,
            'labor_cost_per_sqft': payload.labor_cost_per_sqft,
            'material_cost_per_sqft': payload.material_cost_per_sqft,
            'project_duration_months': payload.project_duration_months,
        }]

        df = pd.DataFrame(row)
        pred = pipeline.predict(df)[0]

        return {"prediction": float(pred)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/rfupload")
async def upload_csv(file: UploadFile = File(...)):
    contents = await file.read()
    os.makedirs("data", exist_ok=True)

    with open("data/uploaded.csv", "wb") as f:
        f.write(contents)

    return {"status": "uploaded", "filename": file.filename}


@app.post("/retrain")
async def retrain_from_uploaded():
    from train import FEATURES as TRAIN_FEATURES, TARGET, pipeline as train_pipe

    df = pd.read_csv("data/uploaded.csv")

    X = df[TRAIN_FEATURES]
    y = df[TARGET]

    # retrain
    train_pipe.fit(X, y)

    # save updated model
    save_model(train_pipe, TRAIN_FEATURES)

    return {"status": "retrained"}
