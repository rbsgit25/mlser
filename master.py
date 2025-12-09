from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

# ML imports
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from math import radians, sin, cos, asin, sqrt

#car test model imports
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

#productivity test imports

import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from sklearn.linear_model import LinearRegression
import requests
from io import StringIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


# ============================================================
# FastAPI Init + CORS
# ============================================================

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REAL ESTATE COST PREDICTION MODEL
# ============================================================

REAL_ESTATE_MODEL_PATH = "pipeline.joblib"

def load_real_estate_model():
    if not os.path.exists(REAL_ESTATE_MODEL_PATH):
        raise RuntimeError("Model file not found. Train the model first.")

    data = joblib.load(REAL_ESTATE_MODEL_PATH)
    return data["pipeline"], data["features"]

def save_real_estate_model(pipeline, features):
    joblib.dump({"pipeline": pipeline, "features": features}, REAL_ESTATE_MODEL_PATH)


try:
    pipeline, FEATURES = load_real_estate_model()
except:
    pipeline = None
    FEATURES = []


class PredictRequest(BaseModel):
    area_sqft: float
    num_floors: int
    num_rooms: int
    material_quality: str
    location_factor: float
    labor_cost_per_sqft: float
    material_cost_per_sqft: float
    project_duration_months: float


@app.post("/rfpredict")
async def rfpredict(payload: PredictRequest):

    if pipeline is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    df = pd.DataFrame([payload.dict()])
    pred = pipeline.predict(df)[0]
    return {"prediction": float(pred)}


@app.post("/rfupload")
async def rfupload(file: UploadFile = File(...)):
    contents = await file.read()
    os.makedirs("data", exist_ok=True)
    with open("data/uploaded.csv", "wb") as f:
        f.write(contents)
    return {"status": "uploaded"}


# ============================================================
# E-COMMERCE RECOMMENDATION SYSTEM
# ============================================================

ECOM_MODEL_PATH = "ecom_model.joblib"
ECOM_DATASET_PATH = "synthetic_ecom.csv"


class UserEvent(BaseModel):
    UserAge: int
    BrowsingTime: float
    PastPurchases: int
    CartAdds: int


def generate_ecom_dataset(n=1000, path=ECOM_DATASET_PATH):
    np.random.seed(42)
    ages = np.random.randint(16, 70, size=n)
    browsing = np.round(np.random.exponential(scale=5.0, size=n), 2)
    past = np.random.poisson(1.2, size=n)
    cart = np.random.poisson(0.8, size=n)

    score = (browsing * 0.6) + (past * 2.0) + (cart * 3.0) - (ages - 30) * 0.05
    prob = 1 / (1 + np.exp(-0.1 * (score - 3)))
    recommend = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "UserAge": ages,
        "BrowsingTime": browsing,
        "PastPurchases": past,
        "CartAdds": cart,
        "Recommend": recommend
    })
    df.to_csv(path, index=False)
    return df


def train_ecom_model():
    df = pd.read_csv(ECOM_DATASET_PATH)
    X = df[["UserAge", "BrowsingTime", "PastPurchases", "CartAdds"]]
    y = df["Recommend"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    model = DecisionTreeClassifier(max_depth=6, random_state=1)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    joblib.dump(model, ECOM_MODEL_PATH)
    return model, acc


# Ensure dataset + model exist
if not os.path.exists(ECOM_DATASET_PATH):
    generate_ecom_dataset()

if not os.path.exists(ECOM_MODEL_PATH):
    ecom_model, acc = train_ecom_model()
else:
    ecom_model = joblib.load(ECOM_MODEL_PATH)


@app.post("/recommend")
def recommend(event: UserEvent):
    X = [[event.UserAge, event.BrowsingTime, event.PastPurchases, event.CartAdds]]
    pred = ecom_model.predict(X)[0]
    proba = ecom_model.predict_proba(X)[0].tolist()
    return {"recommend": bool(pred), "probabilities": proba}


# ============================================================
# TAXI FARE PREDICTION (SVM)
# ============================================================

TAXI_MODEL_PATH = "fare_predictor_svr_pipeline.joblib"
taxi_model = None


def haversine_km(lat1, lon1, lat2, lon2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


class TripRequest(BaseModel):
    pickup_lat: float
    pickup_lon: float
    dropoff_lat: float
    dropoff_lon: float
    hour: int
    dayofweek: int
    surge_multiplier: float = 1.0
    pickup_zone: str = "ZoneA"


@app.on_event("startup")
def load_taxi_model():
    global taxi_model
    try:
        taxi_model = joblib.load(TAXI_MODEL_PATH)
        print("Taxi model loaded successfully")
    except:
        taxi_model = None
        print("Taxi model not loaded")


@app.post("/taxifarepredict")
def taxifarepredict(trip: TripRequest):

    if taxi_model is None:
        raise HTTPException(status_code=500, detail="Taxi fare model not loaded")

    dist = haversine_km(
        trip.pickup_lat, trip.pickup_lon,
        trip.dropoff_lat, trip.dropoff_lon
    )

    est_duration = max(2.0, dist / (25.0 / 60.0))

    X = [{
        "distance_km": dist,
        "duration_min": est_duration,
        "hour": trip.hour,
        "dayofweek": trip.dayofweek,
        "surge_multiplier": trip.surge_multiplier,
        "pickup_zone": trip.pickup_zone
    }]

    df = pd.DataFrame(X)
    pred = taxi_model.predict(df)[0]

    return {
        "predicted_fare": float(pred),
        "rounded": round(float(pred), 2),
        "features_used": X[0]
    }




#CAR TEST MODEL START

# CAR_DATA_PATH = "data/car.csv"

# Create folder if missing
df = pd.read_csv("data/car.csv")
df.columns = ['buying','maint','doors','persons','lug_boot','safety','class']

# Feature/target split
X = df.drop("class", axis=1)
y = df["class"]

categorical_cols = X.columns.tolist()

pipeline_car = Pipeline([
    ('onehot', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )),
    ('model', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'
    ))
])

pipeline_car.fit(X, y)

class CarFeatures(BaseModel):
    buying: str
    maint: str
    doors: str
    persons: str
    lug_boot: str
    safety: str

@app.post("/carpredict")
def carpredict(features: CarFeatures):
    input_df = pd.DataFrame([features.dict()])
    pred = pipeline_car.predict(input_df)[0]
    return {"prediction": pred}

#CAR TEST MODEL END



# ============================================================
# ROOT + HEALTH
# ============================================================


#productivty test


# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("productivity-api")


# -------------------------
# Storage paths
# -------------------------
HISTORY_DIR = "history"
MODEL_PATH = "bulk_productivity_model.joblib"
os.makedirs(HISTORY_DIR, exist_ok=True)

REQUIRED_COLUMNS = ["employee_id", "experience", "hours_worked", "complexity"]


# -------------------------
# Helper functions
# -------------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names: strip, lowercase, spaces -> underscores."""
    df.columns = (
        df.columns.astype(str)  # ensure strings
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def _validate_required_columns(df: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")


def _train_and_predict(df: pd.DataFrame) -> pd.DataFrame:
    """Train linear regression (use synthetic target if 'productivity' missing) and add predictions."""
    _validate_required_columns(df)

    if "productivity" not in df.columns:
        # Create a synthetic target if none provided (replace with real target in production)
        df["productivity"] = df["experience"] * 2.5 + df["hours_worked"] * 1.8 - df["complexity"] * 1.2

    X = df[["experience", "hours_worked", "complexity"]]
    y = df["productivity"]

    model = LinearRegression()
    model.fit(X, y)
    joblib.dump(model, MODEL_PATH)

    df["predicted_productivity"] = model.predict(X)
    df["employee_id"] = df["employee_id"].astype(str)

    return df


def _save_excel(df: pd.DataFrame, out_path: str):
    """Save dataframe to .xlsx using openpyxl engine."""
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="predictions", index=False)


def _save_pdf(df: pd.DataFrame, out_path: str, title="Employee Productivity Report"):
    doc = SimpleDocTemplate(out_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = [Paragraph(title, styles["Title"]), Spacer(1, 12)]

    cols = ["employee_id", "experience", "hours_worked", "complexity", "predicted_productivity"]
    table_data = [cols]
    for _, row in df[cols].iterrows():
        table_data.append([
            str(row["employee_id"]),
            f"{row['experience']}",
            f"{row['hours_worked']}",
            f"{row['complexity']}",
            f"{row['predicted_productivity']:.2f}"
        ])

    table = Table(table_data, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    elements.append(table)
    doc.build(elements)


def _timestamped_paths(prefix="report"):
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    excel_name = f"{prefix}_{ts}.xlsx"
    pdf_name = f"{prefix}_{ts}.pdf"
    excel_path = os.path.join(HISTORY_DIR, excel_name)
    pdf_path = os.path.join(HISTORY_DIR, pdf_name)
    return excel_name, pdf_name, excel_path, pdf_path


def convert_to_csv_url(url: str) -> str:
    """
    Convert many Google Sheets URL styles to a CSV-export URL.

    Accepts:
      - edit URL:    https://docs.google.com/spreadsheets/d/<ID>/edit#gid=0
      - export URL:  https://docs.google.com/spreadsheets/d/<ID>/export?format=csv&gid=0
      - publish URL: https://docs.google.com/spreadsheets/d/e/.../pub?output=csv
      - already-passed csv URL

    Returns a URL that can be fetched with requests.get(...)
    """
    url = str(url).strip()
    # If already looks like a csv export or pub link — return as-is
    if "output=csv" in url or "export?format=csv" in url or url.endswith(".csv"):
        return url

    if "spreadsheets" not in url:
        raise HTTPException(status_code=400, detail="Invalid Google Sheet link")

    try:
        # Extract sheet id
        sheet_id = url.split("/d/")[1].split("/")[0]
    except Exception:
        raise HTTPException(status_code=400, detail="Could not parse Sheet ID from URL")

    # extract gid if present
    gid = "0"
    if "gid=" in url:
        # get first gid param
        gid = url.split("gid=")[1].split("&")[0].split("#")[0]

    csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
    return csv_url


# -------------------------
# API Endpoints
# -------------------------
@app.post("/upload_csv")
async def upload_csv(file: UploadFile = File(...)):
    try:
        content_bytes = await file.read()
        # Try to decode as UTF-8, fallback to latin1
        try:
            content = content_bytes.decode("utf-8")
        except Exception:
            content = content_bytes.decode("latin1")

        # read CSV robustly: let pandas detect separators
        try:
            df = pd.read_csv(StringIO(content))
        except Exception as e:
            # fallback with python engine and auto sep sniff
            df = pd.read_csv(StringIO(content), sep=None, engine="python")

        df = normalize_columns(df)
        df = _train_and_predict(df)

        excel_name, pdf_name, excel_path, pdf_path = _timestamped_paths()
        _save_excel(df, excel_path)
        _save_pdf(df, pdf_path)

        records = df[["employee_id", "experience", "hours_worked", "complexity", "predicted_productivity"]].to_dict(orient="records")
        return {"status": "success", "employees": records, "excel": excel_name, "pdf": pdf_name}

    except Exception as e:
        logger.exception("upload_csv error")
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/upload-google-sheet")
async def upload_google_sheet(url: str = Form(...)):
    try:
        # 1️⃣ DO NOT convert published CSV link
        if "output=csv" in url:
            csv_url = url
        else:
            csv_url = convert_to_csv_url(url)

        logger.info(f"Fetching Google Sheet CSV: {csv_url}")

        resp = requests.get(csv_url, timeout=15)
        logger.info(f"Google CSV fetch status: {resp.status_code}")

        if resp.status_code == 404:
            raise HTTPException(status_code=404, detail="Google sheet CSV URL returned 404 — check sharing (must be 'Anyone with link' or published).")
        if resp.status_code == 403:
            raise HTTPException(status_code=403, detail="Access denied (403). Ensure sheet is shared publicly or published.")
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail=f"Failed to fetch CSV (status {resp.status_code})")

        # 2️⃣ parse CSV safely
        text = resp.text
        try:
            df = pd.read_csv(StringIO(text))
        except Exception:
            df = pd.read_csv(StringIO(text), sep=None, engine="python")

        df = normalize_columns(df)
        df = _train_and_predict(df)

        excel_name, pdf_name, excel_path, pdf_path = _timestamped_paths()
        _save_excel(df, excel_path)
        _save_pdf(df, pdf_path)

        records = df[["employee_id", "experience", "hours_worked", "complexity", "predicted_productivity"]].to_dict(orient="records")

        return {
            "status": "success",
            "employees": records,
            "excel": excel_name,
            "pdf": pdf_name
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("upload-google-sheet error")
        raise HTTPException(status_code=400, detail=f"Google Sheet fetch error: {str(e)}")


@app.get("/history")
def list_history():
    files = []
    for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
        if fname.endswith(".xlsx") or fname.endswith(".pdf"):
            stat = os.stat(os.path.join(HISTORY_DIR, fname))
            files.append({
                "name": fname,
                "size": stat.st_size,
                "modified": datetime.utcfromtimestamp(stat.st_mtime).isoformat() + "Z"
            })
    return {"files": files}


@app.get("/download/{file_name}")
def download_file(file_name: str):
    safe_path = os.path.join(HISTORY_DIR, file_name)
    if not os.path.exists(safe_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path=safe_path, filename=file_name, media_type="application/octet-stream")


@app.get("/predict_single")
def predict_single(experience: float, hours_worked: float, complexity: float):
    if not os.path.exists(MODEL_PATH):
        raise HTTPException(status_code=404, detail="Model not found. Upload CSV first.")
    model = joblib.load(MODEL_PATH)
    pred = model.predict([[experience, hours_worked, complexity]])
    return {"predicted_productivity": float(pred[0])}

#productivity test end


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/health")
def health():
    return {"status": "ok"}
