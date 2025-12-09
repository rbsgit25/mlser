import os
import logging
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import requests
from io import StringIO
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

# --- logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("productivity-api")

app = FastAPI()

# -------------------------
# CORS for Angular
# -------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200", "http://127.0.0.1:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
