from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

#support files
#visualize_tree.py
#generate_dataset.py


app = FastAPI(title="Ecom Recommender API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # Angular dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
MODEL_PATH = "model.joblib"
CSV_PATH = "synthetic_ecom.csv"


class UserEvent(BaseModel):
	UserAge: int
	BrowsingTime: float
	PastPurchases: int
	CartAdds: int




def generate_dataset(n=1000, path=CSV_PATH, random_state=42):
    np.random.seed(random_state)
    ages = np.random.randint(16, 70, size=n)
    browsing = np.round(np.random.exponential(scale=5.0, size=n), 2) # minutes
    past = np.random.poisson(1.2, size=n)
    cart = np.random.poisson(0.8, size=n)

    # Heuristic for label
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


def train_and_save_model(csv_path=CSV_PATH, model_path=MODEL_PATH):
    df = pd.read_csv(csv_path)
    X = df[["UserAge", "BrowsingTime", "PastPurchases", "CartAdds"]]
    y = df["Recommend"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    model = DecisionTreeClassifier(max_depth=6, random_state=1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    joblib.dump(model, model_path)
    return model, acc


# Ensure dataset and model exist on startup
if not os.path.exists(CSV_PATH):
    print("Generating dataset...")
    generate_dataset(n=1200)

if not os.path.exists(MODEL_PATH):
    print("Training model...")
    model, acc = train_and_save_model()
    print(f"Trained and saved model. Test accuracy: {acc:.3f}")
else:
    model = joblib.load(MODEL_PATH)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/dataset")
def get_dataset():
    if not os.path.exists(CSV_PATH):
        raise HTTPException(status_code=404, detail="Dataset not found")
    return {"csv_path": CSV_PATH}


@app.post("/recommend")
def recommend(event: UserEvent):
    features = [[event.UserAge, event.BrowsingTime, event.PastPurchases, event.CartAdds]]
    pred = model.predict(features)[0]
    proba = model.predict_proba(features)[0].tolist() if hasattr(model, "predict_proba") else None
    return {"recommend": bool(pred), "probabilities": proba}