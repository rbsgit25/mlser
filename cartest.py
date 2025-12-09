from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI
app = FastAPI()

# CORS for Angular
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying','maint','doors','persons','lug_boot','safety','class']
df = pd.read_csv(url, names=columns)

# Feature/target split
X = df.drop("class", axis=1)
y = df["class"]

# Build Pipeline with OneHotEncoder
categorical_cols = X.columns.tolist()

pipeline = Pipeline([
    ('onehot', ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ]
    )),
    ('model', LogisticRegression(
        multi_class='multinomial',
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced'   # FIX IMBALANCE
    ))
])

# Train model
pipeline.fit(X, y)

# Input model
class CarFeatures(BaseModel):
    buying: str
    maint: str
    doors: str
    persons: str
    lug_boot: str
    safety: str

@app.post("/predict")
def predict_car(features: CarFeatures):
    input_df = pd.DataFrame([features.dict()])
    pred = pipeline.predict(input_df)[0]
    return {"prediction": pred}
