import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


CSV_PATH = "data/sample_construction_costs.csv"
MODEL_OUT = "pipeline.joblib"


# 1) load sample data
df = pd.read_csv(CSV_PATH)


# Example feature columns (these must match CSV)
FEATURES = [
'area_sqft',
'num_floors',
'num_rooms',
'material_quality', # categorical: low, medium, high
'location_factor', # numerical factor (0.8..1.3)
'labor_cost_per_sqft',
'material_cost_per_sqft',
'project_duration_months'
]
TARGET = 'total_cost'


X = df[FEATURES]
y = df[TARGET]


# Identify numeric and categorical columns
numeric_features = [c for c in FEATURES if c not in ['material_quality']]
cat_features = ['material_quality']


preprocessor = ColumnTransformer(
transformers=[
("num", StandardScaler(), numeric_features),
("cat", OneHotEncoder(handle_unknown='ignore'), cat_features),
]
)


pipe = Pipeline([
("pre", preprocessor),
("rf", RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


pipe.fit(X_train, y_train)


preds = pipe.predict(X_test)
print("MAE:", mean_absolute_error(y_test, preds))
# print("RMSE:", mean_squared_error(y_test, preds, squared=False))
print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))


# Save pipeline
joblib.dump({'pipeline': pipe, 'features': FEATURES}, MODEL_OUT)
print(f"Saved model pipeline to {MODEL_OUT}")