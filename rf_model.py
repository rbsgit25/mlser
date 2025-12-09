import joblib
from typing import Dict


MODEL_FILE = 'pipeline.joblib'


def load_model(path=MODEL_FILE):
 data = joblib.load(path)
 return data['pipeline'], data.get('features')


def save_model(pipeline, features, path=MODEL_FILE):
 joblib.dump({'pipeline': pipeline, 'features': features}, path)