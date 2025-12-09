# train_model.py
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from math import radians, sin, cos, asin, sqrt

RNG = np.random.RandomState(42)

def haversine_km(lat1, lon1, lat2, lon2):
    # distance in kilometers
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return 6371 * c

def generate_synthetic_data(n=20000):
    # Simulate cities roughly within a bounding box
    # Example: random coords around a city center (e.g., Mumbai approx)
    center_lat, center_lon = 19.0760, 72.8777
    lats_pick = center_lat + RNG.normal(scale=0.05, size=n)
    lons_pick = center_lon + RNG.normal(scale=0.05, size=n)
    lats_drop = center_lat + RNG.normal(scale=0.06, size=n)
    lons_drop = center_lon + RNG.normal(scale=0.06, size=n)

    distances = np.array([
        haversine_km(a,b,c,d) for a,b,c,d in zip(lats_pick, lons_pick, lats_drop, lons_drop)
    ])

    # Duration roughly proportional to distance plus noise and traffic effects
    base_speed_kmph = 25.0
    durations_min = distances / (base_speed_kmph/60.0) + RNG.normal(scale=5.0, size=n)
    durations_min = np.clip(durations_min, 2, None)

    # Time features
    hours = RNG.randint(0,24,size=n)
    dayofweek = RNG.randint(0,7,size=n)

    # Demand / surge multiplier (1.0 normal, up to 2.5)
    surge = RNG.choice([1.0, 1.2, 1.5, 2.0], size=n, p=[0.7,0.15,0.1,0.05])

    # Simple location zones (coarse)
    zones = RNG.choice(['ZoneA','ZoneB','ZoneC','ZoneD','ZoneE'], size=n, p=[0.3,0.25,0.2,0.15,0.1])

    # Fare formula (synthetic ground truth)
    base_fare = 25.0
    per_km = 12.0
    per_min = 1.5
    # Add extra for pickups in ZoneE to simulate airport/long-fee
    zone_extra = np.array([10.0 if z=='ZoneE' else 0.0 for z in zones])

    fare = (base_fare + per_km*distances + per_min*durations_min + zone_extra)
    # apply surge
    fare = fare * surge
    # simulate rounding & noise
    fare = np.round(fare + RNG.normal(scale=5.0, size=n), 2)
    fare = np.clip(fare, 15.0, None)

    df = pd.DataFrame({
        'pickup_lat': lats_pick,
        'pickup_lon': lons_pick,
        'dropoff_lat': lats_drop,
        'dropoff_lon': lons_drop,
        'distance_km': distances,
        'duration_min': durations_min,
        'hour': hours,
        'dayofweek': dayofweek,
        'surge_multiplier': surge,
        'pickup_zone': zones,
        'fare': fare
    })
    return df

def build_and_train(df):
    # features and target
    X = df.drop(columns=['fare'])
    y = df['fare']

    # Which columns to treat as numeric vs categorical
    numeric_features = ['distance_km', 'duration_min', 'hour', 'dayofweek', 'surge_multiplier']
    categorical_features = ['pickup_zone']

    # Preprocessing
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
        ],
        remainder='drop'  # drop other columns
    )

    # Pipeline: preprocessing + SVR
    pipeline = Pipeline([
        ('pre', preprocessor),
        ('svr', SVR(kernel='rbf'))
    ])

    # Train / test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

    # Hyperparameter search (small grid for speed; adjust as needed)
    param_grid = {
          'svr__C': [10.0],
    'svr__gamma': ['scale'],
    'svr__epsilon': [1.0]
    }
    gs = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=2, verbose=1)
    gs.fit(X_train, y_train)

    print("Best params:", gs.best_params_)
    best = gs.best_estimator_

    # evaluate
    preds = best.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, preds))
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print("RMSE:", rmse)
    # print("RMSE:", mean_squared_error(y_test, preds, squared=False))
    print("R2:", r2_score(y_test, preds))

    # Save pipeline
    print("Saving model now...")
    joblib.dump(best, 'fare_predictor_svr_pipeline.joblib')
    print("Saved model to fare_predictor_svr_pipeline.joblib")
    return best

if __name__ == '__main__':
    print("Generating data...")
    df = generate_synthetic_data(n=1000)
    print("Training model...")
    model = build_and_train(df)
