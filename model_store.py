# app/model_store.py
from sklearn.linear_model import SGDRegressor
import numpy as np
from threading import Lock

class OnlineLinearRegression:
    def __init__(self):
        # SGDRegressor supports partial_fit for online learning
        self.model = SGDRegressor(max_iter=1, tol=None, learning_rate="invscaling", eta0=0.01)
        self.initialized = False
        self.lock = Lock()

    def partial_fit(self, X, y):
        X = np.asarray(X).reshape(-1, 1)
        y = np.asarray(y).ravel()
        with self.lock:
            if not self.initialized:
                # partial_fit requires classes for classifiers; for regressor we just call once with shape
                # We to initialize by calling partial_fit once with appropriate shapes.
                # But SGDRegressor for regression simply accepts partial_fit.
                self.model.partial_fit(X, y)
                self.initialized = True
            else:
                self.model.partial_fit(X, y)

    def predict(self, X):
        X = np.asarray(X).reshape(-1, 1)
        with self.lock:
            if not self.initialized:
                # no training yet -> return zeros
                return np.zeros(len(X)).tolist()
            return self.model.predict(X).tolist()

    def get_params(self):
        with self.lock:
            if not self.initialized:
                return {"coef": None, "intercept": None}
            return {"coef": self.model.coef_.tolist(), "intercept": float(self.model.intercept_[0])}
