import numpy as np


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


class LinearRegressor:
    def fit(self, X, y):
        Xb = add_bias(X)
        return np.linalg.inv(Xb.T @ Xb) @ (Xb.T @ y)

    def predict(self, X, w):
        Xb = add_bias(X)
        return Xb @ w
