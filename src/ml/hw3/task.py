import numpy as np


# Task 1

def mse(y_true: np.ndarray, y_predicted: np.ndarray):
    return np.mean((y_true - y_predicted) ** 2)


def r2(y_true: np.ndarray, y_predicted: np.ndarray):
    i = np.sum((y_true - y_predicted) ** 2)
    j = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - i / j


# Task 2

class NormalLR:
    def __init__(self):
        self.weights = None  # Save weights here

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1), dtype=float)
        X_aug = np.hstack([X, ones])

        self.weights = np.linalg.pinv(X_aug) @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)

        n_samples = X.shape[0]
        ones = np.ones((n_samples, 1), dtype=float)
        X_aug = np.hstack([X, ones])

        return X_aug @ self.weights


# Task 3


class GradientLR:
    def __init__(self, alpha: float, iterations=10000, l=0.):
        self.alpha = alpha
        self.iterations = iterations
        self.l = l
        self.weights = None

    @staticmethod
    def _soft_threshold(w: np.ndarray, t: float) -> np.ndarray:
        return np.sign(w) * np.maximum(np.abs(w) - t, 0.0)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1, dtype=float)

        X_ext = np.c_[np.ones(n_samples), X]

        for _ in range(self.iterations):
            error = (X_ext @ self.weights) - y

            grad = (2.0 / n_samples) * (X_ext.T @ error)

            self.weights -= self.alpha * grad

            if self.l != 0.0:
                self.weights[1:] = self._soft_threshold(self.weights[1:], self.alpha * self.l)

        return self

    def predict(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n_samples = X.shape[0]
        X_ext = np.c_[np.ones(n_samples), X]
        return X_ext @ self.weights


# Task 4

def get_feature_importance(linear_regression):
    return np.abs(linear_regression.weights[1:])

def get_most_important_features(linear_regression):
    i = get_feature_importance(linear_regression)
    return np.argsort(i)[::-1]
