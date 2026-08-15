"""Linear alignment between two embedding spaces (paper Eq. 1)."""
import numpy as np


def l2(x, eps=1e-12):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


class LinearAlignment:
    """Mean-centered least-squares map from a source space to a target space."""

    def __init__(self):
        self.W = None
        self.mu_a = None
        self.mu_c = None

    def fit(self, A, C):
        A = np.asarray(A, dtype=np.float64)
        C = np.asarray(C, dtype=np.float64)
        self.mu_a = A.mean(0)
        self.mu_c = C.mean(0)
        self.W = np.linalg.lstsq(A - self.mu_a, C - self.mu_c, rcond=None)[0]
        return self

    def transform(self, A):
        return (np.asarray(A, dtype=np.float64) - self.mu_a) @ self.W + self.mu_c
