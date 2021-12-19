import numpy as np


def A_opt_criterion(cov: np.ndarray, A: np.ndarray) -> float:
    return np.trace(np.linalg.inv(cov + A))


def C_opt_criterion(cov: np.ndarray, A: np.ndarray, c: np.ndarray) -> float:
    return c.T @ np.linalg.inv(cov + A) @ c


def D_opt_criterion(cov: np.ndarray, A: np.ndarray) -> np.ndarray:
    return np.linalg.det(cov + A) ** (-1.0 / cov.shape[0])


def V_opt_criterion(X: np.ndarray, cov: np.ndarray, A: np.ndarray) -> float:
    return 1.0 / X.shape[0] * np.trace(X @ np.linalg.inv(cov + A) @ X.T)
