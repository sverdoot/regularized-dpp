import numpy as np
import scipy as sp
import scipy.linalg
from dppy.finite_dpps import FiniteDPP

from .utils import timing_decorator


class RegDPP:
    def __init__(self):
        self.time_cnts = []

    def reset(self):
        self.time_cnts = []

    @timing_decorator
    def __call__(self, X: np.ndarray, A: np.ndarray, p: np.ndarray) -> np.ndarray:
        """Sample efficient subspace

        Args:
            X - data matrix of shape (n x d), n - number of experiments, d - dimension
            A - prior precision matrix of shape (d x d)
            p - bernoulli probability vector of shape n

        Returns:
            array with indices of points in subspace
        """
        diag_p = np.diag(p)
        Z = A + X.T @ diag_p @ X
        s, V = sp.linalg.eigh(Z)
        Z_inv_sq = V @ np.diag(s ** (-0.5)) @ sp.linalg.inv(V)
        B = (diag_p ** 0.5) @ X @ Z_inv_sq
        DPP = FiniteDPP("correlation", **{"K": B @ B.T})
        T = DPP.sample_exact()
        b = np.random.binomial(1, p)
        S = np.union1d(T, np.where(b == 1))
        return S
