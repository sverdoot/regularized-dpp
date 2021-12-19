from abc import ABC, abstractmethod
from typing import Optional

import numpy as np
import scipy as sp
import scipy.linalg
from dppy.finite_dpps import FiniteDPP

from .metrics import A_opt_criterion
from .utils import timing_decorator


class Sampler(ABC):
    def __init__(self):
        self.time_cnts = []

    def reset(self):
        self.time_cnts = []

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError


class SamplerRegistry:
    registry = {}

    @classmethod
    def register(cls, name: Optional[str] = None) -> Sampler:
        def inner_wrapper(wrapped_class: Sampler) -> Sampler:
            if name is None:
                name_ = wrapped_class.__name__
            else:
                name_ = name
            cls.registry[name_] = wrapped_class
            return wrapped_class

        return inner_wrapper

    @classmethod
    def create_sampler(cls, name: str, **kwargs) -> Sampler:
        model = cls.registry[name]
        model = model(**kwargs)
        return model


@SamplerRegistry.register()
class RegDPP(Sampler):
    @timing_decorator
    def __call__(self, X: np.ndarray, A: np.ndarray, p: np.ndarray, k: int) -> np.ndarray:
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


@SamplerRegistry.register()
class Uniform(Sampler):
    @timing_decorator
    def __call__(self, X: np.ndarray, A: np.ndarray, p: np.ndarray, k: int):
        return np.random.choice(np.arange(X.shape[0]), k, replace=False)


@SamplerRegistry.register()
class PredictiveLength(Sampler):
    @timing_decorator
    def __call__(self, X: np.ndarray, A: np.ndarray, p: np.ndarray, k: int):
        p = np.linalg.norm(X, axis=1, ord=2)
        p /= p.sum()
        ids = np.random.choice(np.arange(X.shape[0]), k, replace=False, p=p)
        return ids


@SamplerRegistry.register()
class Greedy(Sampler):
    def __init__(self):
        super().__init__()
        self.X = None
        self.A = None
        self.ids = None

    @timing_decorator
    def __call__(self, X: np.ndarray, A: np.ndarray, p: np.ndarray, k: int):
        if (
            self.X is not None
            and self.A is not None
            and self.ids is not None
            # and np.allclose(X, self.X)
            # and np.allclose(A, self.A)
        ):
            k_range = np.arange(k - len(self.ids))
            ids = self.ids
        else:
            self.X = X
            self.A = A
            k_range = np.arange(k)
            ids = []
        left = np.array(list(set(np.arange(X.shape[0])) - set(ids)))
        for _ in k_range:
            left_idx = np.argmin(
                [A_opt_criterion(X[ids + [idx]].T @ X[ids + [idx]], A) for idx in left]
            )
            ids.append(left[left_idx])
            left = np.delete(left, left_idx)
        self.ids = ids
        return ids
