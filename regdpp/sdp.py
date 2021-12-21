import cvxpy as cp
import numpy as np


def cp_D_opt_criterion(cov, A: np.ndarray):
    res = cp.exp(-1.0 / cov.shape[0] * cp.log_det(cov + A))
    return res


def get_optimal_weights(X: np.ndarray, A: np.ndarray, size: int, solver="scs") -> np.ndarray:
    n, d = X.shape

    def objective(weights):
        cov = cp.multiply(weights[:, None] @ np.ones((1, d)), X).T @ X
        val = cp_D_opt_criterion(cov, A)
        return val

    weights = cp.Variable(X.shape[0])
    cp_ob = cp.Minimize(objective(weights))
    constraints = [cp.sum(weights) == size, weights <= 1, weights >= 0]
    prob = cp.Problem(cp_ob, constraints)
    if solver == "scs":
        solver = cp.SCS
    elif solver == "mosek":
        solver = cp.MOSEK

    prob.solve(solver=solver)
    return np.minimum(
        np.maximum(weights.value, np.zeros(weights.value.shape)), np.ones(weights.value.shape)
    )
