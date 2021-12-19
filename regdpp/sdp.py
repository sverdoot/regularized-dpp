# import scs
import cvxpy as cp
import numpy as np


def cp_D_opt_criterion(cov, A: np.ndarray):
    res = cp.exp(-1.0 / cov.shape[0] * cp.log_det(cov + A))
    return res


def get_optimal_weights(X: np.ndarray, A: np.ndarray, size: int) -> np.ndarray:
    def objective(weights):
        # cov = (np.einsum('ij,ik->ijk', x, x) * weights[:, None, None]).sum()
        cov = [w * np.outer(x_i, x_i) for w, x_i in zip(weights, X)]
        cov = np.sum(cov, 0)
        val = cp_D_opt_criterion(cov, A)
        return val

    weights = cp.Variable(X.shape[0])
    cp_ob = cp.Minimize(objective(weights))
    constraints = [cp.sum(weights) == size, weights <= 1, weights >= 0]
    prob = cp.Problem(cp_ob, constraints)
    prob.solve(solver=cp.MOSEK)
    return weights.value
