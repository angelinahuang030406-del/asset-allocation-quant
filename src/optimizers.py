"""Allocation rules. Each takes a window of past monthly returns and returns weights."""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform


def equal_weight(r):
    return pd.Series(1 / r.shape[1], index=r.columns)


def inverse_vol(r):
    iv = 1 / r.std()
    return iv / iv.sum()


def _solve(objective, n, cap=1.0):
    """Long-only, fully invested: weights in [0, cap], sum to 1.
    ftol is tight because the default (1e-6) let SLSQP stop early and still report success."""
    res = minimize(objective, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, cap)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1},
                   options={"ftol": 1e-12, "maxiter": 1000})
    assert res.success, res.message
    return res.x


def min_variance(r, cap=1.0):
    cov = r.cov().values * 1e4  # monthly variances are ~1e-4; scale so the solver can see changes
    w = _solve(lambda w: w @ cov @ w, r.shape[1], cap)
    return pd.Series(w, index=r.columns)


def max_sharpe(r, cap=1.0):
    """Markowitz tangency portfolio. Pass EXCESS returns (asset minus T-bill) so the
    ratio is a real Sharpe ratio; mean and covariance are trailing sample estimates."""
    mu, cov = r.mean().values, r.cov().values
    w = _solve(lambda w: -(w @ mu) / np.sqrt(w @ cov @ w), r.shape[1], cap)
    return pd.Series(w, index=r.columns)


def risk_parity(r):
    """Equal risk contribution: every asset adds the same amount of variance."""
    # convex trick (Spinu 2013): minimize 0.5*w'Cw - mean(log w), then rescale to sum to 1
    cov = r.cov().values * 1e4  # scale up so the solver tolerance is meaningful
    n = r.shape[1]
    res = minimize(lambda w: 0.5 * w @ cov @ w - np.log(w).mean(), np.ones(n) / n,
                   jac=lambda w: cov @ w - 1 / (n * w),
                   method="L-BFGS-B", bounds=[(1e-8, None)] * n)
    return pd.Series(res.x / res.x.sum(), index=r.columns)


def hrp(r):
    """Hierarchical Risk Parity (Lopez de Prado, 2016)."""
    cov, corr = r.cov(), r.corr()
    # 1) cluster assets by correlation distance
    dist = np.sqrt(((1 - corr) / 2).clip(lower=0))
    order = corr.index[leaves_list(linkage(squareform(dist.values, checks=False), "single"))]
    # 2) recursive bisection: split each group in half, give the less risky half more weight
    w = pd.Series(1.0, index=order)
    groups = [list(order)]
    while groups:
        next_groups = []
        for g in groups:
            if len(g) < 2:
                continue
            left, right = g[:len(g) // 2], g[len(g) // 2:]
            v_left, v_right = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = v_right / (v_left + v_right)   # share of weight that goes left
            w[left] *= alpha
            w[right] *= 1 - alpha
            next_groups += [left, right]
        groups = next_groups
    return w.reindex(r.columns)


def _cluster_var(cov, names):
    c = cov.loc[names, names]
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return ivp @ c.values @ ivp
