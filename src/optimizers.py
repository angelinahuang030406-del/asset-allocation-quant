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
    """Long-only, fully invested: weights in [0, cap], sum to 1."""
    res = minimize(objective, np.ones(n) / n, method="SLSQP",
                   bounds=[(0, cap)] * n,
                   constraints={"type": "eq", "fun": lambda w: w.sum() - 1})
    return res.x


def min_variance(r, cap=1.0):
    cov = r.cov().values
    w = _solve(lambda w: w @ cov @ w, r.shape[1], cap)
    return pd.Series(w, index=r.columns)


def max_sharpe(r, cap=1.0):
    """Classic Markowitz tangency portfolio using SAMPLE mean returns."""
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
    # 2) recursive bisection: split risk between the two halves by inverse variance
    w = pd.Series(1.0, index=order)
    clusters = [list(order)]
    while clusters:
        clusters = [c[i:j] for c in clusters for i, j in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for k in range(0, len(clusters), 2):
            left, right = clusters[k], clusters[k + 1]
            v_left, v_right = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - v_left / (v_left + v_right)
            w[left] *= alpha
            w[right] *= 1 - alpha
    return w.reindex(r.columns)


def _cluster_var(cov, names):
    c = cov.loc[names, names]
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return ivp @ c.values @ ivp
