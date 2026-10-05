"""Check that the optimizers actually reach their optimum, not just report success.
Run from the repo root:  python -m pytest tests"""
import sys

import numpy as np

sys.path.insert(0, "src")
import optimizers as opt  # noqa: E402
from data import load  # noqa: E402
from strategies import rebalance_dates  # noqa: E402

rets, _, _ = load()
WINDOWS = [rets[rets.index < d].iloc[-60:] for d in rebalance_dates(rets.index)]


def test_min_variance_is_optimal():
    """At the true long-only minimum, every asset you hold has the same marginal variance
    (C @ w), and every asset you don't hold has a marginal variance at least that high."""
    for r in WINDOWS:
        w = opt.min_variance(r).values
        mv = r.cov().values @ w
        held = w > 1e-4
        assert mv[held].max() / mv[held].min() < 1.001, "held assets have unequal marginal variance"
        assert (mv[~held] >= mv[held].min() * 0.999).all(), "an unheld asset would lower variance"


def test_risk_parity_equal_contributions():
    for r in WINDOWS:
        w = opt.risk_parity(r).values
        rc = w * (r.cov().values @ w)
        assert rc.max() / rc.min() < 1.001, "risk contributions are not equal"
