"""Allocation rules and the walk-forward loop, shared by every script and by the tests."""
import pandas as pd

import optimizers as opt
from backtest import backtest

# every rule gets (past asset returns, past T-bill returns) and returns weights
RULES = {
    "Equal Weight": lambda r, rf: opt.equal_weight(r),
    "Inverse Vol": lambda r, rf: opt.inverse_vol(r),
    "Min Variance": lambda r, rf: opt.min_variance(r),
    "Max Sharpe (no cap)": lambda r, rf: opt.max_sharpe(r.sub(rf, axis=0)),
    "Max Sharpe (30% cap)": lambda r, rf: opt.max_sharpe(r.sub(rf, axis=0), cap=0.30),
    "Risk Parity": lambda r, rf: opt.risk_parity(r),
    "HRP": lambda r, rf: opt.hrp(r),
}


def rebalance_dates(index, first=60):
    """Every January once `first` months of history exist."""
    return [d for d in index[first:] if d.month == 1]


def walkforward_weights(rets, rf, rule, window=60, first=60):
    """Weights for each rebalance date d, estimated ONLY on the `window` months before d."""
    out = {}
    for d in rebalance_dates(rets.index, first):
        past = rets.index < d
        out[d] = rule(rets[past].iloc[-window:], rf[past].iloc[-window:])
    return pd.DataFrame(out).T


def sixty_forty(rets, spx, cost=0.001):
    """60% S&P 500 / 40% 10Y Treasury, rebalanced every January."""
    assets = rets[["10Y Treasury"]].assign(SPX=spx)
    return backtest(assets, pd.Series({"SPX": 0.6, "10Y Treasury": 0.4}), rule="annual", cost=cost)
