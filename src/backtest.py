"""Monthly backtest with rebalancing rules + performance metrics."""
import numpy as np
import pandas as pd


def backtest(returns, target, rule="annual", band=0.05, cost=0.001):
    """Simulate a portfolio held at `target` weights.

    returns : DataFrame of monthly asset returns
    target  : Series of target weights, or DataFrame of weights per month
              (one row per rebalance date, used by the walk-forward test)
    rule    : "monthly" | "annual" | "band" | "annual+band" | "never"
    band    : drift threshold for the band rules (absolute, e.g. 5%)
    cost    : one-way transaction cost per $ traded (0.1% = 10bp)
    """
    w = None
    out, turnover = [], []
    for date, r in returns.iterrows():
        tgt = target.loc[:date].iloc[-1] if isinstance(target, pd.DataFrame) else target
        tgt = tgt.reindex(returns.columns)
        assert tgt.notna().all() and abs(tgt.sum() - 1) < 1e-6, f"bad target weights at {date:%Y-%m}"

        if w is None:
            w, trade = tgt.copy(), tgt.abs().sum()  # building the portfolio from cash also costs
        else:
            drift = (w - tgt).abs().max()
            is_jan = date.month == 1  # rebalance at start of January
            new_target = isinstance(target, pd.DataFrame) and date in target.index
            do = (rule == "monthly"
                  or (rule == "annual" and is_jan)
                  or (rule == "band" and drift > band)
                  or (rule == "annual+band" and (is_jan or drift > band))
                  or new_target)
            trade = (w - tgt).abs().sum() if do else 0.0
            if do:
                w = tgt.copy()

        port_r = (w * r).sum() - trade * cost
        out.append(port_r)
        turnover.append(trade / 2 if len(out) > 1 else 0.0)  # one-way; the initial build is not turnover
        w = w * (1 + r) / (1 + (w * r).sum())  # weights drift with prices

    return pd.Series(out, index=returns.index), pd.Series(turnover, index=returns.index)


def max_drawdown(r):
    """Worst month-end drop from a previous peak (the starting $1 counts as a peak)."""
    wealth = (1 + r).cumprod()
    return (wealth / wealth.cummax().clip(lower=1) - 1).min()


def metrics(r, rf=None, bench=None):
    """Annualized stats for a monthly return series."""
    if rf is None:
        rf = 0.0
    else:
        rf = rf.reindex(r.index)
        assert rf.notna().all(), "missing risk-free rate"
    excess = r - rf
    years = len(r) / 12
    cagr = (1 + r).prod() ** (1 / years) - 1
    vol = r.std() * np.sqrt(12)
    out = {
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": excess.mean() / excess.std() * np.sqrt(12),
        "Max Drawdown": max_drawdown(r),
        "Calmar": cagr / -max_drawdown(r),
    }
    if bench is not None:
        b = bench.reindex(r.index)
        out["Beta"] = np.cov(r, b)[0, 1] / b.var()
        out["Corr to S&P"] = r.corr(b)
    return pd.Series(out)
