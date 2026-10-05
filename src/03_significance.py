"""Module 4: are the differences real?
(a) block-bootstrap confidence interval for Sharpe(strategy) - Sharpe(original)
(b) sensitivity to the estimation window (OOS from 2015 so the 120m window fits)
(c) rebalancing rules claimed in the report (annual + 5% band), after trading costs"""
import numpy as np
import pandas as pd

import optimizers as opt
from backtest import backtest, metrics
from data import ORIGINAL_WEIGHTS, load

rets, spx, rf = load()
w0 = pd.Series(ORIGINAL_WEIGHTS)
rng = np.random.default_rng(0)


def run_walkforward(rule, window, first=120, cost=0.001):
    """first = months of history before the first rebalance (120 -> OOS from 2015, 60 -> from 2010)."""
    dates = [d for d in rets.index[first:] if d.month == 1]
    w = pd.DataFrame({d: rule(rets.loc[:d].iloc[-window - 1:-1]) for d in dates}).T
    return backtest(rets.loc[dates[0]:], w, rule="annual", cost=cost)[0]


def sharpe(x):
    return x.mean() / x.std() * np.sqrt(12)


def bootstrap_sharpe_diff(a, b, n=5000, block=12):
    """Stationary-ish block bootstrap: resample 12-month blocks, keep a & b paired."""
    a, b = a.values, b.values
    T = len(a)
    diffs = []
    for _ in range(n):
        starts = rng.integers(0, T - block, T // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs.append(sharpe(a[idx]) - sharpe(b[idx]))
    return np.percentile(diffs, [2.5, 97.5])


RULES = {
    "Equal Weight": opt.equal_weight,
    "Inverse Vol": opt.inverse_vol,
    "Min Variance": opt.min_variance,
    "Max Sharpe (MVO)": opt.max_sharpe,
    "Max Sharpe (30% cap)": lambda r: opt.max_sharpe(r, cap=0.30),
    "Risk Parity": opt.risk_parity,
    "HRP": opt.hrp,
}

# ---- (b) window sensitivity: Sharpe for 36 / 60 / 120 month windows, common OOS start
rows = {}
for name, rule in RULES.items():
    rows[name] = {f"{w}m window": metrics(run_walkforward(rule, w), rf)["Sharpe"] for w in (36, 60, 120)}
oos_index = run_walkforward(opt.equal_weight, 60).index
orig = backtest(rets.loc[oos_index], w0, rule="annual", cost=0.001)[0]
rows["Original (hand-picked)"] = {f"{w}m window": metrics(orig, rf)["Sharpe"] for w in (36, 60, 120)}
sens = pd.DataFrame(rows).T
print(f"(b) Sharpe by estimation window, OOS {oos_index[0]:%Y-%m} to {oos_index[-1]:%Y-%m}")
print(sens.round(2).to_string(), "\n")
sens.to_csv("results/03_window_sensitivity.csv")

# ---- (a) is any strategy significantly better than the hand-picked portfolio?
# same sample as the main table in 02_walkforward.py: 60m window, OOS from 2010
r0 = run_walkforward(opt.equal_weight, 60, first=60)
orig = backtest(rets.loc[r0.index], w0, rule="annual", cost=0.001)[0]
ex_orig = orig - rf.reindex(orig.index)
ci = {}
for name, rule in RULES.items():
    r = run_walkforward(rule, 60, first=60)
    lo, hi = bootstrap_sharpe_diff(r - rf.reindex(r.index), ex_orig)
    ci[name] = {"Sharpe diff": sharpe(r - rf.reindex(r.index)) - sharpe(ex_orig),
                "95% CI low": lo, "95% CI high": hi, "Significant?": lo > 0 or hi < 0}
ci = pd.DataFrame(ci).T
print(f"(a) Sharpe(strategy) - Sharpe(original), 12-month block bootstrap, OOS {r0.index[0]:%Y-%m} to {r0.index[-1]:%Y-%m}")
print(ci.round(3).to_string(), "\n")
ci.to_csv("results/03_sharpe_bootstrap.csv")

# ---- (c) rebalancing rules for the original weights, full sample, 10bp costs
reb = {}
for rule in ["never", "monthly", "annual", "band", "annual+band"]:
    r, t = backtest(rets, w0, rule=rule, band=0.05, cost=0.001)
    m = metrics(r, rf)
    reb[rule] = {"CAGR": m["CAGR"], "Volatility": m["Volatility"], "Sharpe": m["Sharpe"],
                 "Max Drawdown": m["Max Drawdown"], "Turnover/yr": t.sum() / (len(t) / 12),
                 "Trades": int((t > 0).sum())}
reb = pd.DataFrame(reb).T
print(f"(c) Rebalancing rules, original weights, {rets.index[0]:%Y-%m} to {rets.index[-1]:%Y-%m}")
print(reb.round(3).to_string())
reb.to_csv("results/03_rebalancing.csv")
