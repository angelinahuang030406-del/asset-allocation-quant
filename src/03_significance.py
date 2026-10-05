"""Module 4: are the differences real?
(a) moving-block bootstrap confidence interval for Sharpe(strategy) - Sharpe(original)
(b) sensitivity to the estimation window (OOS from 2015 so the 120m window fits)
(c) rebalancing rules claimed in the report (annual + 5% band), after trading costs
(d) how much the 2025-chosen asset menu helps the systematic rules (drop gold)"""
import numpy as np
import pandas as pd

from backtest import backtest, metrics
from data import ORIGINAL_WEIGHTS, load
from strategies import RULES, walkforward_weights

rets, spx, rf = load()
w0 = pd.Series(ORIGINAL_WEIGHTS)
rng = np.random.default_rng(0)
BLOCK = 12


def oos_returns(rule, window=60, first=60, assets=None):
    """Walk-forward returns; first=60 -> OOS from 2010, first=120 -> OOS from 2015."""
    r = rets if assets is None else rets[assets]
    w = walkforward_weights(r, rf, rule, window=window, first=first)
    return backtest(r.loc[w.index[0]:], w, rule="annual", cost=0.001)[0]


def bootstrap_sharpe_diff(a, b, n=5000, block=BLOCK):
    """Moving-block bootstrap: glue together random 12-month blocks (keeps a and b paired,
    keeps within-year autocorrelation). Every block start 0..T-block is equally likely."""
    a, b = a.values, b.values
    T = len(a)
    diffs = []
    for _ in range(n):
        starts = rng.integers(0, T - block + 1, T // block + 1)
        idx = np.concatenate([np.arange(s, s + block) for s in starts])[:T]
        diffs.append(a[idx].mean() / a[idx].std() * np.sqrt(12) - b[idx].mean() / b[idx].std() * np.sqrt(12))
    return np.percentile(diffs, [2.5, 97.5])


# ---- (b) window sensitivity: Sharpe for 36 / 60 / 120 month windows, common OOS start (2015)
rows = {}
for name, rule in RULES.items():
    rows[name] = {f"{w}m window": metrics(oos_returns(rule, w, first=120), rf)["Sharpe"] for w in (36, 60, 120)}
idx15 = oos_returns(RULES["Equal Weight"], 60, first=120).index
orig15 = backtest(rets.loc[idx15], w0, rule="annual", cost=0.001)[0]
rows["Original (hand-picked)"] = {f"{w}m window": metrics(orig15, rf)["Sharpe"] for w in (36, 60, 120)}
sens = pd.DataFrame(rows).T
print(f"(b) Sharpe by estimation window, OOS {idx15[0]:%Y-%m} to {idx15[-1]:%Y-%m}")
print(sens.round(2).to_string(), "\n")
sens.to_csv("results/03_window_sensitivity.csv")

# ---- (a) is any strategy significantly better than the hand-picked portfolio?
# same sample as the main table in 02_walkforward.py: 60m window, OOS from 2010
idx10 = oos_returns(RULES["Equal Weight"]).index
orig = backtest(rets.loc[idx10], w0, rule="annual", cost=0.001)[0]
ex_orig = orig - rf.reindex(idx10)
ci = {}
for name, rule in RULES.items():
    ex = oos_returns(rule) - rf.reindex(idx10)
    lo, hi = bootstrap_sharpe_diff(ex, ex_orig)
    ci[name] = {"Sharpe diff": metrics(ex + rf.reindex(idx10), rf)["Sharpe"] - metrics(orig, rf)["Sharpe"],
                "95% CI low": lo, "95% CI high": hi, "Significant?": lo > 0 or hi < 0}
ci = pd.DataFrame(ci).T
print(f"(a) Sharpe(strategy) - Sharpe(original), {BLOCK}-month block bootstrap, OOS {idx10[0]:%Y-%m} to {idx10[-1]:%Y-%m}")
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
print(reb.round(3).to_string(), "\n")
reb.to_csv("results/03_rebalancing.csv")

# ---- (d) the asset menu (incl. gold) was picked in 2025 with hindsight. Rerun without gold.
no_gold = [a for a in rets.columns if a != "Gold"]
menu = pd.DataFrame({name: {"All 10 assets": metrics(oos_returns(rule), rf)["Sharpe"],
                            "Without gold": metrics(oos_returns(rule, assets=no_gold), rf)["Sharpe"]}
                     for name, rule in RULES.items()}).T
menu["Change"] = menu["Without gold"] - menu["All 10 assets"]
print(f"(d) Sharpe with and without gold in the menu, OOS {idx10[0]:%Y-%m} to {idx10[-1]:%Y-%m}")
print(menu.round(3).to_string())
menu.to_csv("results/03_menu_without_gold.csv")
