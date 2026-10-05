"""Module 6: can the foundation pay out 5% a year for 50 years and keep its real value?
Block-bootstrap historical (return, inflation) pairs into 10,000 fifty-year paths."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest import backtest
from data import ORIGINAL_WEIGHTS, load

YEARS, PAYOUT, N_PATHS, BLOCK = 50, 0.05, 10_000, 12
rng = np.random.default_rng(0)

rets, spx, rf = load()
cpi = pd.read_csv("data/cpi.csv", index_col=0, parse_dates=True).squeeze()
cpi.index = cpi.index + pd.offsets.MonthEnd(0)
inflation = cpi.pct_change(fill_method=None)

ew = pd.Series(1 / rets.shape[1], index=rets.columns)
strategies = {
    "Original (hand-picked)": backtest(rets, pd.Series(ORIGINAL_WEIGHTS), "annual", cost=0.001)[0],
    "Equal Weight": backtest(rets, ew, "annual", cost=0.001)[0],
    "60/40": backtest(rets[["10Y Treasury"]].assign(SPX=spx),
                      pd.Series({"SPX": 0.6, "10Y Treasury": 0.4}), "annual", cost=0.001)[0],
    "S&P 500": spx,
}
data = pd.DataFrame(strategies).join(inflation.rename("CPI")).dropna()
T = len(data)
print(f"Bootstrapping from {data.index[0]:%Y-%m} to {data.index[-1]:%Y-%m} ({T} months)\n")

# same random blocks for every strategy -> a fair, paired comparison
n_months = YEARS * 12
starts = rng.integers(0, T - BLOCK, size=(N_PATHS, n_months // BLOCK))
idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(N_PATHS, -1)

rows, paths = {}, {}
for name in strategies:
    real = (1 + data[name].values[idx]) / (1 + data["CPI"].values[idx]) - 1
    # each month: earn the real return, then pay out 1/12 of 5% of current value
    wealth = np.cumprod((1 + real) * (1 - PAYOUT / 12), axis=1)
    final = wealth[:, -1]
    rows[name] = {
        "Median real value at yr 50": np.median(final),
        "5th pct real value": np.percentile(final, 5),
        "P(real value kept, yr 50)": (final >= 1).mean(),
        "P(ever lose 50% real)": (wealth.min(axis=1) < 0.5).mean(),
    }
    paths[name] = np.percentile(wealth, [5, 50, 95], axis=0)

table = pd.DataFrame(rows).T
print(table.round(3).to_string())
table.to_csv("results/05_foundation_sim.csv")

fig, ax = plt.subplots(figsize=(10, 5))
t = np.arange(1, n_months + 1) / 12
for (name, (p5, p50, p95)), c in zip(paths.items(), plt.cm.tab10.colors):
    ax.plot(t, p50, color=c, lw=2, label=f"{name} (median)")
    ax.fill_between(t, p5, p95, color=c, alpha=0.08)
ax.axhline(1, color="k", ls="--", lw=1)
ax.set(yscale="log", xlabel="Years", ylabel="Real value (start = 1)",
       title="50-year foundation simulation: 5% payout, block bootstrap (5th-95th pct shaded)")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("results/05_foundation_sim.png", dpi=150)
