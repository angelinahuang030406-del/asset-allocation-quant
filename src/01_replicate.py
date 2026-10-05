"""Module 1: rebuild the Portfolio Visualizer backtest in Python and check it matches."""
import matplotlib.pyplot as plt
import pandas as pd

from backtest import backtest, metrics
from data import ORIGINAL_WEIGHTS, load

rets, spx, rf = load()
w0 = pd.Series(ORIGINAL_WEIGHTS)

# same setup as the PV report: annual rebalancing, no trading costs
ours, _ = backtest(rets, w0, rule="annual", cost=0.0)

# PV used index data from 2001; compare on the overlapping window
pv = pd.read_csv("data/pv_portfolio1_monthly.csv", index_col=0, parse_dates=True).squeeze()
both = pd.concat([ours.rename("Python (ETFs)"), pv.rename("Portfolio Visualizer")], axis=1, sort=True).dropna()

print(f"Overlap: {both.index[0]:%Y-%m} to {both.index[-1]:%Y-%m} ({len(both)} months)")
print(f"Monthly return correlation: {both.corr().iloc[0, 1]:.3f}")
print(f"Mean abs monthly gap:       {(both.iloc[:, 0] - both.iloc[:, 1]).abs().mean():.2%}")
table = pd.DataFrame({c: metrics(both[c], rf) for c in both}).T
print(table[["CAGR", "Volatility", "Sharpe", "Max Drawdown"]].round(3))
table.to_csv("results/01_replication.csv")

(1 + both).cumprod().plot(figsize=(9, 4.5), title="Replication check: Python vs Portfolio Visualizer")
plt.ylabel("Growth of $1")
plt.tight_layout()
plt.savefig("results/01_replication.png", dpi=150)
