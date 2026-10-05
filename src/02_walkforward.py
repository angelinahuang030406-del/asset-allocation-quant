"""Modules 2-3: compare the hand-picked portfolio with systematic allocation rules,
estimated walk-forward (each January, using only the previous 60 months)."""
import matplotlib.pyplot as plt
import pandas as pd

from backtest import backtest, metrics
from data import ORIGINAL_WEIGHTS, load
from strategies import RULES, rebalance_dates, sixty_forty, walkforward_weights

WINDOW = 60       # months of history used to estimate each year's weights
COST = 0.001      # 10bp per $ traded

rets, spx, rf = load()
oos = rets.loc[rebalance_dates(rets.index)[0]:]

results, weights, turnover = {}, {}, {}
for name, rule in RULES.items():
    w = walkforward_weights(rets, rf, rule, window=WINDOW)
    weights[name] = w
    results[name], turnover[name] = backtest(oos, w, rule="annual", cost=COST)

# benchmarks
results["Original (hand-picked)"], turnover["Original (hand-picked)"] = backtest(
    oos, pd.Series(ORIGINAL_WEIGHTS), rule="annual", cost=COST)
results["60/40"], turnover["60/40"] = sixty_forty(oos, spx.loc[oos.index], cost=COST)
results["S&P 500"] = spx.loc[oos.index]
turnover["S&P 500"] = pd.Series(0.0, index=oos.index)

table = pd.DataFrame({k: metrics(v, rf, spx) for k, v in results.items()}).T
table["Avg Annual Turnover"] = pd.Series({k: v.sum() / (len(v) / 12) for k, v in turnover.items()})
table = table.sort_values("Sharpe", ascending=False)
print(f"Out-of-sample: {oos.index[0]:%Y-%m} to {oos.index[-1]:%Y-%m}\n")
print(table[["CAGR", "Volatility", "Sharpe", "Max Drawdown", "Calmar", "Avg Annual Turnover"]].round(3).to_string())
table.to_csv("results/02_walkforward_metrics.csv")
pd.concat(weights).to_csv("results/02_walkforward_weights.csv")

# sub-periods. NOTE: these split points were chosen AFTER seeing what happened in 2022
bonds = ["Global Bonds (Hedged)", "10Y Treasury", "TIPS"]
sub = pd.DataFrame({k: {
    "Sharpe 2010-2021": metrics(v.loc[:"2021"], rf)["Sharpe"],
    "Return 2022": (1 + v.loc["2022"]).prod() - 1,
    "Sharpe 2023-": metrics(v.loc["2023":], rf)["Sharpe"],
    "Bond+TIPS weight Jan 2022": weights[k].loc["2022-01-31", bonds].sum() if k in weights else None,
} for k, v in results.items()}).T
print("\nSub-periods")
print(sub.round(2).to_string())
sub.to_csv("results/02_subperiods.csv")

# wealth curves
fig, ax = plt.subplots(figsize=(10, 5))
for k, v in results.items():
    lw = 2.5 if k in ("Original (hand-picked)", "S&P 500") else 1.2
    ax.plot((1 + v).cumprod(), label=k, lw=lw)
ax.set(title="Walk-forward out-of-sample growth of $1", ylabel="Growth of $1", yscale="log")
ax.legend(fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("results/02_wealth.png", dpi=150)

# how much do the weights jump around year to year?
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
for ax, name in zip(axes, ["Max Sharpe (no cap)", "Risk Parity", "HRP"]):
    weights[name].index = weights[name].index.year
    weights[name].plot.bar(stacked=True, ax=ax, legend=False, width=0.85)
    ax.set_title(name)
axes[-1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.suptitle("Weights chosen each January (estimated on prior 60 months)")
fig.tight_layout()
fig.savefig("results/02_weights.png", dpi=150)
