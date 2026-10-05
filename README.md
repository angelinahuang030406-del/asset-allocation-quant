# Did Intuition Beat the Optimizer?
### Out-of-sample testing of a discretionary 10-asset foundation portfolio

This project starts from a portfolio my team designed by judgment for a UC Berkeley course (UGBA 133, Investments). The fund is a hypothetical $75M perpetual foundation with a ~7% target return (5% payout + 2% inflation). I rebuilt the backtest in Python and then asked a quant question: **would a systematic allocation rule, estimated only on past data, have done better than our hand-picked weights?**

> The original asset-allocation write-up was a four-person group project. Everything in this repo (replication, walk-forward framework, optimizers, statistical tests, stress/factor analysis, Monte Carlo) is my own extension.

## Key findings

1. **The replication matches.** On 252 overlapping months, the Python/ETF backtest has a 0.999 monthly correlation with the Portfolio Visualizer report used in class (Sharpe 0.66 vs 0.68).
2. **No method reliably beats the hand-picked weights.** Over 2010–2026 out-of-sample, the long-only max-Sharpe portfolio with a 30% cap had the highest Sharpe (0.93 vs 0.78). But the 95% block-bootstrap confidence interval for the Sharpe difference is [−0.18, +0.66], so the gap is not distinguishable from noise. With ~15 years of data, you cannot rank these strategies with confidence.
3. **Unconstrained MVO behaves the way the textbooks warn.** It spends most years in 60–80% global bonds, then jumps to 57% gold in 2023. Its Sharpe also moves from 0.60 to 0.74 to 0.63 when only the estimation window changes (36/60/120 months). A 30% weight cap helps. Min-variance and HRP did worst, because they load up on low-volatility bonds just before the 2022 rate shock.
4. **The portfolio's "downside protection" story does not hold up.** The original write-up said the allocation provides strong downside protection. In the GFC (−32.7% vs −26.9%), the 2011 Euro crisis and the COVID crash, it lost **more** than a plain 60/40. The 10% REIT and 10% international sleeves sold off together with US equities. It only beat 60/40 slightly in Q4 2018 and in 2022.
5. **The ±5% rebalancing band never fired.** The report proposed annual rebalancing plus a ±5% drift trigger. In 22 years, no sleeve drifted 5% between annual rebalances, so "annual + band" is identical to "annual". The choice of rebalancing rule changes Sharpe by less than 0.03.
6. **The returns are almost all factor beta.** Equity + bond + gold factors explain 96% of monthly variance (market beta 0.59, duration 0.29, gold 0.16, value tilt 0.10). The annual alpha is −1.2% (t = −2.9), which reflects fund costs and the REIT/international drag.
7. **5% payout is hard to sustain.** In 10,000 bootstrapped 50-year paths, the original portfolio keeps its real value only 63% of the time, about the same as 60/40 (65%).

## Out-of-sample results (2010-01 to 2026-09, 10bp trading costs)

Each January, weights are estimated using only the previous 60 months, then held for the year.

| Strategy | CAGR | Vol | Sharpe | Max DD | Turnover/yr |
|:--|--:|--:|--:|--:|--:|
| Max Sharpe (30% cap) | 8.0% | 7.0% | 0.93 | −16.7% | 19% |
| 60/40 | 9.6% | 8.8% | 0.92 | −20.6% | 3% |
| S&P 500 | 14.1% | 14.4% | 0.89 | −24.0% | 0% |
| Max Sharpe (unconstrained) | 8.2% | 7.7% | 0.89 | −17.6% | 20% |
| Inverse Vol | 6.6% | 6.5% | 0.78 | −16.1% | 4% |
| **Original (hand-picked)** | **8.8%** | **9.6%** | **0.78** | **−19.5%** | **4%** |
| Risk Parity | 6.2% | 6.1% | 0.78 | −15.9% | 5% |
| Equal Weight | 8.7% | 9.7% | 0.76 | −19.6% | 4% |
| HRP | 4.7% | 4.6% | 0.69 | −13.4% | 8% |
| Min Variance | 4.8% | 5.0% | 0.67 | −14.3% | 11% |

![wealth](results/02_wealth.png)
![weights](results/02_weights.png)

## A caveat about the baseline

The original weights were chosen in late 2025 by people who had already seen the 2001–2025 returns. Its "out-of-sample" row therefore has **look-ahead bias in its favor**, and it still did not come out on top. The systematic rules have no such advantage, because each one only sees data from before its rebalance date.

## Method

| Step | Script | What it does |
|:--|:--|:--|
| Data | `src/data.py` | ETF proxies via yfinance, monthly total returns, 13-week T-bill as risk-free |
| 1. Replication | `src/01_replicate.py` | Rebuild the course backtest and compare it month by month with Portfolio Visualizer |
| 2–3. Walk-forward | `src/02_walkforward.py` | 7 allocation rules, annual re-estimation on a rolling 60-month window, 10bp costs |
| 4. Robustness | `src/03_robustness.py` | Block-bootstrap CI on Sharpe differences, window sensitivity, rebalancing rules |
| 5. Stress + factors | `src/04_stress_factors.py` | Returns in 6 crisis windows; FF5 + momentum + term + gold regression (HAC errors) |
| 6. Foundation sim | `src/05_foundation_sim.py` | 10,000 × 50-year paths, joint (return, CPI) block bootstrap, 5% payout |

Allocation rules (`src/optimizers.py`): equal weight, inverse volatility, minimum variance, max Sharpe (Markowitz, with and without a 30% cap), equal-risk-contribution risk parity, and Hierarchical Risk Parity (López de Prado, 2016). All are long-only and fully invested.

**Asset proxies:** IVE, IVW, IJJ, IJK (US large/mid value/growth), EFA, TIP, IEF, GLD, VNQ, and PFORX for USD-hedged global bonds (BNDX starts only in 2013; the two have a 0.93 monthly correlation over the overlap). The common sample starts in Dec 2004 because GLD launched then. The benchmark is VFINX.

## Limitations

- Only ~22 years of data, and one dominant regime (strong US equities, falling rates until 2021). Every bootstrap result assumes the future resamples this past. The S&P 500 numbers in the 50-year simulation are optimistic for exactly this reason.
- Mean returns are estimated from trailing sample averages, which is the weakest possible input for MVO. Shrinkage estimators or Black–Litterman are natural next steps.
- ETF expense ratios are inside the returns. Taxes and market impact are ignored.

## Run it

```bash
pip install -r requirements.txt
python src/data.py              # refresh prices
python src/01_replicate.py
python src/02_walkforward.py
python src/03_robustness.py
python src/04_stress_factors.py
python src/05_foundation_sim.py
```

Run the scripts from the repo root. Fama–French factors are from Ken French's data library, and CPI is FRED `CPIAUCSL`.
