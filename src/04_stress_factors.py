"""Module 5: (a) stress-period returns, (b) factor attribution of the original portfolio."""
import pandas as pd
import statsmodels.api as sm

from backtest import backtest
from data import ORIGINAL_WEIGHTS, load

rets, spx, rf = load()
orig, _ = backtest(rets, pd.Series(ORIGINAL_WEIGHTS), rule="annual", cost=0.001)
sixty_forty, _ = backtest(rets[["10Y Treasury"]].assign(SPX=spx),
                          pd.Series({"SPX": 0.6, "10Y Treasury": 0.4}), rule="annual", cost=0.001)

# ---- (a) stress periods (peak-to-trough windows of each episode)
STRESS = {
    "GFC (Nov07-Feb09)": ("2007-11", "2009-02"),
    "Euro crisis (May-Sep 2011)": ("2011-05", "2011-09"),
    "Taper tantrum (May-Aug 2013)": ("2013-05", "2013-08"),
    "Q4 2018 selloff": ("2018-10", "2018-12"),
    "COVID crash (Feb-Mar 2020)": ("2020-02", "2020-03"),
    "2022 inflation + rate hikes": ("2022-01", "2022-09"),
}
series = {"Original": orig, "60/40": sixty_forty, "S&P 500": spx,
          "10Y Treasury": rets["10Y Treasury"], "Gold": rets["Gold"], "TIPS": rets["TIPS"]}
stress = pd.DataFrame({name: {k: (1 + s.loc[a:b]).prod() - 1 for k, s in series.items()}
                       for name, (a, b) in STRESS.items()}).T
print("(a) Cumulative return in stress periods")
print((stress * 100).round(1).to_string(), "\n")
stress.to_csv("results/04_stress.csv")


# ---- (b) factor regression: equity factors (FF5 + momentum) + bond and gold factors
def read_french(path, skip):
    df = pd.read_csv(path, skiprows=skip, index_col=0)
    df = df[df.index.astype(str).str.strip().str.len() == 6]  # monthly rows only (YYYYMM)
    df.index = pd.to_datetime(df.index.astype(str).str.strip(), format="%Y%m") + pd.offsets.MonthEnd(0)
    return df.astype(float) / 100


ff = read_french("data/F-F_Research_Data_5_Factors_2x3.csv", 4)
mom = read_french("data/F-F_Momentum_Factor.csv", 13)
mom.columns = ["MOM"]
X = ff.drop(columns="RF").join(mom)
X["TERM"] = rets["10Y Treasury"] - ff["RF"]   # duration premium
X["GOLD"] = rets["Gold"] - ff["RF"]           # gold premium
X = X.dropna()

y = (orig - ff["RF"]).reindex(X.index).dropna()
X = X.loc[y.index]
model = sm.OLS(y, sm.add_constant(X)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
coef = pd.DataFrame({"coef": model.params, "t-stat": model.tvalues})
coef.loc["const", "coef"] *= 12  # annualize alpha
coef = coef.rename(index={"const": "Alpha (annual)"})
print(f"(b) Factor regression of original portfolio excess returns, {y.index[0]:%Y-%m} to {y.index[-1]:%Y-%m}")
print(coef.round(3).to_string())
print(f"R-squared: {model.rsquared:.3f}")
coef.assign(r2=model.rsquared).to_csv("results/04_factor_regression.csv")
