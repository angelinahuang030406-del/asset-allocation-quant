"""Download ETF prices and build monthly total-return series."""
import pandas as pd
import yfinance as yf

# asset class -> ETF proxy (same 10 sleeves as the UGBA 133 portfolio)
ASSETS = {
    "US Large Value": "IVE",
    "US Large Growth": "IVW",
    "US Mid Value": "IJJ",
    "US Mid Growth": "IJK",
    "Intl Developed": "EFA",
    "TIPS": "TIP",
    "10Y Treasury": "IEF",
    "Gold": "GLD",
    "Global Bonds (Hedged)": "PFORX",  # BNDX only starts 2013; PFORX corr 0.93
    "REITs": "VNQ",
}
BENCHMARK = "VFINX"  # Vanguard 500 Index, same benchmark as the original report

# the hand-picked weights from the original group project
ORIGINAL_WEIGHTS = {
    "US Large Value": 0.18, "US Large Growth": 0.12,
    "US Mid Value": 0.05, "US Mid Growth": 0.05,
    "Intl Developed": 0.10, "TIPS": 0.15, "10Y Treasury": 0.08,
    "Gold": 0.12, "Global Bonds (Hedged)": 0.05, "REITs": 0.10,
}


def download(path="data/monthly_returns.csv"):
    tickers = list(ASSETS.values()) + [BENCHMARK]
    px = yf.download(tickers, start="2000-01-01", auto_adjust=True, progress=False)["Close"]
    monthly = px.resample("ME").last().pct_change()
    names = {v: k for k, v in ASSETS.items()} | {BENCHMARK: "S&P 500"}
    monthly = monthly.rename(columns=names)
    # keep only months where every asset has data (GLD starts Nov 2004)
    monthly = monthly.dropna()
    monthly = monthly[monthly.index <= pd.Timestamp.today()]  # drop unfinished month
    # risk-free: 13-week T-bill yield (annual %) -> monthly return
    irx = yf.download("^IRX", start="2000-01-01", progress=False)["Close"].squeeze()
    monthly["RF"] = (irx.resample("ME").mean() / 100 / 12).reindex(monthly.index)
    monthly.to_csv(path)
    return monthly


def load(path="data/monthly_returns.csv"):
    """Return (asset returns, S&P 500 returns, risk-free rate)."""
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df[list(ASSETS)], df["S&P 500"], df["RF"]


if __name__ == "__main__":
    r = download()
    print(r.index[0].date(), "to", r.index[-1].date(), len(r), "months")
