"""Why PFORX can stand in for BNDX (USD-hedged global bonds) before 2013."""
import pandas as pd
import yfinance as yf

px = yf.download(["PFORX", "BNDX"], start="2013-06-01", end="2026-10-01",
                 auto_adjust=True, progress=False)["Close"]
m = px.resample("ME").last().pct_change().dropna()
out = pd.Series({"months": len(m), "correlation": m.corr().iloc[0, 1],
                 "vol BNDX": m["BNDX"].std() * 12 ** 0.5, "vol PFORX": m["PFORX"].std() * 12 ** 0.5})
print(out.round(3).to_string())
out.to_csv("results/00_proxy_check.csv")
