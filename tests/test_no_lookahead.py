"""Look-ahead tests. They call the SAME functions the scripts use (strategies.py, backtest.py),
replace everything from a cut date onward with random noise, and check that nothing
dated before the cut changes. If a rule or the backtest peeks at the future, the noise
leaks backward and the test fails. The scrambled data is also written to a temporary
data/monthly_returns.csv, so a rule that reads the file itself sees the noise too.

Run from the repo root:  python -m pytest tests"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("src"))
from backtest import backtest  # noqa: E402
from data import load  # noqa: E402
from strategies import RULES, rebalance_dates, walkforward_weights  # noqa: E402

rets, _, rf = load()
RAW = pd.read_csv("data/monthly_returns.csv", index_col=0, parse_dates=True)
CUTS = rebalance_dates(rets.index)[1::4]  # a few rebalance dates spread over the sample


def scramble_from(df, cut, seed):
    """Copy of df where every row dated on/after `cut` is replaced by random returns."""
    out = df.copy()
    future = out.index >= cut
    noise = np.random.default_rng(seed).normal(0, 0.05, size=out[future].shape)
    out[future] = noise if out.ndim == 2 else noise.ravel()
    return out


def test_weights_do_not_see_the_future(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    for name, rule in RULES.items():
        base = walkforward_weights(rets, rf, rule)
        for i, cut in enumerate(CUTS):
            # same noise on disk as in memory: run from a temp folder holding the scrambled file
            scramble_from(RAW, cut, i).to_csv(tmp_path / "data" / "monthly_returns.csv")
            monkeypatch.chdir(tmp_path)
            scrambled = walkforward_weights(scramble_from(rets, cut, i), scramble_from(rf, cut, i), rule)
            monkeypatch.undo()
            # weights chosen ON the cut date may only use data from before it
            assert np.allclose(base.loc[:cut], scrambled.loc[:cut]), f"{name} peeks past {cut:%Y-%m}"


def test_backtest_does_not_see_future_targets():
    w = walkforward_weights(rets, rf, RULES["Equal Weight"])
    oos = rets.loc[w.index[0]:]
    for rule in ["annual", "band"]:  # the band rule also reads the target every month
        base, _ = backtest(oos, w, rule=rule)
        for i, cut in enumerate(CUTS):
            w2 = w.copy()
            later = w2.index > cut
            w2[later] = np.random.default_rng(i).dirichlet(np.ones(w.shape[1]), later.sum())
            r2, _ = backtest(oos, w2, rule=rule)
            # targets set AFTER the cut must not affect returns before the next rebalance date
            nxt = w.index[w.index > cut][0]
            assert np.allclose(base[base.index < nxt], r2[r2.index < nxt]), \
                f"backtest ({rule}) uses targets early ({cut:%Y-%m})"

