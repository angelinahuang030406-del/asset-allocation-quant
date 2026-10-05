"""Truncation test: the weights chosen on date d must not change if every row
from d onward is deleted. If they do, the rule is peeking at the future.
Run from the repo root:  python tests/test_no_lookahead.py"""
import sys

import numpy as np

sys.path.insert(0, "src")
import optimizers as opt  # noqa: E402
from data import load  # noqa: E402

WINDOW = 60
rets, _, _ = load()
rules = [opt.equal_weight, opt.inverse_vol, opt.min_variance, opt.max_sharpe,
         opt.risk_parity, opt.hrp]

for d in [d for d in rets.index[WINDOW:] if d.month == 1]:
    full = rets.loc[:d].iloc[-WINDOW - 1:-1]          # what 02_walkforward.py uses
    truncated = rets[rets.index < d].iloc[-WINDOW:]   # data that ends before d
    assert full.index.max() < d, f"window for {d:%Y-%m} includes {d:%Y-%m}"
    for rule in rules:
        assert np.allclose(rule(full), rule(truncated)), f"{rule.__name__} differs at {d:%Y-%m}"

print("OK: no rule uses data from its own rebalance month or later")
