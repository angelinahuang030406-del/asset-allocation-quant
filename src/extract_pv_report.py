"""How data/pv_portfolio1_monthly.csv was made: the monthly returns table for Portfolio 1
(page 8) of the Portfolio Visualizer report used in the course, exported 2025-12-11.
The PDF itself is not in the repo. Usage: python src/extract_pv_report.py path/to/report.pdf"""
import re
import sys

import pandas as pd
import pypdf

page = pypdf.PdfReader(sys.argv[1]).pages[7].extract_text()
rows = {}
for line in page.split("\n"):
    m = re.match(r"^(20\d\d) (.*)", line)
    if not m:
        continue
    year = int(m.group(1))
    vals = [float(v) for v in re.findall(r"(-?\d+\.\d+)%", m.group(2))]
    months = 11 if year == 2025 else 12  # the report ends Nov 2025; extra columns are totals
    for i, v in enumerate(vals[:months]):
        rows[pd.Timestamp(year, i + 1, 1) + pd.offsets.MonthEnd(0)] = v / 100
s = pd.Series(rows, name="pv_portfolio1")
s.index.name = "date"
s.to_csv("data/pv_portfolio1_monthly.csv")
print(len(s), "months written")
