# Bangladesh Finance Analysis

A small set of Python scripts that pull real fiscal/monetary data for
Bangladesh from public sources, plot historical series, and (for the
inflation series) forecast using three different models with Monte Carlo
confidence bands.

## Contents

| File | What it does |
|---|---|
| `bangladesh_inflation_plot.py` | Plots historical yearly inflation 2005-2025 from the World Bank API. |
| `bangladesh_inflation_forecast.py` | Compares three forecasting models (AR(1), regime-switching, ARIMAX) for 2027-2028 with 10,000 Monte Carlo paths each. |
| `bangladesh_tax_gdp_plot.py` | Plots tax revenue as % of GDP 2005-2025 from the World Bank API. |
| `bangladesh_nbr_monthly_yoy_plot.py` | Plots month-over-year growth of NBR monthly tax collection across FY2021-22 to FY2024-25. |
| `bangladesh_inflation_2005_2026.png` | Output plot from the historical inflation script. |
| `bangladesh_inflation_forecast.png` | Output plot from the inflation forecast script (historical line + 3 model fans). |
| `bangladesh_tax_gdp_2005_2025.png` | Output plot from the tax-to-GDP script. |
| `bangladesh_nbr_monthly_yoy_growth.png` | Output plot from the monthly NBR YoY-growth script. |

## Data sources

- **Inflation** (`FP.CPI.TOTL.ZG`) and **USD/BDT** (`PA.NUS.FCRF`): World Bank API
- **Tax revenue (% of GDP)** (`GC.TAX.TOTL.GD.ZS`): World Bank API
- **Monthly NBR tax collection** (BDT crore): NBR monthly press releases
- **Brent oil** (annual average, USD/bbl): BP Statistical Review / EIA

All scripts work offline too — they include a built-in fallback dataset
of the same series if the API is unreachable.

## Running

```bash
pip install matplotlib
python3 bangladesh_inflation_plot.py
python3 bangladesh_inflation_forecast.py
python3 bangladesh_tax_gdp_plot.py
python3 bangladesh_nbr_monthly_yoy_plot.py
```

Both scripts save a PNG and (in interactive environments) display the plot.

## Methodological notes

- **2026 is left as a gap** on the plots — annual inflation data is only
  published after the year ends, and 2026 is still in progress.
- The forecast fan is **5% to 95%** (a 90% confidence band), not 1-sigma.
- "Lower inflation" in the forecast = slower price growth, **not** falling
  prices. See the discussion in the analysis for why this distinction matters.
- The three models are intentionally simple; a professional forecast would
  also fold in IMF WEO projections, Bangladesh Bank policy rate, and
  reserve dynamics.
