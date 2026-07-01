"""
Bangladesh Inflation Forecast (1990-2028)
=========================================
Compares three forecasting approaches for 2027-2028 inflation:

  Model A: AR(1) on log(CPI level) + Monte Carlo on residuals
  Model B: Markov regime-switching (calm vs shock) + Monte Carlo
  Model C: ARIMAX with Brent oil + USD/BDT as exogenous drivers + Monte Carlo

Data sources
------------
  Inflation: World Bank API, FP.CPI.TOTL.ZG (annual %)
  USD/BDT : World Bank API, PA.NUS.FCRF (annual average)
  Brent   : BP Statistical Review / EIA (annual average, USD/bbl)

Run:  python3 bangladesh_inflation_forecast.py
Output: bangladesh_inflation_forecast.png
"""

import json
import urllib.request
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

START_YEAR = 1990
HIST_END_YEAR = 2025
PROJ_YEARS = [2027, 2028]  # 2026 is left as a gap; we project 2027-2028 only
N_SIM = 10_000
RNG = np.random.default_rng(42)

# Annual average Brent (USD/bbl). Source: BP Statistical Review of World Energy
# 2024 (for 1987-2023) + EIA monthly average (for 2024-2025).
BRENT_ANNUAL = {
    1987: 17.97, 1988: 14.24, 1989: 17.98, 1990: 23.76, 1991: 20.04,
    1992: 19.32, 1993: 17.07, 1994: 15.98, 1995: 17.18, 1996: 20.67,
    1997: 19.11, 1998: 12.72, 1999: 17.97, 2000: 28.50, 2001: 24.44,
    2002: 25.02, 2003: 28.85, 2004: 38.27, 2005: 54.52, 2006: 65.39,
    2007: 72.44, 2008: 97.26, 2009: 61.74, 2010: 79.50, 2011: 111.26,
    2012: 111.97, 2013: 108.66, 2014: 98.97, 2015: 52.32, 2016: 43.73,
    2017: 54.19, 2018: 71.31, 2019: 64.21, 2020: 41.85, 2021: 70.86,
    2022: 100.93, 2023: 82.49, 2024: 80.52, 2025: 69.14,
}


def fetch_worldbank(indicator: str, date: str) -> dict:
    url = (
        "https://api.worldbank.org/v2/country/BGD/indicator/"
        f"{indicator}?format=json&date={date}&per_page=200"
    )
    with urllib.request.urlopen(url, timeout=10) as r:
        payload = json.loads(r.read().decode("utf-8"))
    return {
        int(rec["date"]): rec["value"]
        for rec in (payload[1] or [])
        if rec.get("value") is not None
    }


# --- Fallback (verified) data so the script still works fully offline. -------
FALLBACK_INFL = {
    1990: 6.13, 1991: 6.36, 1992: 3.63, 1993: 3.01, 1994: 5.31,
    1995: 10.30, 1996: 2.38, 1997: 5.31, 1998: 8.40, 1999: 6.11,
    2000: 2.21, 2001: 2.01, 2002: 3.33, 2003: 5.67, 2004: 7.59,
    2005: 7.05, 2006: 6.77, 2007: 9.11, 2008: 8.90, 2009: 5.42,
    2010: 8.13, 2011: 11.40, 2012: 6.22, 2013: 7.53, 2014: 6.99,
    2015: 6.19, 2016: 5.51, 2017: 5.70, 2018: 5.54, 2019: 5.59,
    2020: 5.69, 2021: 5.55, 2022: 7.70, 2023: 9.88, 2024: 10.47,
    2025: 8.77,
}
FALLBACK_FX = {
    1990: 34.57, 1991: 36.60, 1992: 38.95, 1993: 39.57, 1994: 40.21,
    1995: 40.28, 1996: 41.79, 1997: 43.89, 1998: 46.91, 1999: 49.09,
    2000: 52.14, 2001: 55.81, 2002: 57.89, 2003: 58.15, 2004: 59.51,
    2005: 64.33, 2006: 68.93, 2007: 68.87, 2008: 68.60, 2009: 69.04,
    2010: 69.65, 2011: 74.15, 2012: 81.86, 2013: 78.10, 2014: 77.64,
    2015: 77.95, 2016: 78.47, 2017: 80.44, 2018: 83.47, 2019: 84.45,
    2020: 84.87, 2021: 85.08, 2022: 91.75, 2023: 106.31, 2024: 115.60,
    2025: 121.92,
}


def load_data():
    try:
        infl = fetch_worldbank("FP.CPI.TOTL.ZG", "1990:2025")
        fx = fetch_worldbank("PA.NUS.FCRF", "1990:2025")
        if not infl:
            raise ValueError("empty")
        print("Loaded live data from World Bank API.")
    except Exception as exc:
        print(f"Live fetch failed ({exc}); using built-in fallback data.")
        infl = dict(FALLBACK_INFL)
        fx = dict(FALLBACK_FX)

    years = sorted(set(range(START_YEAR, HIST_END_YEAR + 1))
                   & set(infl) & set(fx))
    return (
        np.array(years, dtype=int),
        np.array([infl[y] for y in years], dtype=float),       # π in %
        np.array([fx[y] for y in years], dtype=float),         # BDT/USD
        np.array([BRENT_ANNUAL[y] for y in years], dtype=float),
    )


# =============================================================================
# Model A: AR(1) on inflation rate + bootstrap residual MC
# =============================================================================
def model_a_ar1(pi_pct: np.ndarray) -> dict:
    # AR(1) directly on the inflation rate: pi[t] = a + b * pi[t-1] + e[t]
    y = pi_pct[1:]
    x = pi_pct[:-1]
    n = len(y)
    x_mean, y_mean = x.mean(), y.mean()
    b_hat = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
    a_hat = y_mean - b_hat * x_mean
    # Stationarity clamp
    b = max(min(b_hat, 0.95), -0.95)
    a = (1.0 - b) * pi_pct.mean()
    resid = y - (a_hat + b_hat * x)
    sigma = resid.std(ddof=2)

    # Deterministic central forecast
    central = []
    cur = pi_pct[-1]
    for _ in PROJ_YEARS:
        cur = a + b * cur
        central.append(cur)

    # Monte Carlo: resample residuals
    sims = np.empty((N_SIM, len(PROJ_YEARS)))
    for s in range(N_SIM):
        cur = pi_pct[-1]
        for t in range(len(PROJ_YEARS)):
            shock = RNG.choice(resid)
            cur = a + b * cur + shock
            sims[s, t] = cur
    return {
        "name": "A: AR(1) on inflation rate",
        "central": central,
        "sims": sims,
        "params": {"a": a, "b": b, "sigma": sigma, "n": n,
                   "a_raw": a_hat, "b_raw": b_hat},
    }


# =============================================================================
# Model B: Markov regime-switching (calm vs shock) + MC
# =============================================================================
def model_b_regime(pi_pct: np.ndarray) -> dict:
    # Two regimes: calm (low mean, low vol) vs shock (high mean, high vol).
    # Soft-assign by thresholding on inflation (>= 7% = shock).
    shock = pi_pct >= 7.0
    calm = ~shock
    mu_c, sd_c = pi_pct[calm].mean(), pi_pct[calm].std(ddof=1)
    mu_s, sd_s = pi_pct[shock].mean(), pi_pct[shock].std(ddof=1)
    if calm.sum() < 2:
        mu_c, sd_c = 5.0, 1.0
    if shock.sum() < 2:
        mu_s, sd_s = 9.0, 1.5

    # Transition probabilities from history
    p_cs = (shock[1:] & ~shock[:-1]).sum() / max(calm.sum(), 1)
    p_sc = (~shock[1:] & shock[:-1]).sum() / max(shock.sum(), 1)

    # Start: last observed regime
    state = "shock" if shock[-1] else "calm"

    sims = np.empty((N_SIM, len(PROJ_YEARS)))
    for s in range(N_SIM):
        st = state
        for t in range(len(PROJ_YEARS)):
            if st == "calm":
                pi = RNG.normal(mu_c, sd_c)
                if RNG.random() < p_cs:
                    st = "shock"
            else:
                pi = RNG.normal(mu_s, sd_s)
                if RNG.random() < p_sc:
                    st = "calm"
            sims[s, t] = pi
    central = sims.mean(axis=0)
    return {
        "name": "B: Regime-switching (calm vs shock)",
        "central": central,
        "sims": sims,
        "params": {
            "calm_mean": mu_c, "calm_sd": sd_c,
            "shock_mean": mu_s, "shock_sd": sd_s,
            "p_calm_to_shock": p_cs, "p_shock_to_calm": p_sc,
            "n_calm": int(calm.sum()), "n_shock": int(shock.sum()),
        },
    }


# =============================================================================
# Model C: ARIMAX with Brent oil and log USD/BDT as exogenous drivers
# =============================================================================
def model_c_arimax(pi_pct: np.ndarray, fx: np.ndarray,
                   brent: np.ndarray) -> dict:
    # Regressors: Brent level, log(FX) level change, lagged π
    y = pi_pct[1:]
    x_brent = brent[1:]
    x_fx = np.diff(np.log(fx)) * 100.0  # % FX change
    x_lag = pi_pct[:-1]

    X = np.column_stack([np.ones(len(y)), x_brent, x_fx, x_lag])
    # OLS
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = resid.std(ddof=X.shape[1])

    # Exogenous future assumptions: oil mean-reverts, FX continues mild drift
    # Use 2025 values and assume oil = 5-yr average, FX = linear trend of last 3 yrs
    oil_last5 = brent[-5:].mean()
    fx_last3_growth = (np.diff(np.log(fx[-3:])) * 100.0).mean()
    pi_last = pi_pct[-1]

    # Deterministic central forecast
    central = []
    oil_t = oil_last5
    fx_growth_t = fx_last3_growth
    pi_t = pi_last
    for _ in PROJ_YEARS:
        pi_next = (beta[0] + beta[1] * oil_t + beta[2] * fx_growth_t
                   + beta[3] * pi_t)
        central.append(pi_next)
        pi_t = pi_next

    # Monte Carlo
    sims = np.empty((N_SIM, len(PROJ_YEARS)))
    for s in range(N_SIM):
        oil_t = oil_last5
        fx_growth_t = fx_last3_growth
        pi_t = pi_last
        for t in range(len(PROJ_YEARS)):
            shock = RNG.normal(0, sigma)
            pi_next = (beta[0] + beta[1] * oil_t + beta[2] * fx_growth_t
                       + beta[3] * pi_t + shock)
            sims[s, t] = pi_next
            pi_t = pi_next
            # Add small random walk on oil and FX within sim
            oil_t *= 1.0 + RNG.normal(0, 0.05)
            fx_growth_t = fx_growth_t + RNG.normal(0, 1.0)
    return {
        "name": "C: ARIMAX (oil + Δlog(FX) + lagged π)",
        "central": central,
        "sims": sims,
        "params": {
            "beta_intercept": beta[0], "beta_oil": beta[1],
            "beta_dlogfx": beta[2], "beta_lag": beta[3],
            "sigma": sigma, "oil_assumed": oil_last5,
            "fx_growth_assumed": fx_last3_growth,
        },
    }


# =============================================================================
# Plotting
# =============================================================================
def plot_results(years_hist, pi_hist, model_results):
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.plot(years_hist, pi_hist, color="black", linewidth=2,
            marker="o", markersize=4, label="Historical (World Bank)")

    # 2026 vertical separator for the gap year
    ax.axvline(HIST_END_YEAR + 0.5, color="gray", linestyle=":", alpha=0.6)
    ymax = 13
    ax.set_ylim(0, ymax)
    ax.text(HIST_END_YEAR + 0.6, ymax * 0.98,
            "2026\n(no full-year data yet)",
            fontsize=8, style="italic", color="gray", va="top")

    colors = ["#1f77b4", "#d62728", "#2ca02c"]
    proj_years = PROJ_YEARS  # [2027, 2028]
    label_offsets = [10, -22, 10, -22, 10, -22]  # alternate above/below

    for color, res in zip(colors, model_results):
        sims = res["sims"]
        central = res["central"]
        p05 = np.percentile(sims, 5, axis=0)
        p50 = np.percentile(sims, 50, axis=0)
        p95 = np.percentile(sims, 95, axis=0)
        # Connect to 2025 actual (skip 2026)
        x = [HIST_END_YEAR] + list(proj_years)
        y_c = [pi_hist[-1]] + list(central)
        y_lo = [pi_hist[-1]] + list(p05)
        y_hi = [pi_hist[-1]] + list(p95)
        ax.plot(x, y_c, color=color, linewidth=2, label=f"{res['name']} (median)")
        ax.fill_between(x, y_lo, y_hi, color=color, alpha=0.15,
                        label=f"{res['name']} 5–95% band")
        for i, (yr, cval) in enumerate(zip(proj_years, central)):
            off = label_offsets[i + (colors.index(color) * 2)]
            ax.annotate(f"{cval:.1f}%", (yr, cval),
                        textcoords="offset points", xytext=(0, off),
                        ha="center", fontsize=8, color=color, fontweight="bold")

    ax.set_title("Bangladesh Yearly Inflation — Historical + 2027–2028 Forecast "
                 "(3 models, 10,000 Monte Carlo paths each)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Inflation Rate (%)")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    plt.xticks(list(range(START_YEAR, PROJ_YEARS[-1] + 1, 2)),
               rotation=45)
    plt.tight_layout()
    plt.savefig("bangladesh_inflation_forecast.png", dpi=150, bbox_inches="tight")
    plt.show()


def print_summary(model_results):
    print("\n" + "=" * 72)
    print("FORECAST SUMMARY (10,000 Monte Carlo paths per model)")
    print("=" * 72)
    for res in model_results:
        sims = res["sims"]
        print(f"\n{res['name']}")
        for i, yr in enumerate(PROJ_YEARS):
            col = sims[:, i]
            print(f"  {yr}: mean={col.mean():5.2f}%  "
                  f"median={np.median(col):5.2f}%  "
                  f"5%={np.percentile(col, 5):5.2f}%  "
                  f"95%={np.percentile(col, 95):5.2f}%  "
                  f"P(>10%)={100*(col>10).mean():4.1f}%")
        print(f"  params: {res['params']}")


if __name__ == "__main__":
    years, pi, fx, brent = load_data()
    print(f"Loaded {len(years)} years of inflation data "
          f"({years[0]}-{years[-1]})")

    results = [
        model_a_ar1(pi),
        model_b_regime(pi),
        model_c_arimax(pi, fx, brent),
    ]
    print_summary(results)
    plot_results(years, pi, results)
