"""
Bangladesh Yearly Inflation Rate (2005-2026) - Line Graph
------------------------------------------------------------
Data source: World Bank API - Inflation, consumer prices (annual %)
             Indicator code: FP.CPI.TOTL.ZG
             https://data.worldbank.org/indicator/FP.CPI.TOTL.ZG?locations=BD

Notes:
- World Bank publishes data only for years that have already ended.
  As of the time this script was written, official annual inflation
  data for 2026 has NOT been published yet (the year isn't over).
  The script will try to fetch live data; any year with no published
  value (e.g. 2026) is left as a gap in the line rather than guessed.
- Requires: matplotlib (pip install matplotlib)
  Optional: requests (pip install requests) for live data fetching.
            If requests isn't installed, the script falls back to
            urllib (standard library) automatically.
"""

import json
import urllib.request

import matplotlib.pyplot as plt

START_YEAR = 2005
END_YEAR = 2026

# Verified fallback data (World Bank, FP.CPI.TOTL.ZG, fetched 2026-07-01)
# Used automatically if the live API call fails (e.g. no internet access).
FALLBACK_DATA = {
    2005: 7.05, 2006: 6.77, 2007: 9.11, 2008: 8.90, 2009: 5.42,
    2010: 8.13, 2011: 11.40, 2012: 6.22, 2013: 7.53, 2014: 6.99,
    2015: 6.19, 2016: 5.51, 2017: 5.70, 2018: 5.54, 2019: 5.59,
    2020: 5.69, 2021: 5.55, 2022: 7.70, 2023: 9.88, 2024: 10.47,
    2025: 8.77,
    # 2026: not available - year still in progress
}


def fetch_worldbank_inflation(start_year: int, end_year: int) -> dict:
    """Fetch Bangladesh annual inflation (%) from the World Bank API."""
    url = (
        "https://api.worldbank.org/v2/country/BGD/indicator/FP.CPI.TOTL.ZG"
        f"?format=json&date={start_year}:{end_year}&per_page=100"
    )
    with urllib.request.urlopen(url, timeout=10) as response:
        payload = json.loads(response.read().decode("utf-8"))

    records = payload[1] if len(payload) > 1 and payload[1] else []
    data = {}
    for entry in records:
        if entry.get("value") is not None:
            data[int(entry["date"])] = round(float(entry["value"]), 2)
    if not data:
        raise ValueError("No data returned from World Bank API")
    return data


def get_inflation_data(start_year: int, end_year: int) -> dict:
    try:
        data = fetch_worldbank_inflation(start_year, end_year)
        print("Loaded live inflation data from World Bank API.")
    except Exception as exc:
        print(f"Live fetch failed ({exc}); using built-in fallback data.")
        data = dict(FALLBACK_DATA)
    return data


def plot_inflation(data: dict, start_year: int, end_year: int) -> None:
    years = list(range(start_year, end_year + 1))
    values = [data.get(year) for year in years]  # None -> gap in the line

    plt.figure(figsize=(12, 6))
    plt.plot(years, values, marker="o", linewidth=2, color="#1f77b4")

    # Annotate each known point with its value
    for year, val in zip(years, values):
        if val is not None:
            plt.annotate(
                f"{val:.1f}%",
                (year, val),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )

    plt.title("Bangladesh Yearly Inflation Rate (2005-2026)", fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Inflation Rate (%)")
    plt.xticks(years, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)

    missing_years = [y for y in years if data.get(y) is None]
    if missing_years:
        note = f"No published data yet for: {', '.join(map(str, missing_years))}"
        plt.figtext(0.5, -0.02, note, ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig("bangladesh_inflation_2005_2026.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    inflation_data = get_inflation_data(START_YEAR, END_YEAR)
    plot_inflation(inflation_data, START_YEAR, END_YEAR)
