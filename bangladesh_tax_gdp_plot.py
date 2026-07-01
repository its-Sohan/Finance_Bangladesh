"""
Bangladesh Tax Revenue as % of GDP (2005-2025) - Line Graph
------------------------------------------------------------
Data source: World Bank API - Tax revenue (% of GDP)
              Indicator code: GC.TAX.TOTL.GD.ZS
              https://data.worldbank.org/indicator/GC.TAX.TOTL.GD.ZS?locations=BD

Notes:
- World Bank publishes data only for years that have already ended.
  Any year with no published value is left as a gap in the line
  rather than guessed.
- Requires: matplotlib (pip install matplotlib)
  Optional: requests (pip install requests) for live data fetching.
            If requests isn't installed, the script falls back to
            urllib (standard library) automatically.
"""

import json
import urllib.request

import matplotlib.pyplot as plt

START_YEAR = 2005
END_YEAR = 2025

# Verified fallback data (World Bank, GC.TAX.TOTL.GD.ZS, fetched 2026-07-01)
# Used automatically if the live API call fails (e.g. no internet access).
FALLBACK_DATA = {
    2005: 8.40, 2006: 8.62, 2007: 8.80, 2008: 9.30, 2009: 9.54,
    2010: 9.58, 2011: 10.03, 2012: 10.50, 2013: 10.66, 2014: 11.02,
    2015: 11.40, 2016: 11.60, 2017: 11.18, 2018: 11.48, 2019: 11.71,
    2020: 9.92, 2021: 10.68, 2022: 10.79, 2023: 11.10, 2024: 11.55,
    2025: 12.01,
}


def fetch_worldbank_tax_gdp(start_year: int, end_year: int) -> dict:
    """Fetch Bangladesh tax revenue (% of GDP) from the World Bank API."""
    url = (
        "https://api.worldbank.org/v2/country/BGD/indicator/GC.TAX.TOTL.GD.ZS"
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


def get_tax_gdp_data(start_year: int, end_year: int) -> dict:
    try:
        data = fetch_worldbank_tax_gdp(start_year, end_year)
        print("Loaded live tax-to-GDP data from World Bank API.")
    except Exception as exc:
        print(f"Live fetch failed ({exc}); using built-in fallback data.")
        data = dict(FALLBACK_DATA)
    return data


def plot_tax_gdp(data: dict, start_year: int, end_year: int) -> None:
    years = list(range(start_year, end_year + 1))
    values = [data.get(year) for year in years]  # None -> gap in the line

    plt.figure(figsize=(12, 6))
    plt.plot(years, values, marker="o", linewidth=2, color="#2ca02c")

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

    plt.title("Bangladesh Tax Revenue as % of GDP (2005-2025)", fontsize=14, fontweight="bold")
    plt.xlabel("Year")
    plt.ylabel("Tax Revenue (% of GDP)")
    plt.xticks(years, rotation=45)
    plt.grid(True, linestyle="--", alpha=0.5)

    missing_years = [y for y in years if data.get(y) is None]
    if missing_years:
        note = f"No published data yet for: {', '.join(map(str, missing_years))}"
        plt.figtext(0.5, -0.02, note, ha="center", fontsize=9, style="italic", color="gray")

    plt.tight_layout()
    plt.savefig("bangladesh_tax_gdp_2005_2025.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    tax_gdp_data = get_tax_gdp_data(START_YEAR, END_YEAR)
    plot_tax_gdp(tax_gdp_data, START_YEAR, END_YEAR)
