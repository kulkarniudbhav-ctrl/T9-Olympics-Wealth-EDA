"""Fetch World Bank WDI indicators used in DATA209 T9 reproducibly.

Olympic source files are downloaded separately from Kaggle and stored under
data/raw/; this script retrieves the WDI source directly from the official API.
"""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from urllib.parse import urlencode


INDICATORS = {
    "NY.GDP.PCAP.CD": "gdp_per_capita_current_usd",
    "SP.POP.TOTL": "population_total",
    "NY.GDP.MKTP.CD": "gdp_current_usd",
    "SH.XPD.CHEX.GD.ZS": "health_expenditure_pct_gdp",
}
OUT = Path(__file__).resolve().parents[1] / "data" / "raw" / "wdi_indicators.csv"


def fetch_indicator(indicator: str) -> list[dict]:
    params = urlencode({"format": "json", "per_page": 20000})
    url = f"https://api.worldbank.org/v2/country/all/indicator/{indicator}?{params}"
    # curl uses the system certificate store. This avoids the macOS framework
    # Python certificate issue while retaining TLS verification.
    response = subprocess.run(
        ["curl", "--fail", "--silent", "--show-error", "--location", url],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(response.stdout)
    return payload[1] if len(payload) > 1 and payload[1] else []


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for code, name in INDICATORS.items():
        for record in fetch_indicator(code):
            # Exclude World Bank aggregate regions; ISO3 country codes are needed for the join.
            iso3 = record.get("countryiso3code") or ""
            if len(iso3) != 3:
                continue
            rows.append({
                "iso3": iso3,
                "country": record["country"]["value"],
                "year": record["date"],
                "indicator_code": code,
                "indicator": name,
                "value": record["value"],
            })
    with OUT.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iso3", "country", "year", "indicator_code", "indicator", "value"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows):,} country-indicator-year observations to {OUT}")


if __name__ == "__main__":
    main()
