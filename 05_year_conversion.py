import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "year_multipliers.csv")
EQUIVALENCE_LONG_PATH = os.path.join(SCRIPT_DIR, "year_equivalences.csv")

GBIF_OCCURRENCE_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
GBIF_TAXON_KEY = int(os.getenv("YEAR_CONVERSION_TAXON_KEY", "1"))
GBIF_FACET_LIMIT = int(os.getenv("YEAR_CONVERSION_FACET_LIMIT", "3000"))
GBIF_TIMEOUT_SEC = int(os.getenv("YEAR_CONVERSION_TIMEOUT_SEC", "120"))
GBIF_MAX_RETRIES = int(os.getenv("YEAR_CONVERSION_MAX_RETRIES", "5"))
GBIF_RETRY_BASE_DELAY_SEC = float(os.getenv("YEAR_CONVERSION_RETRY_BASE_DELAY_SEC", "1.5"))

YEAR_START = int(os.getenv("YEAR_CONVERSION_YEAR_START", "1000"))
# Default to 2025 so 2024 extinctions can map to a +1 "after" year and
# still produce year-pair conversions (e.g., 2023 -> 2025).
YEAR_END_INCLUSIVE = int(os.getenv("YEAR_CONVERSION_YEAR_END", "2025"))
YEAR_EDGE_PADDING = int(os.getenv("YEAR_CONVERSION_EDGE_PADDING", "0"))
YEAR_REFERENCE_STAT = os.getenv("YEAR_REFERENCE_STAT", "median").strip().lower()


def fetch_json(params: Dict[str, object]) -> Dict[str, object]:
    url = f"{GBIF_OCCURRENCE_SEARCH_URL}?{urllib.parse.urlencode(params, doseq=True)}"
    headers = {
        "Accept": "application/json",
        "User-Agent": "school-bio-year-conversion/1.0",
    }
    retryable = {429, 500, 502, 503, 504}

    for attempt in range(1, GBIF_MAX_RETRIES + 1):
        req = urllib.request.Request(url=url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=GBIF_TIMEOUT_SEC) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code not in retryable or attempt >= GBIF_MAX_RETRIES:
                raise
            retry_after = err.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(float(retry_after), GBIF_RETRY_BASE_DELAY_SEC)
            else:
                delay = GBIF_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            print(
                f"GBIF HTTP {err.code}; retrying in {delay:.2f}s "
                f"(attempt {attempt}/{GBIF_MAX_RETRIES})"
            )
            time.sleep(delay)
        except urllib.error.URLError:
            if attempt >= GBIF_MAX_RETRIES:
                raise
            delay = GBIF_RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            print(
                f"GBIF network error; retrying in {delay:.2f}s "
                f"(attempt {attempt}/{GBIF_MAX_RETRIES})"
            )
            time.sleep(delay)

    raise RuntimeError("Unexpected GBIF request failure.")


def collect_year_counts_from_api() -> Dict[int, int]:
    params = {
        "taxonKey": GBIF_TAXON_KEY,
        "year": f"{YEAR_START},{YEAR_END_INCLUSIVE}",
        "limit": 0,
        "facet": "year",
        "facetLimit": GBIF_FACET_LIMIT,
    }
    payload = fetch_json(params)
    facets = payload.get("facets", [])
    if not facets:
        raise RuntimeError("GBIF response did not include year facets.")

    counts: Dict[int, int] = {}
    for item in facets[0].get("counts", []):
        year_text = str(item.get("name") or "").strip()
        count_value = item.get("count")
        if not year_text:
            continue
        try:
            year = int(year_text)
            count = int(count_value)
        except (TypeError, ValueError):
            continue
        if year < YEAR_START or year > YEAR_END_INCLUSIVE:
            continue
        counts[year] = count

    if not counts:
        raise RuntimeError("No year counts were returned from GBIF.")

    print(f"GBIF total matching occurrences: {int(payload.get('count', 0))}")
    print(f"GBIF distinct years returned: {len(counts)}")
    print(f"GBIF year range returned: {min(counts)}-{max(counts)}")
    return counts


def build_year_plus_two_equivalences(multiplier_df: pd.DataFrame) -> pd.DataFrame:
    year_to_mult = {}
    year_to_total = {}
    year_to_observed = {}
    for _, row in multiplier_df.iterrows():
        year = int(row["Year"])
        year_to_mult[year] = float(row["Year Multiplier"])
        year_to_total[year] = float(row["Total Occurrences"])
        year_to_observed[year] = bool(row["Observed Year"])

    rows = []
    for from_year in sorted(year_to_total.keys()):
        to_year = from_year + 2
        if to_year not in year_to_total:
            continue
        if not (year_to_observed.get(from_year, False) and year_to_observed.get(to_year, False)):
            continue

        from_total = year_to_total[from_year]
        to_total = year_to_total[to_year]
        ratio_to_from = to_total / from_total if from_total > 0 else np.nan
        ratio_from_to = from_total / to_total if to_total > 0 else np.nan

        m_from = year_to_mult.get(from_year, np.nan)
        m_to = year_to_mult.get(to_year, np.nan)
        equivalent = ratio_to_from

        rows.append(
            {
                "From Year": int(from_year),
                "To Year": int(to_year),
                "Year Gap": 2,
                "From Year Total Occurrences": float(from_total),
                "To Year Total Occurrences": float(to_total),
                "To/From Occurrence Ratio": float(ratio_to_from),
                "From/To Occurrence Ratio": float(ratio_from_to),
                "From Year Multiplier": m_from,
                "To Year Multiplier": m_to,
                "1 Occurrence In From Equals In To": float(equivalent),
            }
        )
    return pd.DataFrame(rows)


def build_continuous_year_totals(year_counts: Dict[int, int]) -> pd.DataFrame:
    years_sorted = sorted(int(y) for y in year_counts.keys())
    if not years_sorted:
        return pd.DataFrame(columns=["Year", "Total Occurrences", "Observed Year"])

    min_year = max(YEAR_START, years_sorted[0] - YEAR_EDGE_PADDING)
    max_year = min(YEAR_END_INCLUSIVE + YEAR_EDGE_PADDING, years_sorted[-1] + YEAR_EDGE_PADDING)
    full_years = pd.Index(range(min_year, max_year + 1), dtype="int64")

    counts_series = pd.Series(
        {int(year): float(count) for year, count in year_counts.items()},
        dtype="float64",
    ).reindex(full_years)

    counts_series = counts_series.interpolate(method="linear", limit_direction="both")
    counts_series = counts_series.ffill().bfill().fillna(1.0).clip(lower=1.0)

    out = pd.DataFrame(
        {
            "Year": full_years.astype(int),
            "Total Occurrences": counts_series.astype(float),
        }
    )
    observed_years = set(years_sorted)
    out["Observed Year"] = out["Year"].isin(observed_years)
    return out


def main() -> None:
    print("Using GBIF API source:")
    print(f"  - endpoint: {GBIF_OCCURRENCE_SEARCH_URL}")
    print(f"  - taxonKey: {GBIF_TAXON_KEY}")
    print(f"  - year range: {YEAR_START}-{YEAR_END_INCLUSIVE}")

    year_counts = collect_year_counts_from_api()
    dense_df = build_continuous_year_totals(year_counts)
    if dense_df.empty:
        raise RuntimeError("Could not build year totals from GBIF response.")

    years_sorted = dense_df["Year"].astype(int).tolist()
    totals = dense_df["Total Occurrences"].astype(float).tolist()
    average_occurrences = float(sum(totals) / len(totals))
    median_occurrences = float(pd.Series(totals).median())
    reference_occurrences = median_occurrences if YEAR_REFERENCE_STAT != "average" else average_occurrences

    df = pd.DataFrame(
        {
            "Year": years_sorted,
            "Total Occurrences": totals,
            "Observed Year": dense_df["Observed Year"].astype(bool).tolist(),
        }
    )
    year_multiplier = float(reference_occurrences) / df["Total Occurrences"].clip(lower=1).astype(float)

    rows = []
    for idx, row in df.iterrows():
        year = int(row["Year"])
        total = int(row["Total Occurrences"])
        rows.append(
            {
                "Year": year,
                "Total Occurrences": total,
                "Average Occurrences Across Years": average_occurrences,
                "Median Occurrences Across Years": median_occurrences,
                "Reference Occurrences Used": float(reference_occurrences),
                "Reference Statistic": "median" if YEAR_REFERENCE_STAT != "average" else "average",
                "Year Multiplier": float(year_multiplier.iloc[idx]),
                "Observed Year": bool(df.iloc[idx]["Observed Year"]),
            }
        )

    output_df = pd.DataFrame(rows)
    output_df.to_csv(OUTPUT_PATH, index=False)

    eq_long_df = build_year_plus_two_equivalences(output_df)
    eq_long_df.to_csv(EQUIVALENCE_LONG_PATH, index=False)

    print(f"\nSaved {len(output_df)} year multipliers to: {OUTPUT_PATH}")
    print(f"Saved year equivalence rows: {len(eq_long_df)} to: {EQUIVALENCE_LONG_PATH}")
    print(f"Reference statistic used: {'median' if YEAR_REFERENCE_STAT != 'average' else 'average'}")
    print(
        "Year Multiplier range: "
        f"{float(output_df['Year Multiplier'].min()):.12f} .. "
        f"{float(output_df['Year Multiplier'].max()):.12f}"
    )
    print(
        "Observed years represented directly: "
        f"{int(output_df['Observed Year'].sum())} / {len(output_df)}"
    )
    print("\nPreview:")
    print(output_df.head(10))


if __name__ == "__main__":
    main()
