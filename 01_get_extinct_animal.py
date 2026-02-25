import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_DIR = Path(__file__).resolve().parents[1]
OUTPUT_FILE = PROJECT_DIR / "animal_names_parsed.csv"
CUDA_OUTPUT_FILE = PROJECT_DIR / "cuda" / "animal_names_parsed.csv"

GBIF_OCCURRENCE_SEARCH_URL = "https://api.gbif.org/v1/occurrence/search"
SPECIES_FACET_LIMIT = int(os.getenv("GBIF_SPECIES_FACET_LIMIT", "200000"))
REQUEST_TIMEOUT_SEC = int(os.getenv("GBIF_TIMEOUT_SEC", "60"))
REQUEST_PAUSE_SEC = float(os.getenv("GBIF_PAUSE_SEC", "0.02"))
MAX_RETRIES = int(os.getenv("GBIF_MAX_RETRIES", "5"))
RETRY_BASE_DELAY_SEC = float(os.getenv("GBIF_RETRY_BASE_DELAY_SEC", "1.5"))

YEAR_START = 1000
YEAR_END_INCLUSIVE = 2024
ALLOWED_CATEGORIES = {"EX", "EW"}


def base_params() -> Dict[str, object]:
    return {
        # GBIF backbone key for Animalia, stricter than kingdom text matching.
        "taxonKey": 1,
        "iucnRedListCategory": sorted(ALLOWED_CATEGORIES),
        "year": f"{YEAR_START},{YEAR_END_INCLUSIVE}",
    }


def build_search_url(params: Dict[str, object]) -> str:
    query = urllib.parse.urlencode(params, doseq=True)
    return f"{GBIF_OCCURRENCE_SEARCH_URL}?{query}"


def fetch_json(params: Dict[str, object]) -> Dict[str, object]:
    url = build_search_url(params=params)
    retryable_codes = {429, 500, 502, 503, 504}
    headers = {
        "Accept": "application/json",
        "User-Agent": "school-bio-extinct-extractor/1.0",
    }

    for attempt in range(1, MAX_RETRIES + 1):
        request = urllib.request.Request(url=url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SEC) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as err:
            if err.code not in retryable_codes or attempt >= MAX_RETRIES:
                raise
            retry_after = err.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                delay = max(float(retry_after), RETRY_BASE_DELAY_SEC)
            else:
                delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            print(
                f"GBIF request failed with HTTP {err.code}; retrying in {delay:.2f}s "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            time.sleep(delay)
        except urllib.error.URLError:
            if attempt >= MAX_RETRIES:
                raise
            delay = RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
            print(
                f"GBIF request network error; retrying in {delay:.2f}s "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            time.sleep(delay)

    raise RuntimeError("Unexpected failure while requesting GBIF API.")


def fetch_species_keys() -> Tuple[List[int], int]:
    payload = fetch_json(
        {
            **base_params(),
            "limit": 0,
            "facet": "speciesKey",
            "facetLimit": SPECIES_FACET_LIMIT,
        }
    )
    total_occurrence_count = int(payload.get("count", 0))
    facets = payload.get("facets", [])
    if not facets:
        return [], total_occurrence_count

    counts = facets[0].get("counts", [])
    species_keys: List[int] = []
    for item in counts:
        raw = str(item.get("name") or "").strip()
        if not raw:
            continue
        try:
            species_keys.append(int(raw))
        except ValueError:
            continue

    return sorted(set(species_keys)), total_occurrence_count


def fetch_latest_occurrence(species_key: int) -> Optional[Dict[str, object]]:
    payload = fetch_json(
        {
            **base_params(),
            "speciesKey": species_key,
            "sort": "year",
            "order": "desc",
            "limit": 1,
        }
    )
    results = payload.get("results", [])
    if not results:
        return None
    return results[0]


def extract_unique_extinct_animals() -> List[Dict[str, object]]:
    unique_species: Dict[str, Dict[str, object]] = {}

    species_keys, total_occurrence_count = fetch_species_keys()
    print(f"GBIF matching occurrences: {total_occurrence_count}")
    print(f"GBIF species keys found: {len(species_keys)}")

    for index, species_key in enumerate(species_keys, start=1):
        item = fetch_latest_occurrence(species_key=species_key)
        if item is None:
            continue

        species = str(item.get("species") or "").strip()
        category = str(item.get("iucnRedListCategory") or "").strip().upper()
        year_raw = item.get("year")

        if not species or category not in ALLOWED_CATEGORIES:
            continue

        try:
            year_int = int(year_raw)
        except (TypeError, ValueError):
            continue

        if year_int < YEAR_START or year_int > YEAR_END_INCLUSIVE:
            continue

        existing = unique_species.get(species)
        if existing is None or year_int > int(existing["last_recorded_year"]):
            unique_species[species] = {
                "last_recorded_year": year_int,
                "iucn_category": category,
            }
        elif year_int == int(existing["last_recorded_year"]):
            if category == "EX" and str(existing["iucn_category"]) != "EX":
                existing["iucn_category"] = "EX"

        if index % 50 == 0:
            print(
                f"Processed {index}/{len(species_keys)} species keys; "
                f"unique species so far: {len(unique_species)}"
            )

        if REQUEST_PAUSE_SEC > 0:
            time.sleep(REQUEST_PAUSE_SEC)

    rows = [
        {
            "scientific_name": name,
            "last_recorded_year": int(data["last_recorded_year"]),
            "iucn_category": str(data["iucn_category"]),
        }
        for name, data in unique_species.items()
    ]
    rows.sort(key=lambda item: (int(item["last_recorded_year"]), str(item["scientific_name"])))
    return rows


def write_csv(rows: List[Dict[str, object]]) -> None:
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["scientific_name", "last_recorded_year", "iucn_category"])
        for row in rows:
            writer.writerow(
                [
                    row["scientific_name"],
                    row["last_recorded_year"],
                    row["iucn_category"],
                ]
            )


def write_cuda_csv(rows: List[Dict[str, object]]) -> None:
    CUDA_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with CUDA_OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Animal Name", "Year"])
        for row in rows:
            writer.writerow([row["scientific_name"], row["last_recorded_year"]])


def main() -> None:
    rows = extract_unique_extinct_animals()
    write_csv(rows)
    write_cuda_csv(rows)

    years = [int(row["last_recorded_year"]) for row in rows]
    min_year = min(years) if years else "n/a"
    max_year = max(years) if years else "n/a"

    print(f"Allowed IUCN categories: {', '.join(sorted(ALLOWED_CATEGORIES))}")
    print(f"Target year range: {YEAR_START}-{YEAR_END_INCLUSIVE}")
    print(f"Wrote {len(rows)} unique extinct animals to {OUTPUT_FILE}")
    print(f"Wrote {len(rows)} rows to {CUDA_OUTPUT_FILE}")
    print(f"Year range in output: {min_year} - {max_year}")


if __name__ == "__main__":
    main()
