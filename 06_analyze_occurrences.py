import os
import re
import zipfile
import json
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
ANIMALS_PATH = os.path.join(SCRIPT_DIR, "animal_names_parsed.csv")
BROAD_DIET_PATH = os.path.join(SCRIPT_DIR, "animal_diet_info.csv")
SPECIFIC_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "animal_names_info_specific_names.csv"),
    os.path.join(SCRIPT_DIR, "animal_diet_info_specific_names.csv"),
]
OCCURRENCE_ZIP_CANDIDATES = [
    os.path.join(SCRIPT_DIR, "data", "animal data.zip"),
    os.path.join(SCRIPT_DIR, "data", "extinct animal data.zip"),
    os.path.join(SCRIPT_DIR, "data", "obis_occurrence.zip"),
    os.path.join(SCRIPT_DIR, "data", "inat_occurrence.zip"),
]
OCCURRENCE_FILE = "occurrence.txt"
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "raw nums.csv")
SOURCE_CONTRIB_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "groups", "occurrence_source_contributions.csv")
YEAR_MULTIPLIERS_PATH = os.path.join(SCRIPT_DIR, "year_multipliers.csv")
GBIF_CACHE_PATH = os.path.join(SCRIPT_DIR, "gbif_match_cache.json")
GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GBIF_CACHE_STRATEGY = "strict_v2"
OCCURRENCE_ZIP_PATHS_ENV = os.getenv("OCCURRENCE_ZIP_PATHS", "").strip()
GBIF_BROAD_TAXON_EXPANSION = os.getenv("GBIF_BROAD_TAXON_EXPANSION", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
GBIF_MIN_CONFIDENCE_EXPANSION = int(os.getenv("GBIF_MIN_CONFIDENCE_EXPANSION", "70"))

CHUNK_SIZE = int(os.getenv("ANALYZE_CHUNK_SIZE", "200000"))
GBIF_TIMEOUT_SEC = int(os.getenv("GBIF_TIMEOUT_SEC", "20"))
GBIF_PAUSE_SEC = float(os.getenv("GBIF_PAUSE_SEC", "0.01"))
UNKNOWN = "Unknown"
MISSING_VALUES = {"", "nan", "none", "null"}
TAXON_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z-]*")
TAXON_COLUMNS = ["species", "genus", "family", "order", "class", "phylum"]
STRICT_REGION_COLUMNS = ["countryCode", "level1Name", "stateProvince"]
GBIF_ACCEPTED_RANKS = {"SPECIES", "SUBSPECIES", "GENUS"}
GBIF_EXACT_MIN_CONFIDENCE = int(os.getenv("GBIF_EXACT_MIN_CONFIDENCE", "80"))
GBIF_FUZZY_MIN_CONFIDENCE = int(os.getenv("GBIF_FUZZY_MIN_CONFIDENCE", "95"))
NON_TAXON_PATTERNS = [
    r"\bunknown\b",
    r"\bno name\b",
    r"\bdetritus\b",
    r"\borganic matter\b",
    r"\borganic detritus\b",
    r"\bsediment\b",
    r"\bsilt\b",
    r"\bvector\b",
    r"\bvectors\b",
    r"\bflora\b",
    r"\bhuman\b",
    r"\bhumans\b",
    r"\bhomo sapiens\b",
    r"\bhomo\b",
    r"\bhominidae\b",
]
TERM_REMAP = {
    "spiders": ["araneae"],
    "ants": ["formicidae"],
    "beetles": ["coleoptera"],
    "flies": ["diptera"],
    "moths": ["lepidoptera"],
    "snakes": ["serpentes"],
    "frogs": ["anura"],
    "toads": ["bufonidae"],
    "birds": ["aves"],
    "mammals": ["mammalia"],
    "reptiles": ["reptilia"],
    "amphibians": ["amphibia"],
    "fish": ["actinopterygii"],
    "invertebrates": ["arthropoda"],
    "crustaceans": ["crustacea"],
    "mollusks": ["mollusca"],
    "annelids": ["annelida"],
    "plants": ["plantae"],
    "fungi": ["fungi"],
}


def find_specific_path() -> str:
    existing = [candidate for candidate in SPECIFIC_CANDIDATES if os.path.exists(candidate)]
    if existing:
        # Prefer the most recently generated specific-name file.
        return max(existing, key=os.path.getmtime)
    raise FileNotFoundError(
        "Could not find specific-name source file. "
        "Expected one of: animal_names_info_specific_names.csv, animal_diet_info_specific_names.csv"
    )


def find_occurrence_zips() -> List[str]:
    candidates: List[str] = []
    if OCCURRENCE_ZIP_PATHS_ENV:
        for part in re.split(r"[;,]", OCCURRENCE_ZIP_PATHS_ENV):
            path = part.strip().strip('"').strip("'")
            if path:
                candidates.append(path)
    candidates.extend(OCCURRENCE_ZIP_CANDIDATES)

    found: List[str] = []
    seen = set()
    for candidate in candidates:
        full = os.path.abspath(candidate)
        key = normalize_key(full)
        if not key or key in seen:
            continue
        seen.add(key)
        if os.path.exists(full):
            found.append(full)

    if not found:
        raise FileNotFoundError(
            "Could not find any occurrence zip sources. "
            "Set OCCURRENCE_ZIP_PATHS or place at least one zip in data/ "
            "(animal data.zip, extinct animal data.zip, obis_occurrence.zip, inat_occurrence.zip)."
        )
    return found


def as_clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if text.lower() in MISSING_VALUES:
        return ""
    return text


def parse_year(value: Any) -> Optional[int]:
    text = as_clean_text(value)
    if not text:
        return None
    try:
        return int(float(text))
    except Exception:
        return None


def normalize_key(value: Any) -> str:
    text = as_clean_text(value)
    if not text:
        return ""
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def load_gbif_cache() -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(GBIF_CACHE_PATH):
        return {}
    try:
        with open(GBIF_CACHE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def save_gbif_cache(cache: Dict[str, Dict[str, Any]]) -> None:
    try:
        with open(GBIF_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=True, indent=2, sort_keys=True)
    except Exception:
        pass


def is_non_taxon_term(term: str) -> bool:
    t = normalize_key(term)
    if not t:
        return True
    for pattern in NON_TAXON_PATTERNS:
        if re.search(pattern, t):
            return True
    return False


def gbif_match(name: str) -> Dict[str, Any]:
    params = urllib.parse.urlencode({"name": name, "verbose": "true"})
    url = f"{GBIF_MATCH_URL}?{params}"
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "animal-occurrence-analyzer/1.0", "Accept": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=GBIF_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def gbif_keys_from_match(match: Dict[str, Any]) -> Set[str]:
    keys: Set[str] = set()

    def add_key(value: Any) -> None:
        norm = normalize_key(value)
        if norm and len(norm) >= 3:
            keys.add(norm)
        s_key, g_key = parse_taxon_keys(value)
        if s_key:
            keys.add(s_key)
        if g_key and len(g_key) >= 3:
            keys.add(g_key)

    for field in [
        "canonicalName",
        "scientificName",
        "species",
        "genus",
        "family",
        "order",
        "class",
        "phylum",
        "kingdom",
    ]:
        add_key(match.get(field, ""))

    for alt in match.get("alternatives", []) or []:
        if not isinstance(alt, dict):
            continue
        for field in [
            "canonicalName",
            "scientificName",
            "species",
            "genus",
            "family",
            "order",
            "class",
            "phylum",
            "kingdom",
        ]:
            add_key(alt.get(field, ""))

    return keys


def parse_taxon_keys(name: Any) -> Tuple[str, str]:
    text = as_clean_text(name)
    if not text:
        return "", ""

    text = re.sub(r"\([^)]*\)", " ", text)
    tokens = TAXON_TOKEN_RE.findall(text)
    if not tokens:
        return "", ""

    genus = tokens[0].lower()
    species = ""
    if len(tokens) > 1 and tokens[1][:1].islower():
        species = f"{genus} {tokens[1].lower()}"

    return species, genus


def looks_like_species_name(name: str) -> bool:
    text = as_clean_text(name)
    if not text:
        return False
    text = re.sub(r"\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return bool(re.match(r"^[A-Z][A-Za-z-]+\s+[a-z][A-Za-z-]+", text))


def term_search_keys(term: str) -> Set[str]:
    out: Set[str] = set()

    norm_full = normalize_key(term)
    if norm_full and len(norm_full) >= 6:
        out.add(norm_full)

    species_key, genus_key = parse_taxon_keys(term)
    if species_key:
        out.add(species_key)
    if genus_key and len(genus_key) >= 4:
        out.add(genus_key)

    return out


def resolve_term_keys(term: str, gbif_cache: Dict[str, Dict[str, Any]]) -> Tuple[Set[str], str]:
    normalized = normalize_key(term)
    if not normalized:
        return set(), "empty"

    if is_non_taxon_term(term):
        return set(), "filtered_non_taxon"

    if normalized in TERM_REMAP:
        return set(TERM_REMAP[normalized]), "remapped"

    if normalized in gbif_cache:
        cached = gbif_cache[normalized]
        if cached.get("strategy") == GBIF_CACHE_STRATEGY:
            return set(cached.get("keys", [])), str(cached.get("source", "cached"))

    keys = term_search_keys(term)
    source = "parsed_only"
    try:
        match = gbif_match(term)
        match_type = str(match.get("matchType", "NONE")).upper()
        rank = str(match.get("rank", "")).upper()
        confidence = int(match.get("confidence") or 0)

        should_use = False
        if match_type == "EXACT" and confidence >= GBIF_EXACT_MIN_CONFIDENCE and rank in GBIF_ACCEPTED_RANKS:
            should_use = True
        elif match_type == "FUZZY" and confidence >= GBIF_FUZZY_MIN_CONFIDENCE and rank in GBIF_ACCEPTED_RANKS:
            should_use = True
        elif (
            match_type == "HIGHERRANK"
            and rank == "GENUS"
            and confidence >= GBIF_FUZZY_MIN_CONFIDENCE
        ):
            should_use = True

        if GBIF_BROAD_TAXON_EXPANSION and match_type in {"EXACT", "FUZZY", "HIGHERRANK"}:
            gbif_keys = (
                gbif_keys_from_match(match)
                if confidence >= max(0, GBIF_MIN_CONFIDENCE_EXPANSION)
                else set()
            )
        else:
            gbif_keys = gbif_keys_from_match(match) if should_use else set()
        if gbif_keys:
            keys.update(gbif_keys)
            source = f"gbif_{match_type.lower()}_{confidence}"
    except Exception:
        source = "parsed_only"

    gbif_cache[normalized] = {
        "keys": sorted(keys),
        "source": source,
        "strategy": GBIF_CACHE_STRATEGY,
    }
    time.sleep(GBIF_PAUSE_SEC)
    return keys, source


def split_specific_terms(
    value: Any, gbif_cache: Dict[str, Dict[str, Any]]
) -> Tuple[Set[str], List[str], List[str]]:
    search_keys: Set[str] = set()
    display_terms: List[str] = []
    skipped_terms: List[str] = []

    raw = as_clean_text(value)
    if not raw or raw.lower() == "unknown":
        return search_keys, display_terms, skipped_terms

    for part in raw.split(";"):
        term = as_clean_text(part)
        if not term or term.lower() == "unknown":
            continue
        keys, source = resolve_term_keys(term, gbif_cache)
        if keys:
            search_keys.update(keys)
            display_terms.append(term)
        else:
            skipped_terms.append(f"{term} ({source})")

    return search_keys, display_terms, skipped_terms


def split_broad_terms(
    value: Any, gbif_cache: Dict[str, Dict[str, Any]]
) -> Tuple[Set[str], List[str], List[str]]:
    search_keys: Set[str] = set()
    display_terms: List[str] = []
    skipped_terms: List[str] = []

    raw = as_clean_text(value)
    if not raw or raw.lower() == "unknown":
        return search_keys, display_terms, skipped_terms

    for part in raw.split(","):
        term = as_clean_text(part)
        if not term or term.lower() == "unknown":
            continue
        keys, source = resolve_term_keys(term, gbif_cache)
        if keys:
            search_keys.update(keys)
            display_terms.append(term)
        else:
            skipped_terms.append(f"{term} ({source})")

    return search_keys, display_terms, skipped_terms


def normalize_taxon_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"\([^)]*\)", " ", regex=True)
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def extract_binomial_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.extract(r"^([a-z][a-z-]* [a-z][a-z-]*)", expand=False)
        .fillna("")
    )


def extract_genus_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.extract(r"^([a-z][a-z-]*)", expand=False)
        .fillna("")
    )


def normalize_region_series(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def build_region_key_series(df: pd.DataFrame) -> pd.Series:
    country = normalize_region_series(df.get("countryCode", pd.Series(index=df.index, dtype=str)))
    level1 = normalize_region_series(df.get("level1Name", pd.Series(index=df.index, dtype=str)))
    state = normalize_region_series(df.get("stateProvince", pd.Series(index=df.index, dtype=str)))

    # Strict key: country + first-level region fallback.
    subregion = level1.where(level1 != "", state)
    region_key = country + "|" + subregion
    return region_key.where((country != "") & (subregion != ""), "")


def format_region_keys(keys: Set[str], limit: int = 20) -> str:
    if not keys:
        return UNKNOWN
    sorted_keys = sorted(keys)
    if len(sorted_keys) <= limit:
        return "; ".join(sorted_keys)
    extra = len(sorted_keys) - limit
    return "; ".join(sorted_keys[:limit]) + f"; ... (+{extra} more)"


def format_region_count_map(counts_by_region: Dict[str, int], limit: int = 30) -> str:
    if not counts_by_region:
        return UNKNOWN
    items = sorted(counts_by_region.items(), key=lambda kv: (-int(kv[1]), kv[0]))
    shown = items[:limit]
    text = "; ".join(f"{region}:{int(count)}" for region, count in shown)
    extra = len(items) - len(shown)
    if extra > 0:
        text = f"{text}; ... (+{extra} more)"
    return text

# gets year multipliers from year_multipliers.csv
def load_year_multipliers() -> Dict[int, float]:
    if not os.path.exists(YEAR_MULTIPLIERS_PATH):
        return {}

    try:
        df = pd.read_csv(YEAR_MULTIPLIERS_PATH)
    except Exception:
        return {}

    if "Year" not in df.columns or "Year Multiplier" not in df.columns:
        return {}

    year_series = pd.to_numeric(df["Year"], errors="coerce")
    multiplier_series = pd.to_numeric(df["Year Multiplier"], errors="coerce")
    valid = year_series.notna() & multiplier_series.notna()
    year_series = year_series[valid].astype("int64")
    multiplier_series = multiplier_series[valid].astype("float64")

    return {int(year): float(mult) for year, mult in zip(year_series, multiplier_series)}


def load_animals(specific_path: str, gbif_cache: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    df_animals = pd.read_csv(ANIMALS_PATH)
    df_specific = pd.read_csv(specific_path)
    df_broad = pd.read_csv(BROAD_DIET_PATH) if os.path.exists(BROAD_DIET_PATH) else pd.DataFrame()

    specific_lookup: Dict[str, Dict[str, str]] = {}
    for _, row in df_specific.iterrows():
        animal_name = as_clean_text(row.get("Animal Name"))
        if not animal_name or animal_name in specific_lookup:
            continue
        specific_lookup[animal_name] = {
            "predators": as_clean_text(row.get("Specific Predators (Names)", "")),
            "prey": as_clean_text(row.get("Specific Prey (Names)", "")),
        }

    broad_lookup: Dict[str, Dict[str, str]] = {}
    if not df_broad.empty:
        for _, row in df_broad.iterrows():
            animal_name = as_clean_text(row.get("Animal Name"))
            if not animal_name or animal_name in broad_lookup:
                continue
            broad_lookup[animal_name] = {
                "predators": as_clean_text(row.get("What Eats It (Predators)", "")),
                "prey": as_clean_text(row.get("What It Eats (Prey/Diet)", "")),
            }

    rows: List[Dict[str, Any]] = []
    skipped_non_species = 0
    for _, row in df_animals.iterrows():
        animal_name = as_clean_text(row.get("Animal Name"))
        extinction_year = parse_year(row.get("Year"))
        if not animal_name or extinction_year is None:
            continue
        if not looks_like_species_name(animal_name):
            skipped_non_species += 1
            continue
        animal_species_key, animal_genus_key = parse_taxon_keys(animal_name)
        animal_name_key = normalize_key(animal_name)

        spec = specific_lookup.get(animal_name, {"predators": "", "prey": ""})
        broad = broad_lookup.get(animal_name, {"predators": "", "prey": ""})

        prey_keys_specific, prey_terms_specific, prey_skipped_specific = split_specific_terms(
            spec["prey"], gbif_cache
        )
        predator_keys_specific, predator_terms_specific, predator_skipped_specific = split_specific_terms(
            spec["predators"], gbif_cache
        )
        prey_keys_broad, prey_terms_broad, prey_skipped_broad = split_broad_terms(
            broad["prey"], gbif_cache
        )
        predator_keys_broad, predator_terms_broad, predator_skipped_broad = split_broad_terms(
            broad["predators"], gbif_cache
        )

        prey_keys = prey_keys_specific | prey_keys_broad
        predator_keys = predator_keys_specific | predator_keys_broad

        prey_terms = prey_terms_specific + [f"[broad] {t}" for t in prey_terms_broad]
        predator_terms = predator_terms_specific + [f"[broad] {t}" for t in predator_terms_broad]
        prey_skipped = prey_skipped_specific + [f"[broad] {t}" for t in prey_skipped_broad]
        predator_skipped = predator_skipped_specific + [f"[broad] {t}" for t in predator_skipped_broad]

        rows.append(
            {
                "animal_name": animal_name,
                "extinction_year": extinction_year,
                "year_before": extinction_year - 1,
                "year_after": extinction_year + 1,
                "animal_species_key": animal_species_key,
                "animal_genus_key": animal_genus_key,
                "animal_name_key": animal_name_key,
                "occupied_region_keys": set(),
                "prey_keys": prey_keys,
                "predator_keys": predator_keys,
                "prey_terms": prey_terms,
                "predator_terms": predator_terms,
                "prey_terms_skipped": prey_skipped,
                "predator_terms_skipped": predator_skipped,
            }
        )

    if skipped_non_species:
        print(f"Skipped non-species/author-like names in occurrence pipeline: {skipped_non_species}")
    return rows


def clone_animals(animals: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cloned: List[Dict[str, Any]] = []
    for item in animals:
        copy_item = dict(item)
        copy_item["occupied_region_keys"] = set(item.get("occupied_region_keys", set()))
        copy_item["prey_keys"] = set(item.get("prey_keys", set()))
        copy_item["predator_keys"] = set(item.get("predator_keys", set()))
        copy_item["prey_terms"] = list(item.get("prey_terms", []))
        copy_item["predator_terms"] = list(item.get("predator_terms", []))
        copy_item["prey_terms_skipped"] = list(item.get("prey_terms_skipped", []))
        copy_item["predator_terms_skipped"] = list(item.get("predator_terms_skipped", []))
        cloned.append(copy_item)
    return cloned


def merge_occupied_regions(base_animals: List[Dict[str, Any]], by_source: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged = clone_animals(base_animals)
    for idx in range(len(merged)):
        region_union: Set[str] = set()
        for source_animals in by_source:
            if idx < len(source_animals):
                region_union.update(source_animals[idx].get("occupied_region_keys", set()))
        merged[idx]["occupied_region_keys"] = region_union
    return merged


def build_animal_identity_masks(
    animals: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int]]:
    species_key_to_mask: Dict[str, int] = defaultdict(int)
    genus_key_to_mask: Dict[str, int] = defaultdict(int)

    for idx, item in enumerate(animals):
        bit = 1 << idx
        species_key = normalize_key(item.get("animal_species_key", ""))
        genus_key = normalize_key(item.get("animal_genus_key", ""))
        name_key = normalize_key(item.get("animal_name_key", ""))

        if species_key:
            species_key_to_mask[species_key] |= bit
        elif name_key:
            # Use full normalized name as a strict fallback when species parsing fails.
            species_key_to_mask[name_key] |= bit
        if genus_key:
            # Species matching is attempted first; genus is a fallback when species does not hit.
            genus_key_to_mask[genus_key] |= bit

    return dict(species_key_to_mask), dict(genus_key_to_mask)


def add_region_for_mask(mask: int, region_key: str, out_regions: List[Set[str]]) -> None:
    m = int(mask)
    while m:
        low_bit = m & -m
        idx = low_bit.bit_length() - 1
        out_regions[idx].add(region_key)
        m ^= low_bit


def collect_occupied_regions(occurrence_zip: str, animals: List[Dict[str, Any]]) -> None:
    n = len(animals)
    occupied_regions: List[Set[str]] = [set() for _ in range(n)]
    species_key_to_mask, genus_key_to_mask = build_animal_identity_masks(animals)

    if not species_key_to_mask and not genus_key_to_mask:
        return

    use_cols = ["species", "genus", "scientificName"] + STRICT_REGION_COLUMNS
    source_name = os.path.basename(occurrence_zip)
    print(f"Scanning occurrence data [{source_name}] to collect strict occupied regions per extinct animal...")

    with zipfile.ZipFile(occurrence_zip, "r") as zip_ref:
        with zip_ref.open(OCCURRENCE_FILE) as f:
            reader = pd.read_csv(
                f,
                sep="\t",
                dtype=str,
                usecols=use_cols,
                low_memory=True,
                chunksize=CHUNK_SIZE,
            )

            for chunk_num, chunk in enumerate(reader, start=1):
                if chunk.empty:
                    if chunk_num % 20 == 0:
                        print(f"  [{source_name}] region chunks processed: {chunk_num}")
                    continue

                species_keys = normalize_taxon_series(chunk["species"])
                genus_keys = normalize_taxon_series(chunk["genus"])
                scientific_keys = normalize_taxon_series(chunk["scientificName"])
                species_binomial = extract_binomial_series(species_keys)
                scientific_binomial = extract_binomial_series(scientific_keys)
                genus_from_species = extract_genus_series(species_keys)
                genus_from_scientific = extract_genus_series(scientific_keys)

                species_masks = species_keys.map(species_key_to_mask).fillna(0).astype("object").to_numpy()
                species_bin_masks = species_binomial.map(species_key_to_mask).fillna(0).astype("object").to_numpy()
                scientific_masks = scientific_keys.map(species_key_to_mask).fillna(0).astype("object").to_numpy()
                scientific_bin_masks = (
                    scientific_binomial.map(species_key_to_mask).fillna(0).astype("object").to_numpy()
                )
                species_masks = np.bitwise_or(species_masks, species_bin_masks)
                species_masks = np.bitwise_or(species_masks, scientific_masks)
                species_masks = np.bitwise_or(species_masks, scientific_bin_masks)

                genus_masks = genus_keys.map(genus_key_to_mask).fillna(0).astype("object").to_numpy()
                genus_from_species_masks = (
                    genus_from_species.map(genus_key_to_mask).fillna(0).astype("object").to_numpy()
                )
                genus_from_scientific_masks = (
                    genus_from_scientific.map(genus_key_to_mask).fillna(0).astype("object").to_numpy()
                )
                genus_masks = np.bitwise_or(genus_masks, genus_from_species_masks)
                genus_masks = np.bitwise_or(genus_masks, genus_from_scientific_masks)

                animal_masks = species_masks.copy()
                zero_mask = animal_masks == 0
                animal_masks[zero_mask] = genus_masks[zero_mask]

                region_keys = build_region_key_series(chunk[STRICT_REGION_COLUMNS]).to_numpy()
                pair_df = pd.DataFrame({"region": region_keys, "mask": animal_masks})
                pair_df = pair_df[(pair_df["region"] != "") & (pair_df["mask"] != 0)]
                if pair_df.empty:
                    if chunk_num % 20 == 0:
                        print(f"  [{source_name}] region chunks processed: {chunk_num}")
                    continue

                unique_pairs = pair_df.drop_duplicates(["region", "mask"])
                for _, pair in unique_pairs.iterrows():
                    add_region_for_mask(int(pair["mask"]), str(pair["region"]), occupied_regions)

                if chunk_num % 20 == 0:
                    print(f"  [{source_name}] region chunks processed: {chunk_num}")

    for idx, item in enumerate(animals):
        item["occupied_region_keys"] = occupied_regions[idx]


def build_region_masks(animals: List[Dict[str, Any]]) -> Tuple[Dict[str, int], int]:
    region_key_to_mask: Dict[str, int] = defaultdict(int)
    fallback_all_regions_mask = 0
    for idx, item in enumerate(animals):
        bit = 1 << idx
        occupied = item.get("occupied_region_keys", set())
        if occupied:
            for region_key in occupied:
                region_key_to_mask[region_key] |= bit
        else:
            # Fallback requested by user: if no occupied regions were found,
            # treat the animal as matching all regions.
            fallback_all_regions_mask |= bit
    return dict(region_key_to_mask), int(fallback_all_regions_mask)


def build_key_masks(
    animals: List[Dict[str, Any]],
) -> Tuple[Dict[str, int], Dict[str, int], Dict[int, int], Dict[int, int]]:
    prey_key_to_mask: Dict[str, int] = defaultdict(int)
    predator_key_to_mask: Dict[str, int] = defaultdict(int)
    before_mask_by_year: Dict[int, int] = defaultdict(int)
    after_mask_by_year: Dict[int, int] = defaultdict(int)

    for idx, item in enumerate(animals):
        bit = 1 << idx
        before_mask_by_year[item["year_before"]] |= bit
        after_mask_by_year[item["year_after"]] |= bit

        for key in item["prey_keys"]:
            prey_key_to_mask[key] |= bit
        for key in item["predator_keys"]:
            predator_key_to_mask[key] |= bit

    return dict(prey_key_to_mask), dict(predator_key_to_mask), dict(before_mask_by_year), dict(after_mask_by_year)


def add_count_for_mask(mask: int, count: int, out_counts: List[int]) -> None:
    m = int(mask)
    while m:
        low_bit = m & -m
        idx = low_bit.bit_length() - 1
        out_counts[idx] += int(count)
        m ^= low_bit


def add_count_for_mask_with_region(
    mask: int, count: int, region_key: str, out_counts: List[int], out_region_counts: List[Dict[str, int]]
) -> None:
    m = int(mask)
    while m:
        low_bit = m & -m
        idx = low_bit.bit_length() - 1
        out_counts[idx] += int(count)
        out_region_counts[idx][region_key] = int(out_region_counts[idx].get(region_key, 0)) + int(count)
        m ^= low_bit


def update_side_counts(
    masks: np.ndarray,
    years_arr: np.ndarray,
    region_keys_arr: np.ndarray,
    before_mask_by_year: Dict[int, int],
    after_mask_by_year: Dict[int, int],
    before_counts: List[int],
    after_counts: List[int],
    before_region_counts: List[Dict[str, int]],
    after_region_counts: List[Dict[str, int]],
) -> None:
    if len(masks) == 0:
        return

    pair_df = pd.DataFrame({"year": years_arr, "region": region_keys_arr, "mask": masks})
    pair_df = pair_df[(pair_df["region"] != "") & (pair_df["mask"] != 0)]
    if pair_df.empty:
        return

    pair_counts = pair_df.value_counts(["year", "region", "mask"])
    for (year, region_key, mask), value in pair_counts.items():
        y = int(year)
        r = str(region_key)
        m = int(mask)

        before_mask = m & before_mask_by_year.get(y, 0)
        after_mask = m & after_mask_by_year.get(y, 0)

        if before_mask:
            add_count_for_mask_with_region(before_mask, int(value), r, before_counts, before_region_counts)
        if after_mask:
            add_count_for_mask_with_region(after_mask, int(value), r, after_counts, after_region_counts)


def aggregate_animal_counts(
    occurrence_zip: str,
    animals: List[Dict[str, Any]],
    years_of_interest: Set[int],
) -> Dict[str, List[Any]]:
    use_cols = ["year"] + TAXON_COLUMNS + STRICT_REGION_COLUMNS
    n = len(animals)
    prey_before_counts = [0] * n
    prey_after_counts = [0] * n
    predator_before_counts = [0] * n
    predator_after_counts = [0] * n
    prey_before_region_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(n)]
    prey_after_region_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(n)]
    predator_before_region_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(n)]
    predator_after_region_counts: List[Dict[str, int]] = [defaultdict(int) for _ in range(n)]

    (
        prey_key_to_mask,
        predator_key_to_mask,
        before_mask_by_year,
        after_mask_by_year,
    ) = build_key_masks(animals)
    region_key_to_mask, fallback_all_regions_mask = build_region_masks(animals)

    source_name = os.path.basename(occurrence_zip)
    print(
        f"Scanning occurrence data [{source_name}] and counting prey/predator matches "
        "(strict region + deduplicated per row)..."
    )
    with zipfile.ZipFile(occurrence_zip, "r") as zip_ref:
        with zip_ref.open(OCCURRENCE_FILE) as f:
            reader = pd.read_csv(
                f,
                sep="\t",
                dtype=str,
                usecols=use_cols,
                low_memory=True,
                chunksize=CHUNK_SIZE,
            )

            for chunk_num, chunk in enumerate(reader, start=1):
                year_numeric = pd.to_numeric(chunk["year"], errors="coerce")
                valid_year_mask = year_numeric.isin(years_of_interest)
                if not valid_year_mask.any():
                    if chunk_num % 20 == 0:
                        print(f"  [{source_name}] chunks processed: {chunk_num}")
                    continue

                years_arr = year_numeric[valid_year_mask].astype("int64").to_numpy()
                filtered = chunk.loc[valid_year_mask, TAXON_COLUMNS + STRICT_REGION_COLUMNS]
                row_count = len(filtered)
                if row_count == 0:
                    if chunk_num % 20 == 0:
                        print(f"  [{source_name}] chunks processed: {chunk_num}")
                    continue

                region_keys = build_region_key_series(filtered[STRICT_REGION_COLUMNS])
                region_masks = region_keys.map(region_key_to_mask).fillna(0).astype("object").to_numpy()
                if fallback_all_regions_mask:
                    region_masks = np.bitwise_or(region_masks, fallback_all_regions_mask)
                if not np.any(region_masks):
                    if chunk_num % 20 == 0:
                        print(f"  [{source_name}] chunks processed: {chunk_num}")
                    continue

                prey_masks = np.zeros(row_count, dtype=object)
                predator_masks = np.zeros(row_count, dtype=object)

                for col in TAXON_COLUMNS:
                    keys = normalize_taxon_series(filtered[col])
                    prey_col_masks = keys.map(prey_key_to_mask).fillna(0).astype("object").to_numpy()
                    pred_col_masks = keys.map(predator_key_to_mask).fillna(0).astype("object").to_numpy()
                    prey_masks = np.bitwise_or(prey_masks, prey_col_masks)
                    predator_masks = np.bitwise_or(predator_masks, pred_col_masks)

                prey_masks = np.bitwise_and(prey_masks, region_masks)
                predator_masks = np.bitwise_and(predator_masks, region_masks)

                update_side_counts(
                    prey_masks,
                    years_arr,
                    region_keys.to_numpy(),
                    before_mask_by_year,
                    after_mask_by_year,
                    prey_before_counts,
                    prey_after_counts,
                    prey_before_region_counts,
                    prey_after_region_counts,
                )
                update_side_counts(
                    predator_masks,
                    years_arr,
                    region_keys.to_numpy(),
                    before_mask_by_year,
                    after_mask_by_year,
                    predator_before_counts,
                    predator_after_counts,
                    predator_before_region_counts,
                    predator_after_region_counts,
                )

                if chunk_num % 20 == 0:
                    print(f"  [{source_name}] chunks processed: {chunk_num}")

    return {
        "prey_before": prey_before_counts,
        "prey_after": prey_after_counts,
        "predator_before": predator_before_counts,
        "predator_after": predator_after_counts,
        "prey_before_regions": [dict(d) for d in prey_before_region_counts],
        "prey_after_regions": [dict(d) for d in prey_after_region_counts],
        "predator_before_regions": [dict(d) for d in predator_before_region_counts],
        "predator_after_regions": [dict(d) for d in predator_after_region_counts],
    }


def average_counts_across_sources(source_counts: List[Dict[str, List[Any]]]) -> Dict[str, List[Any]]:
    if not source_counts:
        raise RuntimeError("No occurrence count sources available for averaging.")

    source_total = len(source_counts)
    vector_keys = ["prey_before", "prey_after", "predator_before", "predator_after"]
    region_keys = [
        "prey_before_regions",
        "prey_after_regions",
        "predator_before_regions",
        "predator_after_regions",
    ]
    animal_count = len(source_counts[0]["prey_before"])

    averaged: Dict[str, List[Any]] = {}

    for key in vector_keys:
        averaged_vals: List[float] = []
        for idx in range(animal_count):
            total_val = 0.0
            for counts in source_counts:
                total_val += float(counts[key][idx])
            averaged_vals.append(total_val / source_total)
        averaged[key] = averaged_vals

    for key in region_keys:
        averaged_maps: List[Dict[str, float]] = []
        for idx in range(animal_count):
            acc: Dict[str, float] = defaultdict(float)
            for counts in source_counts:
                region_map = counts[key][idx]
                for region_key, value in region_map.items():
                    acc[str(region_key)] += float(value)
            averaged_maps.append({k: (v / source_total) for k, v in acc.items()})
        averaged[key] = averaged_maps

    return averaged


def build_source_contributions(
    animals: List[Dict[str, Any]],
    source_counts: List[Dict[str, List[Any]]],
    occurrence_zips: List[str],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    source_total = len(source_counts)
    for idx, item in enumerate(animals):
        prey_totals = [
            float(counts["prey_before"][idx]) + float(counts["prey_after"][idx])
            for counts in source_counts
        ]
        pred_totals = [
            float(counts["predator_before"][idx]) + float(counts["predator_after"][idx])
            for counts in source_counts
        ]
        prey_sum = float(sum(prey_totals))
        pred_sum = float(sum(pred_totals))

        for source_idx, counts in enumerate(source_counts):
            prey_before = float(counts["prey_before"][idx])
            prey_after = float(counts["prey_after"][idx])
            pred_before = float(counts["predator_before"][idx])
            pred_after = float(counts["predator_after"][idx])
            prey_total = prey_before + prey_after
            pred_total = pred_before + pred_after
            contrib_amount = (prey_total + pred_total) / 2.0
            if contrib_amount <= 0:
                continue

            rows.append(
                {
                    "Animal Name": item["animal_name"],
                    "Occurrence Source File": os.path.basename(occurrence_zips[source_idx]),
                    "Raw Prey Occurrences (Before)": round(prey_before, 6),
                    "Raw Prey Occurrences (After)": round(prey_after, 6),
                    "Raw Predator Occurrences (Before)": round(pred_before, 6),
                    "Raw Predator Occurrences (After)": round(pred_after, 6),
                    "Raw Total Prey In Checked Regions": round(prey_total, 6),
                    "Raw Total Predator In Checked Regions": round(pred_total, 6),
                    "Source Contribution Amount": round(contrib_amount, 6),
                    "Prey Share In Animal (%)": round((100.0 * prey_total / prey_sum), 4)
                    if prey_sum > 0
                    else 0.0,
                    "Predator Share In Animal (%)": round((100.0 * pred_total / pred_sum), 4)
                    if pred_sum > 0
                    else 0.0,
                    "Averaging Weight": round(1.0 / source_total, 6) if source_total > 0 else 0.0,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            by=["Animal Name", "Source Contribution Amount"],
            ascending=[True, False],
        ).reset_index(drop=True)
    return out


def build_output(
    animals: List[Dict[str, Any]],
    counts: Dict[str, List[Any]],
    specific_path: str,
    occurrence_zips: List[str],
    year_multipliers: Dict[int, float],
) -> pd.DataFrame:
    output_rows: List[Dict[str, Any]] = []
    source_names = [os.path.basename(p) for p in occurrence_zips]
    source_label = "; ".join(source_names)
    source_count = len(source_names)

    for idx, item in enumerate(animals):
        year_before = item["year_before"]
        year_after = item["year_after"]
        before_multiplier = float(year_multipliers.get(year_before, 1.0))
        after_multiplier = float(year_multipliers.get(year_after, 1.0))

        prey_before_regions = dict(counts["prey_before_regions"][idx])
        prey_after_regions = dict(counts["prey_after_regions"][idx])
        predator_before_regions = dict(counts["predator_before_regions"][idx])
        predator_after_regions = dict(counts["predator_after_regions"][idx])

        prey_before_raw = float(counts["prey_before"][idx])
        prey_after_raw = float(counts["prey_after"][idx])
        predator_before_raw = float(counts["predator_before"][idx])
        predator_after_raw = float(counts["predator_after"][idx])

        prey_before = prey_before_raw * before_multiplier
        prey_after = prey_after_raw * after_multiplier
        predator_before = predator_before_raw * before_multiplier
        predator_after = predator_after_raw * after_multiplier
        prey_total_adjusted = prey_before + prey_after
        predator_total_adjusted = predator_before + predator_after
        occupied_regions = item.get("occupied_region_keys", set())
        region_mode = "strict" if occupied_regions else "fallback_all_regions"
        occupied_regions_text = (
            format_region_keys(occupied_regions)
            if occupied_regions
            else "ALL REGIONS (fallback: no occupied regions found)"
        )

        output_rows.append(
            {
                "Animal Name": item["animal_name"],
                "Extinction Year": item["extinction_year"],
                "Year Before": year_before,
                "Year After": year_after,
                "Region Matching Mode": region_mode,
                "Occupied Region Count": len(occupied_regions),
                "Occupied Regions": occupied_regions_text,
                "Prey Terms Used": "; ".join(item["prey_terms"]) if item["prey_terms"] else UNKNOWN,
                "Predator Terms Used": "; ".join(item["predator_terms"]) if item["predator_terms"] else UNKNOWN,
                "Prey Terms Skipped": "; ".join(item["prey_terms_skipped"]) if item["prey_terms_skipped"] else "",
                "Predator Terms Skipped": "; ".join(item["predator_terms_skipped"]) if item["predator_terms_skipped"] else "",
                "Prey Region Counts (Before Raw)": format_region_count_map(prey_before_regions),
                "Prey Region Counts (After Raw)": format_region_count_map(prey_after_regions),
                "Predator Region Counts (Before Raw)": format_region_count_map(predator_before_regions),
                "Predator Region Counts (After Raw)": format_region_count_map(predator_after_regions),
                "Year Multiplier (Before)": before_multiplier,
                "Year Multiplier (After)": after_multiplier,
                "Raw Prey Occurrences (Before)": round(prey_before_raw, 6),
                "Raw Prey Occurrences (After)": round(prey_after_raw, 6),
                "Raw Predator Occurrences (Before)": round(predator_before_raw, 6),
                "Raw Predator Occurrences (After)": round(predator_after_raw, 6),
                "Prey Occurrences (Before)": round(prey_before, 6),
                "Prey Occurrences (After)": round(prey_after, 6),
                "Predator Occurrences (Before)": round(predator_before, 6),
                "Predator Occurrences (After)": round(predator_after, 6),
                "Raw Total Prey In Checked Regions": round(prey_before_raw + prey_after_raw, 6),
                "Raw Total Predator In Checked Regions": round(predator_before_raw + predator_after_raw, 6),
                "Total Prey In Checked Regions (Adjusted)": round(prey_total_adjusted, 6),
                "Total Predator In Checked Regions (Adjusted)": round(predator_total_adjusted, 6),
                "Specific Source File": os.path.basename(specific_path),
                "Occurrence Source File": source_label,
                "Occurrence Source Count": source_count,
                "Occurrence Averaging Method": "mean",
            }
        )

    return pd.DataFrame(output_rows)


def main() -> None:
    if not os.path.exists(ANIMALS_PATH):
        raise FileNotFoundError(f"Missing file: {ANIMALS_PATH}")

    specific_path = find_specific_path()
    occurrence_zips = find_occurrence_zips()
    print(f"Using specific-name source: {specific_path}")
    print("Using occurrence sources:")
    for path in occurrence_zips:
        print(f"  - {path}")

    gbif_cache = load_gbif_cache()
    print(f"Loaded GBIF cache entries: {len(gbif_cache)}")
    year_multipliers = load_year_multipliers()
    print(f"Loaded year multipliers: {len(year_multipliers)}")

    animals = load_animals(specific_path, gbif_cache=gbif_cache)
    save_gbif_cache(gbif_cache)
    print(f"Loaded {len(animals)} extinct animals with extinction years.")

    years_of_interest: Set[int] = set()
    all_keys: Set[str] = set()
    for item in animals:
        years_of_interest.add(item["year_before"])
        years_of_interest.add(item["year_after"])
        all_keys.update(item["prey_keys"])
        all_keys.update(item["predator_keys"])

    all_keys.discard("")
    all_keys.discard("unknown")
    print(f"Tracking {len(years_of_interest)} years and {len(all_keys)} unique prey/predator keys.")

    animals_by_source: List[List[Dict[str, Any]]] = []
    counts_by_source: List[Dict[str, List[Any]]] = []

    for occurrence_zip in occurrence_zips:
        animals_for_source = clone_animals(animals)
        collect_occupied_regions(occurrence_zip, animals_for_source)
        animals_with_regions = sum(1 for a in animals_for_source if a.get("occupied_region_keys"))
        print(
            f"Animals with strict occupied regions [{os.path.basename(occurrence_zip)}]: "
            f"{animals_with_regions}/{len(animals_for_source)}"
        )
        source_counts = aggregate_animal_counts(occurrence_zip, animals_for_source, years_of_interest)
        animals_by_source.append(animals_for_source)
        counts_by_source.append(source_counts)

    merged_animals = merge_occupied_regions(animals, animals_by_source)
    averaged_counts = average_counts_across_sources(counts_by_source)
    source_contrib_df = build_source_contributions(merged_animals, counts_by_source, occurrence_zips)
    output_df = build_output(merged_animals, averaged_counts, specific_path, occurrence_zips, year_multipliers)
    output_df.to_csv(OUTPUT_PATH, index=False)
    os.makedirs(os.path.dirname(SOURCE_CONTRIB_OUTPUT_PATH), exist_ok=True)
    source_contrib_df.to_csv(SOURCE_CONTRIB_OUTPUT_PATH, index=False)

    print(f"\nCompleted. Wrote {len(output_df)} rows to: {OUTPUT_PATH}")
    print(f"Wrote source contribution rows: {len(source_contrib_df)} to: {SOURCE_CONTRIB_OUTPUT_PATH}")
    print(f"Occurrence source count averaged: {len(occurrence_zips)}")
    print("\nFirst 10 rows:")
    print(output_df.head(10))


if __name__ == "__main__":
    main()
