import json
import os
import re
import time
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Set, Tuple

import pandas as pd


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
CSV_PATH = os.path.join(SCRIPT_DIR, "animal_names_parsed.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "animal_diet_info.csv")
REVIEW_PATH = os.path.join(SCRIPT_DIR, "animal_diet_info_needs_review.csv")
UNKNOWN_PATH = os.path.join(SCRIPT_DIR, "animal_diet_info_unknown.csv")
SPECIFIC_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "animal_diet_info_specific_names.csv")

MAX_ITEMS = int(os.getenv("MAX_DIET_ITEMS", "5"))
MAX_ROWS = int(os.getenv("MAX_ROWS", "0"))
REVIEW_THRESHOLD = int(os.getenv("REVIEW_CONFIDENCE_THRESHOLD", "75"))
GLOBI_LIMIT = int(os.getenv("GLOBI_LIMIT", "100"))
HTTP_TIMEOUT_SEC = int(os.getenv("HTTP_TIMEOUT_SEC", "25"))
REQUEST_PAUSE_SEC = float(os.getenv("REQUEST_PAUSE_SEC", "0.05"))

GBIF_MATCH_URL = "https://api.gbif.org/v1/species/match"
GLOBI_INTERACTION_URL = "https://api.globalbioticinteractions.org/interaction"

UNKNOWN = "Unknown"
DATA_SOURCE = "GBIF+GloBI (taxonomy aliases)"
RANK_FALLBACK_ORDER = ["species", "genus", "family", "order", "class", "phylum"]
RANK_PENALTY = {
    "species": 0,
    "genus": 10,
    "family": 20,
    "order": 30,
    "class": 40,
    "phylum": 50,
}
MAX_TAXON_ALIASES_PER_RANK = int(os.getenv("MAX_TAXON_ALIASES_PER_RANK", "6"))


_gbif_cache: Dict[str, Dict[str, Any]] = {}
_globi_cache: Dict[Tuple[str, str, int], List[Dict[str, str]]] = {}


def http_get_json(url: str) -> Dict[str, Any]:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "animal-diet-extractor/1.0",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as response:
        payload = response.read().decode("utf-8", errors="replace")
    return json.loads(payload)


def unique_take(items: List[str], limit: int = MAX_ITEMS) -> List[str]:
    out: List[str] = []
    seen = set()
    for item in items:
        cleaned = re.sub(r"\s+", " ", str(item or "")).strip(" ,;\t\n")
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
        if len(out) >= limit:
            break
    return out


def contains_unknown(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    tokens = [token.strip().lower() for token in value.replace(";", ",").split(",")]
    return "unknown" in tokens


def extract_binomial(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    text = re.sub(r"\([^)]*\)", "", text).strip()
    text = text.split(",")[0].strip()

    m = re.match(r"^([A-Z][A-Za-z-]+)\s+([a-z][A-Za-z-]+)(?:\s+([a-z][A-Za-z-]+))?", text)
    if not m:
        return text

    parts = [m.group(1), m.group(2)]
    third = m.group(3)
    if third:
        parts.append(third)
    return " ".join(parts)


def extract_genus(name: str) -> str:
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    parts = text.split()
    if len(parts) >= 2 and parts[0][:1].isalpha() and parts[0][0].isupper():
        return parts[0]
    return ""


def looks_like_species_name(name: str) -> bool:
    text = re.sub(r"\s+", " ", str(name or "")).strip()
    text = re.sub(r"\([^)]*\)", "", text).strip()
    text = text.split(",")[0].strip()
    return bool(re.match(r"^[A-Z][A-Za-z-]+\s+[a-z][A-Za-z-]+", text))


def name_candidates(animal_name: str) -> List[str]:
    raw = re.sub(r"\s+", " ", str(animal_name or "")).strip()
    first_chunk = raw.split(",")[0].strip() if raw else ""
    binomial = extract_binomial(raw)
    first_two = " ".join(first_chunk.split()[:2]).strip() if first_chunk else ""
    genus = extract_genus(binomial)

    candidates = [raw, first_chunk, binomial, first_two, genus]
    return unique_take(candidates, limit=10)


def gbif_match(name: str) -> Dict[str, Any]:
    cache_key = name.strip().lower()
    if cache_key in _gbif_cache:
        return _gbif_cache[cache_key]

    params = urllib.parse.urlencode({"name": name, "verbose": "true"})
    url = f"{GBIF_MATCH_URL}?{params}"

    try:
        data = http_get_json(url)
    except Exception as exc:
        data = {"matchType": "NONE", "error": str(exc), "query": name}

    _gbif_cache[cache_key] = data
    time.sleep(REQUEST_PAUSE_SEC)
    return data


def build_taxonomy(data: Dict[str, Any], fallback_species: str) -> Dict[str, str]:
    species = str(data.get("canonicalName") or data.get("scientificName") or fallback_species).strip()
    taxa = {
        "species": species,
        "genus": str(data.get("genus") or "").strip(),
        "family": str(data.get("family") or "").strip(),
        "order": str(data.get("order") or "").strip(),
        "class": str(data.get("class") or "").strip(),
        "phylum": str(data.get("phylum") or "").strip(),
    }
    if not taxa["genus"]:
        taxa["genus"] = extract_genus(taxa["species"])
    return taxa


def build_alias_map() -> Dict[str, List[str]]:
    return {rank: [] for rank in RANK_FALLBACK_ORDER}


def add_alias(alias_map: Dict[str, List[str]], rank: str, value: Any) -> None:
    if rank not in alias_map:
        return
    cleaned = re.sub(r"\s+", " ", str(value or "")).strip(" ,;\t\n")
    if not cleaned:
        return
    key = cleaned.lower()
    existing = {v.lower() for v in alias_map[rank]}
    if key not in existing:
        alias_map[rank].append(cleaned)


def add_aliases_from_taxa(alias_map: Dict[str, List[str]], taxa: Dict[str, str]) -> None:
    for rank in RANK_FALLBACK_ORDER:
        add_alias(alias_map, rank, taxa.get(rank, ""))


def add_aliases_from_gbif_match(alias_map: Dict[str, List[str]], data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        return
    for rank in RANK_FALLBACK_ORDER:
        add_alias(alias_map, rank, data.get(rank, ""))
    add_alias(alias_map, "species", data.get("canonicalName", ""))
    add_alias(alias_map, "species", data.get("scientificName", ""))
    add_alias(alias_map, "genus", extract_genus(str(data.get("canonicalName") or data.get("scientificName") or "")))

    for alt in data.get("alternatives", []) or []:
        if not isinstance(alt, dict):
            continue
        for rank in RANK_FALLBACK_ORDER:
            add_alias(alias_map, rank, alt.get(rank, ""))
        add_alias(alias_map, "species", alt.get("canonicalName", ""))
        add_alias(alias_map, "species", alt.get("scientificName", ""))
        add_alias(alias_map, "genus", extract_genus(str(alt.get("canonicalName") or alt.get("scientificName") or "")))


def resolve_taxon_name(animal_name: str) -> Tuple[Dict[str, str], Dict[str, List[str]], int, str]:
    best_data: Dict[str, Any] = {}
    best_name = ""
    best_conf = -1
    alias_map = build_alias_map()

    for candidate in name_candidates(animal_name):
        if not candidate:
            continue
        add_alias(alias_map, "species", candidate)
        add_alias(alias_map, "genus", extract_genus(candidate))

        data = gbif_match(candidate)
        add_aliases_from_gbif_match(alias_map, data)
        match_type = str(data.get("matchType", "NONE"))
        confidence = int(data.get("confidence") or 0)
        canonical = str(data.get("canonicalName") or data.get("scientificName") or "").strip()
        if match_type != "NONE" and canonical and confidence >= best_conf:
            best_data = data
            best_conf = confidence
            best_name = canonical

        if match_type in {"EXACT", "HIGHERRANK"} and canonical and confidence >= 90:
            taxa = build_taxonomy(data, fallback_species=canonical)
            add_aliases_from_taxa(alias_map, taxa)
            return taxa, alias_map, confidence, f"GBIF {match_type}"

    if best_name:
        taxa = build_taxonomy(best_data, fallback_species=best_name)
        add_aliases_from_taxa(alias_map, taxa)
        return taxa, alias_map, max(0, best_conf), "GBIF fallback"

    fallback = extract_binomial(animal_name)
    if fallback:
        taxa = build_taxonomy({}, fallback_species=fallback)
        add_aliases_from_taxa(alias_map, taxa)
        return taxa, alias_map, 0, "Name fallback"

    raw = str(animal_name).strip()
    taxa = build_taxonomy({}, fallback_species=raw)
    add_aliases_from_taxa(alias_map, taxa)
    return taxa, alias_map, 0, "Raw name"


def fetch_globi_rows(source_taxon: str = "", target_taxon: str = "", limit: int = GLOBI_LIMIT) -> List[Dict[str, str]]:
    key = (source_taxon.strip().lower(), target_taxon.strip().lower(), int(limit))
    if key in _globi_cache:
        return _globi_cache[key]

    params: Dict[str, str] = {
        "interactionType": "eats",
        "format": "json",
        "limit": str(limit),
    }
    if source_taxon:
        params["sourceTaxon"] = source_taxon
    if target_taxon:
        params["targetTaxon"] = target_taxon

    url = f"{GLOBI_INTERACTION_URL}?{urllib.parse.urlencode(params)}"

    rows: List[Dict[str, str]] = []
    try:
        payload = http_get_json(url)
        columns = payload.get("columns", [])
        data = payload.get("data", [])
        idx = {col: i for i, col in enumerate(columns)}

        for raw_row in data:
            row: Dict[str, str] = {}
            for col, i in idx.items():
                row[col] = str(raw_row[i]) if i < len(raw_row) and raw_row[i] is not None else ""
            rows.append(row)
    except Exception:
        rows = []

    _globi_cache[key] = rows
    time.sleep(REQUEST_PAUSE_SEC)
    return rows


def broad_category(taxon_path: str) -> str:
    p = (taxon_path or "").lower()

    if "aves" in p:
        return "birds"
    if "mammalia" in p:
        return "mammals"
    if "reptilia" in p:
        return "reptiles"
    if "amphibia" in p:
        return "amphibians"
    if any(k in p for k in ["actinopterygii", "chondrichthyes", "osteichthyes", "sarcopterygii", "myxini", "petromyzonti"]):
        return "fish"
    if "insecta" in p:
        return "insects"
    if "arachnida" in p:
        return "arachnids"
    if any(k in p for k in ["crustacea", "malacostraca", "branchiopoda", "maxillopoda"]):
        return "crustaceans"
    if "mollusca" in p:
        return "mollusks"
    if "annelida" in p:
        return "annelids"
    if "plantae" in p:
        return "plants"
    if "fungi" in p:
        return "fungi"
    if "animalia" in p:
        return "invertebrates"

    return ""


def summarize_interactions(
    taxa: Dict[str, str],
    alias_map: Dict[str, List[str]],
) -> Tuple[List[str], List[str], List[str], List[str], str, str, str]:
    prey_rows: List[Dict[str, str]] = []
    predator_rows: List[Dict[str, str]] = []
    prey_rank_used = ""
    predator_rank_used = ""
    used_queries: List[str] = []

    def row_key(row: Dict[str, str]) -> Tuple[str, str, str]:
        return (
            str(row.get("source_taxon_name", "")).strip().lower(),
            str(row.get("target_taxon_name", "")).strip().lower(),
            str(row.get("interaction_type", "")).strip().lower(),
        )

    def dedupe_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for row in rows:
            key = row_key(row)
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    for rank in RANK_FALLBACK_ORDER:
        rank_names = alias_map.get(rank, []) + [str(taxa.get(rank, "")).strip()]
        rank_names = unique_take(rank_names, limit=max(1, MAX_TAXON_ALIASES_PER_RANK))
        if not rank_names:
            continue

        for taxon_name in rank_names:
            if not prey_rows:
                candidate_prey = fetch_globi_rows(source_taxon=taxon_name)
                if candidate_prey:
                    prey_rows.extend(candidate_prey)
                    prey_rows = dedupe_rows(prey_rows)
                    prey_rank_used = rank if not prey_rank_used else prey_rank_used
                    used_queries.append(f"prey:{rank}={taxon_name}")

            if not predator_rows:
                candidate_predators = fetch_globi_rows(target_taxon=taxon_name)
                if candidate_predators:
                    predator_rows.extend(candidate_predators)
                    predator_rows = dedupe_rows(predator_rows)
                    predator_rank_used = rank if not predator_rank_used else predator_rank_used
                    used_queries.append(f"predators:{rank}={taxon_name}")

            if prey_rows and predator_rows:
                break

        if prey_rows and predator_rows:
            break

    prey_names_raw = [row.get("target_taxon_name", "") for row in prey_rows]
    predator_names_raw = [row.get("source_taxon_name", "") for row in predator_rows]

    prey_names = unique_take(prey_names_raw)
    predator_names = unique_take(predator_names_raw)

    prey_cats = unique_take([broad_category(row.get("target_taxon_path", "")) for row in prey_rows])
    predator_cats = unique_take([broad_category(row.get("source_taxon_path", "")) for row in predator_rows])

    prey_cats = [c for c in prey_cats if c]
    predator_cats = [c for c in predator_cats if c]

    if not prey_rows and not predator_rows:
        notes = "No GloBI eats-interaction records found from species through phylum."
    elif not prey_rows:
        notes = "Predator records found; no prey records found in GloBI."
    elif not predator_rows:
        notes = "Prey records found; no predator records found in GloBI."
    else:
        notes = "Predator and prey records found in GloBI."

    if used_queries:
        notes = f"{notes} Hits: {', '.join(used_queries)}."
    else:
        notes = f"{notes} Hits: none."

    return predator_cats, prey_cats, predator_names, prey_names, notes, predator_rank_used, prey_rank_used


def confidence_score(
    match_confidence: int,
    predator_names: List[str],
    prey_names: List[str],
    predator_rank_used: str,
    prey_rank_used: str,
) -> int:
    score = int(match_confidence * 0.6)
    if predator_names:
        score += 20
    if prey_names:
        score += 20
    score += min(20, (len(predator_names) + len(prey_names)) * 2)

    penalties = [
        RANK_PENALTY[rank]
        for rank in (predator_rank_used, prey_rank_used)
        if rank in RANK_PENALTY
    ]
    if penalties:
        score -= max(penalties)
    else:
        score -= 40

    return max(0, min(100, score))


def as_csv_value(items: List[str], sep: str = ", ") -> str:
    return sep.join(items) if items else UNKNOWN


def main() -> None:
    df = pd.read_csv(CSV_PATH)
    if MAX_ROWS > 0:
        df = df.head(MAX_ROWS).copy()
        print(f"MAX_ROWS enabled: processing first {len(df)} rows only.")

    if "Animal Name" not in df.columns:
        raise RuntimeError("Input CSV must include an 'Animal Name' column.")

    rows: List[Dict[str, Any]] = []
    skipped_non_species = 0
    total = len(df)
    print(f"Processing {total} animals using {DATA_SOURCE}...")

    for idx, row in df.iterrows():
        animal_name = str(row["Animal Name"]).strip()
        if not animal_name:
            continue
        if not looks_like_species_name(animal_name):
            skipped_non_species += 1
            continue

        if (idx + 1) % 10 == 0:
            print(f"Processing {idx + 1}/{total}: {animal_name}")

        try:
            taxa, alias_map, match_conf, match_note = resolve_taxon_name(animal_name)
            (
                predator_cats,
                prey_cats,
                predator_names,
                prey_names,
                interaction_note,
                predator_rank_used,
                prey_rank_used,
            ) = summarize_interactions(taxa, alias_map)

            confidence = confidence_score(
                match_conf,
                predator_names,
                prey_names,
                predator_rank_used=predator_rank_used,
                prey_rank_used=prey_rank_used,
            )
            predators = predator_cats if predator_cats else [UNKNOWN]
            prey = prey_cats if prey_cats else [UNKNOWN]
            specific_predators = predator_names if predator_names else [UNKNOWN]
            specific_prey = prey_names if prey_names else [UNKNOWN]

            needs_review = (
                UNKNOWN in predators
                or UNKNOWN in prey
                or predator_rank_used in {"family", "order", "class", "phylum"}
                or prey_rank_used in {"family", "order", "class", "phylum"}
                or confidence < REVIEW_THRESHOLD
            )

            matched_taxon = taxa.get("species", "").strip() or animal_name
            alias_total = sum(len(v) for v in alias_map.values())
            notes = (
                f"{match_note}; {interaction_note}; matched taxon: {matched_taxon}; "
                f"ranks used predators/prey: {predator_rank_used or 'none'}/{prey_rank_used or 'none'}; "
                f"taxonomy aliases used: {alias_total}"
            )

            rows.append(
                {
                    "Animal Name": animal_name,
                    "What Eats It (Predators)": as_csv_value(predators),
                    "What It Eats (Prey/Diet)": as_csv_value(prey),
                    "Specific Predators (Names)": as_csv_value(specific_predators, sep="; "),
                    "Specific Prey (Names)": as_csv_value(specific_prey, sep="; "),
                    "Confidence": confidence,
                    "Needs Review": bool(needs_review),
                    "Notes": notes,
                    "Model Used": DATA_SOURCE,
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "Animal Name": animal_name,
                    "What Eats It (Predators)": UNKNOWN,
                    "What It Eats (Prey/Diet)": UNKNOWN,
                    "Specific Predators (Names)": UNKNOWN,
                    "Specific Prey (Names)": UNKNOWN,
                    "Confidence": 0,
                    "Needs Review": True,
                    "Notes": f"Error: {exc}",
                    "Model Used": DATA_SOURCE,
                }
            )

    output_df = pd.DataFrame(rows)

    broad_columns = [
        "Animal Name",
        "What Eats It (Predators)",
        "What It Eats (Prey/Diet)",
        "Confidence",
        "Needs Review",
        "Notes",
        "Model Used",
    ]
    specific_columns = [
        "Animal Name",
        "Specific Predators (Names)",
        "Specific Prey (Names)",
        "Confidence",
        "Needs Review",
        "Notes",
        "Model Used",
    ]

    broad_df = output_df[broad_columns].copy()
    specific_df = output_df[specific_columns].copy()

    unknown_mask = broad_df["What Eats It (Predators)"].map(contains_unknown) | broad_df[
        "What It Eats (Prey/Diet)"
    ].map(contains_unknown)
    unknown_df = broad_df[unknown_mask].copy()
    known_df = broad_df[~unknown_mask].copy()

    known_df.to_csv(OUTPUT_PATH, index=False)
    unknown_df.to_csv(UNKNOWN_PATH, index=False)
    specific_df.to_csv(SPECIFIC_OUTPUT_PATH, index=False)

    review_df = known_df[known_df["Needs Review"] == True].copy()  # noqa: E712
    review_df.to_csv(REVIEW_PATH, index=False)

    print(f"\nCompleted! Saved {len(known_df)} known records to {OUTPUT_PATH}")
    print(f"Moved {len(unknown_df)} unknown records to {UNKNOWN_PATH}")
    print(f"Saved specific-name records to {SPECIFIC_OUTPUT_PATH}")
    print(f"Needs review (known only): {len(review_df)} records saved to {REVIEW_PATH}")
    print(f"Skipped non-species/author-like names: {skipped_non_species}")
    print("\nFirst 5 records:")
    print(known_df.head())


if __name__ == "__main__":
    main()
