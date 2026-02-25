import os
from typing import Any, List, Tuple

import pandas as pd


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
INPUT_PATH = os.path.join(SCRIPT_DIR, "raw nums.csv")
YEAR_EQUIV_PATH = os.path.join(SCRIPT_DIR, "year_equivalences.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "whights")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "animal_final_impact_percentages.csv")
ALL_REGION_FALLBACK_PENALTY = float(os.getenv("ALL_REGION_FALLBACK_PENALTY", "0.75"))
MISSING_SIDE_PENALTY = float(os.getenv("MISSING_SIDE_PENALTY", "0.25"))
BROAD_TERM_PENALTY = float(os.getenv("BROAD_TERM_PENALTY", "0.85"))
BEFORE_FLOOR_QUANTILE = float(os.getenv("BEFORE_FLOOR_QUANTILE", "0.2"))
MIN_BEFORE_FLOOR = float(os.getenv("MIN_BEFORE_FLOOR", "1.0"))
USE_YEAR_MULTIPLIERS = os.getenv("USE_YEAR_MULTIPLIERS", "1").strip().lower() in {"1", "true", "yes", "y"}
BALANCE_BY_TERM_COUNT = os.getenv("BALANCE_BY_TERM_COUNT", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}
MIN_FINAL_IMPACT_PERCENTAGE = float(os.getenv("MIN_FINAL_IMPACT_PERCENTAGE", "0.0"))
DROP_DUPLICATE_IMPACT_ROWS = os.getenv("DROP_DUPLICATE_IMPACT_ROWS", "1").strip().lower() in {
    "1",
    "true",
    "yes",
    "y",
}

PREY_BEFORE_COL = "Prey Occurrences (Before)"
PREY_AFTER_COL = "Prey Occurrences (After)"
PRED_BEFORE_COL = "Predator Occurrences (Before)"
PRED_AFTER_COL = "Predator Occurrences (After)"
RAW_PREY_BEFORE_COL = "Raw Prey Occurrences (Before)"
RAW_PREY_AFTER_COL = "Raw Prey Occurrences (After)"
RAW_PRED_BEFORE_COL = "Raw Predator Occurrences (Before)"
RAW_PRED_AFTER_COL = "Raw Predator Occurrences (After)"
YEAR_MULT_BEFORE_COL = "Year Multiplier (Before)"
YEAR_MULT_AFTER_COL = "Year Multiplier (After)"
YEAR_BEFORE_COL = "Year Before"
YEAR_AFTER_COL = "Year After"
YEAR_EQ_FACTOR_COL = "1 Occurrence In From Equals In To"
YEAR_EQ_TO_MULT_COL = "To Year Multiplier"
PREY_TOTAL_CANDIDATES = [
    "Total Prey In Checked Regions (Adjusted)",
    "Raw Total Prey In Checked Regions",
]
PRED_TOTAL_CANDIDATES = [
    "Total Predator In Checked Regions (Adjusted)",
    "Raw Total Predator In Checked Regions",
]


def first_existing_column(columns: List[str], candidates: List[str]) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return ""


def count_semicolon_terms(value: Any) -> int:
    text = str(value or "").strip()
    if not text:
        return 1
    parts = [part.strip() for part in text.split(";")]
    cleaned = [
        part
        for part in parts
        if part and part.lower() not in {"unknown", "nan", "none", "null"}
    ]
    return max(1, len(cleaned))


def load_year_equivalence_map(path: str) -> dict[tuple[int, int], float]:
    if not os.path.exists(path):
        return {}

    eq_df = pd.read_csv(path)
    required = {"From Year", "To Year", YEAR_EQ_FACTOR_COL}
    if not required.issubset(set(eq_df.columns)):
        # Backward-compat fallback: use To Year Multiplier if constant-factor column is unavailable.
        required = {"From Year", "To Year", YEAR_EQ_TO_MULT_COL}
        if not required.issubset(set(eq_df.columns)):
            return {}
        value_col = YEAR_EQ_TO_MULT_COL
    else:
        value_col = YEAR_EQ_FACTOR_COL

    eq_df = eq_df[["From Year", "To Year", value_col]].copy()
    eq_df["From Year"] = pd.to_numeric(eq_df["From Year"], errors="coerce")
    eq_df["To Year"] = pd.to_numeric(eq_df["To Year"], errors="coerce")
    eq_df[value_col] = pd.to_numeric(eq_df[value_col], errors="coerce")
    eq_df = eq_df.dropna(subset=["From Year", "To Year", value_col])
    eq_df = eq_df[eq_df[value_col] > 0]

    eq_map: dict[tuple[int, int], float] = {}
    for _, row in eq_df.iterrows():
        key = (int(row["From Year"]), int(row["To Year"]))
        if key not in eq_map:
            eq_map[key] = float(row[value_col])

    return eq_map


def weighted_percent_change(
    before: float, after: float, total: float, scale: float, before_floor: float
) -> Tuple[float, float, float, float, float]:
    """
    Returns:
      weighted_abs_ratio: unbounded ratio form
      weighted_signed_ratio: unbounded ratio form
      raw_abs_ratio: unbounded ratio form
      raw_signed_ratio: unbounded ratio form
      weight: total/median_total (unbounded)
    """
    before = max(float(before), 0.0)
    after = max(float(after), 0.0)
    total = max(float(total), 0.0)

    before_floor = max(float(before_floor), MIN_BEFORE_FLOOR)
    denom = max(before, before_floor)
    if denom <= 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    raw_signed_ratio = (after - before) / denom
    raw_abs_ratio = abs(raw_signed_ratio)
    # No bounded limiter: use totals directly as an unbounded weight.
    scale = max(float(scale), 1.0)
    weight = total / scale
    weighted_abs_ratio = raw_abs_ratio * weight
    weighted_signed_ratio = raw_signed_ratio * weight

    return weighted_abs_ratio, weighted_signed_ratio, raw_abs_ratio, raw_signed_ratio, weight


def drop_exact_duplicate_animals(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Keep one row per exact Final Impact Percentage value and remove the rest.
    Tie-breaker is alphabetical Animal Name.
    """
    if df.empty or "Animal Name" not in df.columns or "Final Impact Percentage" not in df.columns:
        return df, []

    working = df.copy()
    working["__impact"] = pd.to_numeric(working["Final Impact Percentage"], errors="coerce")
    working = working.sort_values(
        by=["__impact", "Animal Name"],
        ascending=[False, True],
        kind="stable",
    ).reset_index(drop=True)

    removed_dup_impact = working.loc[
        working.duplicated(subset=["__impact"], keep="first"),
        "Animal Name",
    ].astype(str).tolist()

    filtered = (
        working.drop_duplicates(subset=["__impact"], keep="first")
        .drop(columns=["__impact"])
        .reset_index(drop=True)
    )
    return filtered, removed_dup_impact


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    required = [PREY_BEFORE_COL, PREY_AFTER_COL, PRED_BEFORE_COL, PRED_AFTER_COL, "Animal Name"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in input CSV: {missing}")

    prey_total_col = first_existing_column(list(df.columns), PREY_TOTAL_CANDIDATES)
    pred_total_col = first_existing_column(list(df.columns), PRED_TOTAL_CANDIDATES)

    for col in [PREY_BEFORE_COL, PREY_AFTER_COL, PRED_BEFORE_COL, PRED_AFTER_COL]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    has_raw_counts = all(c in df.columns for c in [RAW_PREY_BEFORE_COL, RAW_PREY_AFTER_COL, RAW_PRED_BEFORE_COL, RAW_PRED_AFTER_COL])
    has_year_multiplier_cols = all(c in df.columns for c in [YEAR_MULT_BEFORE_COL, YEAR_MULT_AFTER_COL])
    has_year_pair_cols = all(c in df.columns for c in [YEAR_BEFORE_COL, YEAR_AFTER_COL])

    year_adjustment_mode = "none"
    year_equiv_exact_count = 0
    year_equiv_fallback_multiplier_count = 0
    year_equiv_fallback_unity_count = 0

    if has_raw_counts:
        for col in [RAW_PREY_BEFORE_COL, RAW_PREY_AFTER_COL, RAW_PRED_BEFORE_COL, RAW_PRED_AFTER_COL]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if has_year_multiplier_cols:
        for col in [YEAR_MULT_BEFORE_COL, YEAR_MULT_AFTER_COL]:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    if has_year_pair_cols:
        for col in [YEAR_BEFORE_COL, YEAR_AFTER_COL]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    has_raw_and_year_context = has_raw_counts and (
        has_year_multiplier_cols or has_year_pair_cols
    )
    if has_raw_and_year_context:
        year_pair_factor = pd.Series(float("nan"), index=df.index, dtype="float64")
        exact_mask = pd.Series(False, index=df.index)
        multiplier_fallback_mask = pd.Series(False, index=df.index)
        unity_fallback_mask = pd.Series(False, index=df.index)

        year_equiv_map = load_year_equivalence_map(YEAR_EQUIV_PATH)

        if year_equiv_map and has_year_pair_cols:
            factor_values = []
            for _, row in df.iterrows():
                y_before = row.get(YEAR_BEFORE_COL)
                y_after = row.get(YEAR_AFTER_COL)
                if pd.isna(y_before) or pd.isna(y_after):
                    factor_values.append(float("nan"))
                    continue
                factor_values.append(year_equiv_map.get((int(y_before), int(y_after)), float("nan")))

            year_pair_factor = pd.Series(factor_values, index=df.index, dtype="float64")
            exact_mask = year_pair_factor.notna()

        if has_year_multiplier_cols:
            multiplier_factor = df[YEAR_MULT_AFTER_COL].replace(0, pd.NA)
            multiplier_fallback_mask = year_pair_factor.isna() & multiplier_factor.notna()
            year_pair_factor = year_pair_factor.fillna(multiplier_factor)

        unity_fallback_mask = year_pair_factor.isna()
        year_pair_factor = year_pair_factor.fillna(1.0).clip(lower=0)

        year_equiv_exact_count = int(exact_mask.sum())
        year_equiv_fallback_multiplier_count = int(multiplier_fallback_mask.sum())
        year_equiv_fallback_unity_count = int(unity_fallback_mask.sum())

        if USE_YEAR_MULTIPLIERS:
            # Convert "before" counts with year-equivalence factor (From->To constant factor).
            df[PREY_BEFORE_COL] = df[RAW_PREY_BEFORE_COL] * year_pair_factor
            df[PREY_AFTER_COL] = df[RAW_PREY_AFTER_COL]
            df[PRED_BEFORE_COL] = df[RAW_PRED_BEFORE_COL] * year_pair_factor
            df[PRED_AFTER_COL] = df[RAW_PRED_AFTER_COL]

            if year_equiv_exact_count > 0:
                year_adjustment_mode = "year_equivalence_constant_factor"
            elif year_equiv_fallback_multiplier_count > 0:
                year_adjustment_mode = "year_multiplier_after_fallback"
            else:
                year_adjustment_mode = "unity_fallback"
        else:
            # Requested behavior: do not apply year normalization.
            df[PREY_BEFORE_COL] = df[RAW_PREY_BEFORE_COL]
            df[PREY_AFTER_COL] = df[RAW_PREY_AFTER_COL]
            df[PRED_BEFORE_COL] = df[RAW_PRED_BEFORE_COL]
            df[PRED_AFTER_COL] = df[RAW_PRED_AFTER_COL]
            year_adjustment_mode = "disabled_raw_counts"

    has_raw_and_multiplier = all(
        c in df.columns
        for c in [
            RAW_PREY_BEFORE_COL,
            RAW_PREY_AFTER_COL,
            RAW_PRED_BEFORE_COL,
            RAW_PRED_AFTER_COL,
            YEAR_MULT_BEFORE_COL,
            YEAR_MULT_AFTER_COL,
        ]
    )
    if has_raw_and_multiplier and year_adjustment_mode == "none":
        # Backward-compatible path for older raw files that do not carry year-pair columns.
        if USE_YEAR_MULTIPLIERS:
            mult_before = df[YEAR_MULT_BEFORE_COL].clip(lower=0)
            mult_after = df[YEAR_MULT_AFTER_COL].clip(lower=0)
            df[PREY_BEFORE_COL] = df[RAW_PREY_BEFORE_COL] * mult_before
            df[PREY_AFTER_COL] = df[RAW_PREY_AFTER_COL] * mult_after
            df[PRED_BEFORE_COL] = df[RAW_PRED_BEFORE_COL] * mult_before
            df[PRED_AFTER_COL] = df[RAW_PRED_AFTER_COL] * mult_after
            year_adjustment_mode = "legacy_before_after_multipliers"
        else:
            df[PREY_BEFORE_COL] = df[RAW_PREY_BEFORE_COL]
            df[PREY_AFTER_COL] = df[RAW_PREY_AFTER_COL]
            df[PRED_BEFORE_COL] = df[RAW_PRED_BEFORE_COL]
            df[PRED_AFTER_COL] = df[RAW_PRED_AFTER_COL]
            year_adjustment_mode = "disabled_raw_counts"

    if prey_total_col:
        df[prey_total_col] = pd.to_numeric(df[prey_total_col], errors="coerce").fillna(0)
    else:
        prey_total_col = "__prey_total_fallback"
        df[prey_total_col] = df[PREY_BEFORE_COL] + df[PREY_AFTER_COL]

    if pred_total_col:
        df[pred_total_col] = pd.to_numeric(df[pred_total_col], errors="coerce").fillna(0)
    else:
        pred_total_col = "__pred_total_fallback"
        df[pred_total_col] = df[PRED_BEFORE_COL] + df[PRED_AFTER_COL]

    # Use recomputed totals from current (possibly damped) before/after values.
    prey_total_col = "__prey_total_effective"
    pred_total_col = "__pred_total_effective"
    df[prey_total_col] = df[PREY_BEFORE_COL] + df[PREY_AFTER_COL]
    df[pred_total_col] = df[PRED_BEFORE_COL] + df[PRED_AFTER_COL]

    # Keep all rows so the full extinct-animal set stays available.
    if df.empty:
        raise RuntimeError("Input CSV has no rows.")

    prey_before_positive = df.loc[df[PREY_BEFORE_COL] > 0, PREY_BEFORE_COL]
    pred_before_positive = df.loc[df[PRED_BEFORE_COL] > 0, PRED_BEFORE_COL]
    prey_before_floor = (
        float(prey_before_positive.quantile(BEFORE_FLOOR_QUANTILE)) if not prey_before_positive.empty else MIN_BEFORE_FLOOR
    )
    pred_before_floor = (
        float(pred_before_positive.quantile(BEFORE_FLOOR_QUANTILE)) if not pred_before_positive.empty else MIN_BEFORE_FLOOR
    )
    prey_before_floor = max(prey_before_floor, MIN_BEFORE_FLOOR)
    pred_before_floor = max(pred_before_floor, MIN_BEFORE_FLOOR)

    prey_terms_series = df.get("Prey Terms Used", pd.Series("", index=df.index))
    pred_terms_series = df.get("Predator Terms Used", pd.Series("", index=df.index))
    df["Prey Terms Count"] = prey_terms_series.map(count_semicolon_terms).astype(float)
    df["Predator Terms Count"] = pred_terms_series.map(count_semicolon_terms).astype(float)

    if BALANCE_BY_TERM_COUNT:
        # Average per prey/predator term so unequal list lengths do not dominate.
        df["Average Prey Before"] = (
            pd.to_numeric(df[PREY_BEFORE_COL], errors="coerce").fillna(0)
            / df["Prey Terms Count"].clip(lower=1.0)
        ).abs()
        df["Average Prey After"] = (
            pd.to_numeric(df[PREY_AFTER_COL], errors="coerce").fillna(0)
            / df["Prey Terms Count"].clip(lower=1.0)
        ).abs()
        df["Average Predator Before"] = (
            pd.to_numeric(df[PRED_BEFORE_COL], errors="coerce").fillna(0)
            / df["Predator Terms Count"].clip(lower=1.0)
        ).abs()
        df["Average Predator After"] = (
            pd.to_numeric(df[PRED_AFTER_COL], errors="coerce").fillna(0)
            / df["Predator Terms Count"].clip(lower=1.0)
        ).abs()

        prey_floor_series = (prey_before_floor / df["Prey Terms Count"].clip(lower=1.0)).clip(lower=MIN_BEFORE_FLOOR)
        pred_floor_series = (pred_before_floor / df["Predator Terms Count"].clip(lower=1.0)).clip(lower=MIN_BEFORE_FLOOR)
    else:
        df["Average Prey Before"] = pd.to_numeric(df[PREY_BEFORE_COL], errors="coerce").fillna(0).abs()
        df["Average Prey After"] = pd.to_numeric(df[PREY_AFTER_COL], errors="coerce").fillna(0).abs()
        df["Average Predator Before"] = pd.to_numeric(df[PRED_BEFORE_COL], errors="coerce").fillna(0).abs()
        df["Average Predator After"] = pd.to_numeric(df[PRED_AFTER_COL], errors="coerce").fillna(0).abs()

        prey_floor_series = pd.Series(prey_before_floor, index=df.index, dtype="float64")
        pred_floor_series = pd.Series(pred_before_floor, index=df.index, dtype="float64")

    # Stage differences as absolute values so no negative values are carried forward.
    df["Prey Average Difference"] = (df["Average Prey Before"] - df["Average Prey After"]).abs()
    df["Predator Average Difference"] = (df["Average Predator Before"] - df["Average Predator After"]).abs()

    # Build separate stage averages (before and after) across prey/predator, then diff them.
    df["Stage Before Average"] = (
        (df["Average Prey Before"] + df["Average Predator Before"]) / 2.0
    ).abs()
    df["Stage After Average"] = (
        (df["Average Prey After"] + df["Average Predator After"]) / 2.0
    ).abs()
    df["Stage Average Difference"] = (df["Stage Before Average"] - df["Stage After Average"]).abs()

    prey_denom = pd.concat([df["Average Prey Before"], prey_floor_series], axis=1).max(axis=1).clip(lower=MIN_BEFORE_FLOOR)
    pred_denom = pd.concat([df["Average Predator Before"], pred_floor_series], axis=1).max(axis=1).clip(lower=MIN_BEFORE_FLOOR)

    # Percent difference at each stage, then combined across prey/predator.
    df["Total Prey Percentage"] = (df["Prey Average Difference"].abs() / prey_denom) * 100.0
    df["Total Predator Percentage"] = (df["Predator Average Difference"].abs() / pred_denom) * 100.0
    df["Balanced Prey Percentage"] = df["Total Prey Percentage"]
    df["Balanced Predator Percentage"] = df["Total Predator Percentage"]

    stage_floor_series = ((prey_floor_series + pred_floor_series) / 2.0).clip(lower=MIN_BEFORE_FLOOR)
    stage_denom = pd.concat([df["Stage Before Average"], stage_floor_series], axis=1).max(axis=1).clip(lower=MIN_BEFORE_FLOOR)
    df["Base Final Impact Percentage"] = (df["Stage Average Difference"] / stage_denom) * 100.0

    if "Region Matching Mode" in df.columns:
        fallback_all_regions_mask = df["Region Matching Mode"].astype(str).str.lower().eq("fallback_all_regions")
    elif "Occupied Region Count" in df.columns:
        fallback_all_regions_mask = pd.to_numeric(df["Occupied Region Count"], errors="coerce").fillna(0).eq(0)
    else:
        fallback_all_regions_mask = pd.Series(False, index=df.index)

    missing_side_mask = (df["Total Prey Percentage"] <= 0) | (df["Total Predator Percentage"] <= 0)
    penalty_factor = pd.Series(1.0, index=df.index)
    penalty_factor[missing_side_mask] *= max(float(MISSING_SIDE_PENALTY), 0.0)
    penalty_factor[fallback_all_regions_mask] *= max(float(ALL_REGION_FALLBACK_PENALTY), 0.0)
    prey_terms = df.get("Prey Terms Used", pd.Series("", index=df.index)).astype(str).str.lower()
    predator_terms = df.get("Predator Terms Used", pd.Series("", index=df.index)).astype(str).str.lower()
    broad_terms_mask = prey_terms.str.contains(r"\[broad\]", regex=True) | predator_terms.str.contains(r"\[broad\]", regex=True)
    penalty_factor[broad_terms_mask] *= max(float(BROAD_TERM_PENALTY), 0.0)
    # Keep the true unscaled value (no rank scaling).
    df["Final Impact Percentage"] = df["Base Final Impact Percentage"] * penalty_factor
    df["Final Impact Percentage"] = pd.to_numeric(df["Final Impact Percentage"], errors="coerce").fillna(0)

    positive_mask = df["Final Impact Percentage"] > max(0.0, MIN_FINAL_IMPACT_PERCENTAGE)
    removed_non_positive_count = int((~positive_mask).sum())
    scoring_df = df.loc[positive_mask].copy()

    if DROP_DUPLICATE_IMPACT_ROWS:
        deduped_df, removed_dup_impact_names = drop_exact_duplicate_animals(scoring_df)
    else:
        deduped_df = scoring_df.copy()
        removed_dup_impact_names = []

    output_cols = ["Animal Name", "Final Impact Percentage"]

    result = deduped_df[output_cols].sort_values(
        "Final Impact Percentage", ascending=False
    ).reset_index(drop=True)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print(f"Prey total source: {prey_total_col}")
    print(f"Predator total source: {pred_total_col}")
    if USE_YEAR_MULTIPLIERS:
        print("Year normalization: enabled")
    else:
        print("Year normalization: disabled")
    print(f"Year adjustment mode: {year_adjustment_mode}")
    if year_equiv_exact_count or year_equiv_fallback_multiplier_count or year_equiv_fallback_unity_count:
        print(
            "Year pair factors - exact:"
            f" {year_equiv_exact_count}, multiplier fallback: {year_equiv_fallback_multiplier_count},"
            f" unity fallback: {year_equiv_fallback_unity_count}"
        )
    print(f"Before floors: prey={prey_before_floor:.6f}, predator={pred_before_floor:.6f}")
    if BALANCE_BY_TERM_COUNT:
        print("Term-count balancing: enabled (prey/pred totals averaged by term count)")
        print(
            "Prey terms count range: "
            f"{float(df['Prey Terms Count'].min()):.0f} .. {float(df['Prey Terms Count'].max()):.0f}"
        )
        print(
            "Predator terms count range: "
            f"{float(df['Predator Terms Count'].min()):.0f} .. {float(df['Predator Terms Count'].max()):.0f}"
        )
    else:
        print("Term-count balancing: disabled")
    print(f"Missing-side penalty rows: {int(missing_side_mask.sum())} (factor {MISSING_SIDE_PENALTY})")
    print(f"Fallback-all-regions penalty rows: {int(fallback_all_regions_mask.sum())} (factor {ALL_REGION_FALLBACK_PENALTY})")
    print(f"Broad-term penalty rows: {int(broad_terms_mask.sum())} (factor {BROAD_TERM_PENALTY})")
    print(
        "Removed non-positive impact rows: "
        f"{removed_non_positive_count} (threshold > {max(0.0, MIN_FINAL_IMPACT_PERCENTAGE)})"
    )
    print(f"Input rows used: {len(df)}")
    print(f"Rows after non-positive filter: {len(scoring_df)}")
    print(f"Rows after filters: {len(deduped_df)}")
    if DROP_DUPLICATE_IMPACT_ROWS:
        print(f"Animals removed for duplicate exact Final Impact Percentage (kept one each): {len(removed_dup_impact_names)}")
        if removed_dup_impact_names:
            preview_count = min(len(removed_dup_impact_names), 10)
            print(
                "Removed due to duplicate exact Final Impact Percentage: "
                + ", ".join(removed_dup_impact_names[:preview_count])
            )
            if len(removed_dup_impact_names) > preview_count:
                print(f"... plus {len(removed_dup_impact_names) - preview_count} more.")
    else:
        print("Duplicate exact Final Impact Percentage rows: kept")
    print(f"Saved: {OUTPUT_PATH}")
    print("\nTop 10 animals by Final Impact Percentage:")
    print(result.head(10))


if __name__ == "__main__":
    main()
