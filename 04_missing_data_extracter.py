import os
import re
from typing import Any, List

import pandas as pd
# getting animals with a lot of fallbacks or not in GBIF or GloBI to try to get another source of data on them.
# This is a first step to try to get more data on the animals with the least data,
# which are also the most likely to be endangered and in need of conservation efforts. The dataset is from

SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
INPUT_PATH = os.path.join(SCRIPT_DIR, "animal_diet_info_specific_names.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "animal_diet_info_missing_data.csv")

PREDATORS_COL = "Specific Predators (Names)"
PREY_COL = "Specific Prey (Names)"
REVIEW_COL = "Needs Review"
NAME_COL = "Animal Name"

MISSING_VALUES = {"", "nan", "none", "null", "na", "n/a"}
FALSE_VALUES = {"false", "0", "no", "n"}


def as_clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def has_unknown_or_missing(value: Any) -> bool:
    text = as_clean_text(value)
    if text.lower() in MISSING_VALUES:
        return True

    parts = re.split(r"[;,]", text)
    for part in parts:
        token = as_clean_text(part).lower()
        if token in MISSING_VALUES:
            return True
        if re.search(r"\bunknown\b", token):
            return True
    return False


def is_flagged_false(value: Any) -> bool:
    text = as_clean_text(value).lower()
    return text in FALSE_VALUES


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)
    required = [NAME_COL, PREDATORS_COL, PREY_COL]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in input CSV: {missing}")

    unknown_pred_mask = df[PREDATORS_COL].apply(has_unknown_or_missing)
    unknown_prey_mask = df[PREY_COL].apply(has_unknown_or_missing)

    if REVIEW_COL in df.columns:
        flagged_false_mask = df[REVIEW_COL].apply(is_flagged_false)
    else:
        flagged_false_mask = pd.Series([False] * len(df), index=df.index)

    selected_mask = unknown_pred_mask | unknown_prey_mask | flagged_false_mask
    out = df[selected_mask].copy()

    reasons: List[str] = []
    for idx in out.index:
        reason_parts: List[str] = []
        if bool(unknown_pred_mask.loc[idx]):
            reason_parts.append("Unknown or missing predators")
        if bool(unknown_prey_mask.loc[idx]):
            reason_parts.append("Unknown or missing prey")
        if bool(flagged_false_mask.loc[idx]):
            reason_parts.append("Needs Review is False")
        reasons.append("; ".join(reason_parts))

    out.insert(1, "Missing Data Reasons", reasons)
    out.insert(2, "Has Unknown/Missing Predators", unknown_pred_mask.loc[out.index].values)
    out.insert(3, "Has Unknown/Missing Prey", unknown_prey_mask.loc[out.index].values)
    out.insert(4, "Needs Review Flagged False", flagged_false_mask.loc[out.index].values)

    out = out.sort_values(by=[NAME_COL]).reset_index(drop=True)
    out.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Rows with unknown/missing predators: {int(unknown_pred_mask.sum())}")
    print(f"Rows with unknown/missing prey: {int(unknown_prey_mask.sum())}")
    print(f"Rows with '{REVIEW_COL}' flagged false: {int(flagged_false_mask.sum())}")
    print(f"Rows exported (combined): {len(out)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
