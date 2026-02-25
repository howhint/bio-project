import os
import pandas as pd


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
INPUT_PATH = os.path.join(SCRIPT_DIR, "raw nums.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "raw nums_no_zeros.csv")

COUNT_COLUMNS = [
    "Prey Occurrences (Before)",
    "Prey Occurrences (After)",
    "Predator Occurrences (Before)",
    "Predator Occurrences (After)",
]


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    df = pd.read_csv(INPUT_PATH)

    missing = [col for col in COUNT_COLUMNS if col not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in raw nums.csv: {missing}")

    for col in COUNT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    filtered = df[(df[COUNT_COLUMNS] > 0).all(axis=1)].copy()
    filtered.to_csv(OUTPUT_PATH, index=False)

    print(f"Input rows: {len(df)}")
    print(f"Rows with no zero counts: {len(filtered)}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
