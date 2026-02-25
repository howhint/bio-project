import math
import os
import re
import zipfile
from typing import Dict, List, Tuple

import pandas as pd

try:
    from statsmodels.stats.multicomp import pairwise_tukeyhsd

    HAS_STATSMODELS = True
except Exception:
    pairwise_tukeyhsd = None
    HAS_STATSMODELS = False


SCRIPT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "cuda")
INPUT_PATH = os.path.join(SCRIPT_DIR, "whights", "animal_final_impact_percentages.csv")
EXTINCT_ZIP_PATH = os.path.join(SCRIPT_DIR, "data", "extinct animal data.zip")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "groups")

ORDER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "group_order.csv")
ORDER_ANOVA_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "group_order_anova.csv")
ORDER_LEVENE_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "group_order_levene.csv")
ORDER_TUKEY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "group_order_tukey_hsd.csv")

ORDER_MIN_GROUP_SIZE = int(os.getenv("ORDER_ANOVA_MIN_GROUP_SIZE", "4"))

OBSOLETE_OUTPUT_FILES = [
    "group_tertiary_level.csv",
    "group_diet_evaluation.csv",
    "group_family.csv",
    "group_class.csv",
    "group_genus.csv",
    "group_species_epithet.csv",
    "group_impact_summary.csv",
    "animal_trophic_levels.csv",
    "animal_top_contributors.csv",
    "group_binomial.csv",
    "group_trinomial.csv",
    "group_quaternary_level.csv",
    "group_impact_tier.csv",
    "group_prey_evidence_level.csv",
    "group_name_depth_level.csv",
    "group_taxonomy_resolution.csv",
]

NAME_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z.-]*")
UNKNOWN_TERMS = {"", "unknown", "none", "n/a", "na", "other", "others"}


def tokenize_name(name: str) -> List[str]:
    return NAME_TOKEN_RE.findall(str(name))


def canonical_binomial(name: str) -> str:
    tokens = tokenize_name(name)
    if len(tokens) >= 2:
        return f"{tokens[0]} {tokens[1]}"
    if len(tokens) == 1:
        return tokens[0]
    return ""


def choose_best_tax_row(group: pd.DataFrame) -> pd.Series:
    score = (
        group["Family"].astype(str).str.strip().ne("").astype(int)
        + group["Order"].astype(str).str.strip().ne("").astype(int)
        + group["Class"].astype(str).str.strip().ne("").astype(int)
    )
    idx = score.idxmax()
    return group.loc[idx]


def load_taxonomy_lookup(target_binomials: List[str]) -> pd.DataFrame:
    target_set = {b for b in target_binomials if b}
    if not target_set:
        return pd.DataFrame(columns=["Canonical Binomial", "Family", "Order", "Class"])
    if not os.path.exists(EXTINCT_ZIP_PATH):
        return pd.DataFrame(columns=["Canonical Binomial", "Family", "Order", "Class"])

    with zipfile.ZipFile(EXTINCT_ZIP_PATH, "r") as zf:
        if "occurrence.txt" not in zf.namelist():
            return pd.DataFrame(columns=["Canonical Binomial", "Family", "Order", "Class"])
        with zf.open("occurrence.txt", "r") as handle:
            occ = pd.read_csv(
                handle,
                sep="\t",
                dtype=str,
                low_memory=False,
                usecols=[
                    "species",
                    "scientificName",
                    "acceptedScientificName",
                    "family",
                    "order",
                    "class",
                ],
            )

    occ = occ.fillna("")
    occ["Canonical Binomial"] = occ["species"].map(canonical_binomial)
    missing = occ["Canonical Binomial"].eq("")
    occ.loc[missing, "Canonical Binomial"] = occ.loc[missing, "acceptedScientificName"].map(canonical_binomial)
    missing = occ["Canonical Binomial"].eq("")
    occ.loc[missing, "Canonical Binomial"] = occ.loc[missing, "scientificName"].map(canonical_binomial)

    occ = occ[occ["Canonical Binomial"].isin(target_set)].copy()
    if occ.empty:
        return pd.DataFrame(columns=["Canonical Binomial", "Family", "Order", "Class"])

    occ = occ.rename(columns={"family": "Family", "order": "Order", "class": "Class"})
    occ["Family"] = occ["Family"].astype(str).str.strip()
    occ["Order"] = occ["Order"].astype(str).str.strip()
    occ["Class"] = occ["Class"].astype(str).str.strip()

    best = occ.groupby("Canonical Binomial", as_index=False, sort=False).apply(choose_best_tax_row)
    if isinstance(best.index, pd.MultiIndex):
        best = best.reset_index(drop=True)
    return best[["Canonical Binomial", "Family", "Order", "Class"]].drop_duplicates("Canonical Binomial")


def is_unknown_category_value(value: str) -> bool:
    text = str(value).strip().lower()
    if text in UNKNOWN_TERMS:
        return True
    if text.startswith("unidentified"):
        return True
    return False


def aggregate_order_category(df: pd.DataFrame, min_group_size: int) -> pd.DataFrame:
    subset = df[df["Order"].astype(str).str.strip() != ""].copy()
    subset = subset[~subset["Order"].map(is_unknown_category_value)].copy()
    if subset.empty:
        return pd.DataFrame(
            columns=[
                "Category Type",
                "Category Value",
                "Animal Count",
                "Total Final Impact Percentage",
                "Average Final Impact Percentage",
                "Std Dev Final Impact Percentage",
                "Top Contributing Animal",
                "Top Animal Final Impact Percentage",
            ]
        )

    grouped = (
        subset.groupby("Order", dropna=False)
        .agg(
            **{
                "Animal Count": ("Animal Name", "count"),
                "Total Final Impact Percentage": ("Final Impact Percentage", "sum"),
                "Average Final Impact Percentage": ("Final Impact Percentage", "mean"),
                "Std Dev Final Impact Percentage": ("Final Impact Percentage", "std"),
            }
        )
        .reset_index()
        .rename(columns={"Order": "Category Value"})
    )
    grouped["Std Dev Final Impact Percentage"] = (
        pd.to_numeric(grouped["Std Dev Final Impact Percentage"], errors="coerce").fillna(0.0)
    )

    top_contrib = (
        subset.sort_values(
            by=["Order", "Final Impact Percentage", "Animal Name"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["Order"], keep="first")
        [["Order", "Animal Name", "Final Impact Percentage"]]
        .rename(
            columns={
                "Order": "Category Value",
                "Animal Name": "Top Contributing Animal",
                "Final Impact Percentage": "Top Animal Final Impact Percentage",
            }
        )
    )

    grouped = grouped.merge(top_contrib, on="Category Value", how="left")
    grouped.insert(0, "Category Type", "order")
    grouped = grouped[
        pd.to_numeric(grouped["Animal Count"], errors="coerce").fillna(0).ge(max(int(min_group_size), 1))
    ].copy()
    grouped = grouped.sort_values(
        by=["Total Final Impact Percentage", "Animal Count"],
        ascending=[False, False],
    ).reset_index(drop=True)
    return grouped


def _beta_continued_fraction(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-12
    fpmin = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - (qab * x / qap)
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d

    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = (m * (b - m) * x) / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c

        aa = -((a + m) * (qab + m) * x) / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break

    return h


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    ln_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    bt = math.exp(a * math.log(x) + b * math.log(1.0 - x) - ln_beta)
    threshold = (a + 1.0) / (a + b + 2.0)
    if x < threshold:
        return bt * _beta_continued_fraction(a, b, x) / a
    return 1.0 - (bt * _beta_continued_fraction(b, a, 1.0 - x) / b)


def f_distribution_survival_function(f_stat: float, df_num: int, df_den: int) -> float:
    if f_stat <= 0.0:
        return 1.0
    if df_num <= 0 or df_den <= 0:
        return float("nan")

    x = (df_num * f_stat) / ((df_num * f_stat) + df_den)
    cdf = regularized_incomplete_beta(df_num / 2.0, df_den / 2.0, x)
    return max(0.0, min(1.0, 1.0 - cdf))


def prepare_order_groups(
    df_with_impact: pd.DataFrame, min_group_size: int
) -> Tuple[pd.DataFrame, Dict[str, List[float]], Dict[str, float]]:
    valid = df_with_impact[df_with_impact["Order"].astype(str).str.strip() != ""].copy()
    valid = valid[~valid["Order"].map(is_unknown_category_value)].copy()
    valid["Final Impact Percentage"] = pd.to_numeric(valid["Final Impact Percentage"], errors="coerce")
    valid = valid.dropna(subset=["Final Impact Percentage"]).copy()

    all_groups = {
        str(order): group["Final Impact Percentage"].astype(float).tolist()
        for order, group in valid.groupby("Order", sort=True)
    }
    all_groups = {order: values for order, values in all_groups.items() if values}

    minimum = max(int(min_group_size), 1)
    included_groups = {order: values for order, values in all_groups.items() if len(values) >= minimum}
    included_orders = sorted(included_groups.keys())
    filtered = valid[valid["Order"].isin(included_orders)].copy()

    all_group_sizes = [len(values) for values in all_groups.values()]
    included_group_sizes = [len(values) for values in included_groups.values()]
    meta: Dict[str, float] = {
        "Minimum N Per Order": float(minimum),
        "Total Non-Unknown Orders": float(len(all_groups)),
        "Total Non-Unknown Animals": float(sum(all_group_sizes)),
        "Included Orders": float(len(included_groups)),
        "Included Animals": float(sum(included_group_sizes)),
        "Excluded Orders (< Min N)": float(len(all_groups) - len(included_groups)),
        "Excluded Animals (< Min N)": float(sum(all_group_sizes) - sum(included_group_sizes)),
        "Min Group Size": float(min(included_group_sizes)) if included_group_sizes else 0.0,
        "Max Group Size": float(max(included_group_sizes)) if included_group_sizes else 0.0,
        "Mean Group Size": (
            float(sum(included_group_sizes)) / float(len(included_group_sizes))
            if included_group_sizes
            else 0.0
        ),
    }
    return filtered, included_groups, meta


def run_order_anova(order_groups: Dict[str, List[float]], meta: Dict[str, float]) -> Dict[str, object]:
    group_sizes = [len(values) for values in order_groups.values()]
    k = len(order_groups)
    n_total = sum(group_sizes)

    summary: Dict[str, object] = {
        "Test": "One-way ANOVA",
        "Grouping Column": "Order",
        "Minimum N Per Order": int(meta["Minimum N Per Order"]),
        "Included Orders": int(meta["Included Orders"]),
        "Included Animals": int(meta["Included Animals"]),
        "Excluded Orders (< Min N)": int(meta["Excluded Orders (< Min N)"]),
        "Excluded Animals (< Min N)": int(meta["Excluded Animals (< Min N)"]),
        "Min Group Size": int(meta["Min Group Size"]),
        "Max Group Size": int(meta["Max Group Size"]),
        "Mean Group Size": float(meta["Mean Group Size"]),
        "DF Between": float("nan"),
        "DF Within": float("nan"),
        "F Statistic": float("nan"),
        "P Value": float("nan"),
        "Significant At 0.05": False,
        "Notes": "",
    }

    if k < 2:
        summary["Notes"] = (
            "Insufficient groups for ANOVA after exclusion rule: orders with n < "
            f"{int(meta['Minimum N Per Order'])} were excluded."
        )
        return summary
    if n_total <= k:
        summary["Notes"] = "Insufficient residual degrees of freedom for ANOVA."
        return summary

    grand_mean = sum(sum(values) for values in order_groups.values()) / float(n_total)
    ss_between = 0.0
    ss_within = 0.0

    for values in order_groups.values():
        group_n = len(values)
        group_mean = sum(values) / float(group_n)
        ss_between += group_n * ((group_mean - grand_mean) ** 2)
        ss_within += sum((v - group_mean) ** 2 for v in values)

    df_between = k - 1
    df_within = n_total - k
    ms_between = ss_between / float(df_between)
    ms_within = ss_within / float(df_within)

    if ms_within <= 0.0:
        f_stat = float("inf") if ms_between > 0.0 else 0.0
        p_value = 0.0 if math.isinf(f_stat) else 1.0
    else:
        f_stat = ms_between / ms_within
        p_value = f_distribution_survival_function(f_stat, df_between, df_within)

    summary["DF Between"] = float(df_between)
    summary["DF Within"] = float(df_within)
    summary["F Statistic"] = float(f_stat)
    summary["P Value"] = float(p_value)
    summary["Significant At 0.05"] = bool(p_value < 0.05) if not math.isnan(p_value) else False
    summary["Notes"] = (
        f"Orders represented by fewer than {int(meta['Minimum N Per Order'])} species were excluded from "
        "inferential analysis."
    )
    return summary


def run_levene_variant(order_groups: Dict[str, List[float]], center: str) -> Tuple[float, float, int, int]:
    groups = [values for values in order_groups.values() if values]
    k = len(groups)
    n_total = sum(len(g) for g in groups)
    if k < 2 or n_total <= k:
        return float("nan"), float("nan"), k - 1, n_total - k

    if center == "mean":
        centers = [sum(g) / float(len(g)) for g in groups]
    else:
        centers = []
        for g in groups:
            sorted_vals = sorted(g)
            m = len(sorted_vals)
            if m % 2 == 1:
                centers.append(sorted_vals[m // 2])
            else:
                centers.append((sorted_vals[m // 2 - 1] + sorted_vals[m // 2]) / 2.0)

    z_groups = [[abs(x - c) for x in g] for g, c in zip(groups, centers)]
    z_means = [sum(z) / float(len(z)) for z in z_groups]
    z_all = [v for z in z_groups for v in z]
    z_grand_mean = sum(z_all) / float(len(z_all))

    ss_between = sum(len(z) * ((mean_z - z_grand_mean) ** 2) for z, mean_z in zip(z_groups, z_means))
    ss_within = sum(sum((v - mean_z) ** 2 for v in z) for z, mean_z in zip(z_groups, z_means))

    df_num = k - 1
    df_den = n_total - k
    if ss_within <= 0.0:
        w_stat = float("inf") if ss_between > 0.0 else 0.0
        p_value = 0.0 if math.isinf(w_stat) else 1.0
        return w_stat, p_value, df_num, df_den

    w_stat = (df_den / float(df_num)) * (ss_between / ss_within)
    p_value = f_distribution_survival_function(w_stat, df_num, df_den)
    return float(w_stat), float(p_value), df_num, df_den


def run_order_levene(order_groups: Dict[str, List[float]], meta: Dict[str, float]) -> pd.DataFrame:
    rows = []
    variants = [
        ("Brown-Forsythe", "median"),
        ("Levene", "mean"),
    ]
    for test_name, center in variants:
        stat, p_value, df_num, df_den = run_levene_variant(order_groups, center=center)
        rows.append(
            {
                "Test": test_name,
                "Center": center,
                "Grouping Column": "Order",
                "Minimum N Per Order": int(meta["Minimum N Per Order"]),
                "Included Orders": int(meta["Included Orders"]),
                "Included Animals": int(meta["Included Animals"]),
                "Excluded Orders (< Min N)": int(meta["Excluded Orders (< Min N)"]),
                "Excluded Animals (< Min N)": int(meta["Excluded Animals (< Min N)"]),
                "DF Numerator": int(df_num) if not math.isnan(float(df_num)) else float("nan"),
                "DF Denominator": int(df_den) if not math.isnan(float(df_den)) else float("nan"),
                "W Statistic": float(stat),
                "P Value": float(p_value),
                "Equal Variances At 0.05": bool(p_value >= 0.05) if not math.isnan(p_value) else False,
                "Notes": (
                    f"Orders represented by fewer than {int(meta['Minimum N Per Order'])} species were excluded."
                ),
            }
        )

    return pd.DataFrame(rows)


def run_order_tukey(filtered_df: pd.DataFrame, meta: Dict[str, float]) -> pd.DataFrame:
    base_meta = {
        "Minimum N Per Order": int(meta["Minimum N Per Order"]),
        "Included Orders": int(meta["Included Orders"]),
        "Included Animals": int(meta["Included Animals"]),
        "Excluded Orders (< Min N)": int(meta["Excluded Orders (< Min N)"]),
        "Excluded Animals (< Min N)": int(meta["Excluded Animals (< Min N)"]),
    }

    if filtered_df["Order"].nunique() < 2:
        return pd.DataFrame(
            [
                {
                    **base_meta,
                    "Order 1": "",
                    "Order 2": "",
                    "Mean Difference (Order2 - Order1)": float("nan"),
                    "Adjusted P Value": float("nan"),
                    "95% CI Lower": float("nan"),
                    "95% CI Upper": float("nan"),
                    "Significant (alpha=0.05)": False,
                    "Status": "not_run",
                    "Notes": "Insufficient groups for Tukey HSD.",
                }
            ]
        )

    if not HAS_STATSMODELS or pairwise_tukeyhsd is None:
        return pd.DataFrame(
            [
                {
                    **base_meta,
                    "Order 1": "",
                    "Order 2": "",
                    "Mean Difference (Order2 - Order1)": float("nan"),
                    "Adjusted P Value": float("nan"),
                    "95% CI Lower": float("nan"),
                    "95% CI Upper": float("nan"),
                    "Significant (alpha=0.05)": False,
                    "Status": "not_run",
                    "Notes": "statsmodels is not installed.",
                }
            ]
        )

    tukey = pairwise_tukeyhsd(
        endog=filtered_df["Final Impact Percentage"].astype(float),
        groups=filtered_df["Order"].astype(str),
        alpha=0.05,
    )
    columns = tukey._results_table.data[0]
    rows = tukey._results_table.data[1:]
    out = pd.DataFrame(rows, columns=columns).rename(
        columns={
            "group1": "Order 1",
            "group2": "Order 2",
            "meandiff": "Mean Difference (Order2 - Order1)",
            "p-adj": "Adjusted P Value",
            "lower": "95% CI Lower",
            "upper": "95% CI Upper",
            "reject": "Significant (alpha=0.05)",
        }
    )
    out["Significant (alpha=0.05)"] = out["Significant (alpha=0.05)"].astype(str).str.lower().eq("true")
    out.insert(0, "Status", "ok")
    out.insert(1, "Notes", f"Orders with n < {int(meta['Minimum N Per Order'])} excluded before Tukey HSD.")
    out.insert(2, "Minimum N Per Order", int(meta["Minimum N Per Order"]))
    out.insert(3, "Included Orders", int(meta["Included Orders"]))
    out.insert(4, "Included Animals", int(meta["Included Animals"]))
    out.insert(5, "Excluded Orders (< Min N)", int(meta["Excluded Orders (< Min N)"]))
    out.insert(6, "Excluded Animals (< Min N)", int(meta["Excluded Animals (< Min N)"]))
    return out


def main() -> None:
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"Missing input file: {INPUT_PATH}")

    minimum_n = max(int(ORDER_MIN_GROUP_SIZE), 1)
    df = pd.read_csv(INPUT_PATH)
    required_cols = {"Animal Name", "Final Impact Percentage"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing required columns in input CSV: {missing}")

    df["Animal Name"] = df["Animal Name"].astype(str)
    df["Final Impact Percentage"] = pd.to_numeric(df["Final Impact Percentage"], errors="coerce").fillna(0.0)
    df = df[df["Final Impact Percentage"] > 0].copy()
    if df.empty:
        raise RuntimeError("No animals found with Final Impact Percentage > 0.")

    df["Canonical Binomial"] = df["Animal Name"].map(canonical_binomial)
    taxonomy_lookup = load_taxonomy_lookup(df["Canonical Binomial"].tolist())
    df = df.merge(taxonomy_lookup, on="Canonical Binomial", how="left")
    df["Order"] = df["Order"].fillna("").astype(str).str.strip()
    df.loc[df["Order"].eq(""), "Order"] = "Unknown"

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    order_grouped = aggregate_order_category(df, minimum_n)
    order_grouped.to_csv(ORDER_OUTPUT_PATH, index=False)

    filtered_df, included_groups, meta = prepare_order_groups(df, minimum_n)
    anova_summary = run_order_anova(included_groups, meta)
    pd.DataFrame([anova_summary]).to_csv(ORDER_ANOVA_OUTPUT_PATH, index=False)

    levene_df = run_order_levene(included_groups, meta)
    levene_df.to_csv(ORDER_LEVENE_OUTPUT_PATH, index=False)

    tukey_df = run_order_tukey(filtered_df, meta)
    tukey_df.to_csv(ORDER_TUKEY_OUTPUT_PATH, index=False)

    removed_files = []
    for filename in OBSOLETE_OUTPUT_FILES:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            os.remove(path)
            removed_files.append(filename)

    print(f"Input animals (Final Impact Percentage > 0): {len(df)}")
    print(f"Minimum n per order for inferential tests: {minimum_n}")
    print(f"Saved: {ORDER_OUTPUT_PATH}")
    print(f"Saved: {ORDER_LEVENE_OUTPUT_PATH}")
    print(f"Saved: {ORDER_ANOVA_OUTPUT_PATH}")
    print(f"Saved: {ORDER_TUKEY_OUTPUT_PATH}")
    if removed_files:
        print("Removed obsolete files:")
        for filename in removed_files:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
