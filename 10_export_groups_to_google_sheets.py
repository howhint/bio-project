import os
import re
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.parse import parse_qs, urlparse
# this uses my api key and my app im the only one who can use it, so its fine to have it here,
# but if you want to run this yourself  contact me at corwin.haday2@gmail.com or 
# the suport email that pops up when trying to run
import pandas as pd

try:
    import gspread
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials as UserCredentials
    from google.oauth2.service_account import Credentials as ServiceAccountCredentials
    from googleapiclient.discovery import build
    from google_auth_oauthlib.flow import InstalledAppFlow
except ImportError as exc:
    raise RuntimeError(
        "Missing Google dependencies. Install with:\n"
        "  python -m pip install gspread google-auth google-auth-oauthlib google-api-python-client"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "cuda"
GROUPS_DIR = SCRIPT_DIR / "groups"
RAW_NUMS_PATH = SCRIPT_DIR / "raw nums.csv"
DATA_DIR = SCRIPT_DIR / "data"
OCCURRENCE_CONTRIB_PATH = GROUPS_DIR / "occurrence_source_contributions.csv"
SOURCES_PDF_PATH = GROUPS_DIR / "sources_used.pdf"
SOURCES_TEXT_PATH = GROUPS_DIR / "sources_used.txt"
AUTH_MODE = os.getenv("GOOGLE_AUTH_MODE", "auto").strip().lower()
OAUTH_FLOW_MODE = os.getenv("GOOGLE_OAUTH_FLOW", "local_server").strip().lower()
SERVICE_ACCOUNT_FILE = os.getenv(
    "GOOGLE_SERVICE_ACCOUNT_FILE", str(SCRIPT_DIR / "service_account.json")
)
OAUTH_CLIENT_SECRET_FILE = os.getenv(
    "GOOGLE_OAUTH_CLIENT_SECRET_FILE", str(SCRIPT_DIR / "oauth_client_secret.json")
)
OAUTH_TOKEN_FILE = os.getenv(
    "GOOGLE_OAUTH_TOKEN_FILE", str(SCRIPT_DIR / "oauth_token.json")
)
SPREADSHEET_ID = os.getenv("GOOGLE_SPREADSHEET_ID", "").strip()
SPREADSHEET_TITLE = os.getenv("GOOGLE_SPREADSHEET_TITLE", "Extinct Animal Impact Analysis")
SHARE_EMAIL = os.getenv("GOOGLE_SHARE_EMAIL", "").strip()
CHART_SOURCE_TAB = "chart_source"
CHART_TAB = "order_impact_chart"
CATEGORY_CHARTS_TAB = "category_charts"
CHART_TOP_N = int(os.getenv("GOOGLE_CHART_TOP_N", "5"))
CHART_BOTTOM_N = int(os.getenv("GOOGLE_CHART_BOTTOM_N", "5"))
PRUNE_UNUSED_TABS = os.getenv("GOOGLE_PRUNE_UNUSED_TABS", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}
EXPORT_ORDER_ONLY = os.getenv("GOOGLE_EXPORT_ORDER_ONLY", "true").strip().lower() not in {
    "0",
    "false",
    "no",
}

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]
INVALID_TAB_CHARS = re.compile(r"[\[\]\*\?/\\:]")
WRITE_RETRY_ATTEMPTS = int(os.getenv("GOOGLE_WRITE_RETRY_ATTEMPTS", "5"))
WRITE_RETRY_DELAY_SEC = float(os.getenv("GOOGLE_WRITE_RETRY_DELAY_SEC", "65"))
UNKNOWN_TERMS = {"", "unknown", "none", "n/a", "na", "other", "others"}
DOI_RE = re.compile(r"https?://doi\.org/\S+", flags=re.IGNORECASE)
TAG_TITLE_RE = re.compile(r"<title>\s*(.*?)\s*</title>", flags=re.IGNORECASE | re.DOTALL)
TAG_PUBDATE_RE = re.compile(r"<pubDate>\s*(.*?)\s*</pubDate>", flags=re.IGNORECASE | re.DOTALL)
AVERAGE_COL = "Average Final Impact Percentage"
STDDEV_COL = "Std Dev Final Impact Percentage"
CATEGORY_VALUE_COL = "Category Value"
ANIMAL_COUNT_COL = "Animal Count"


def sanitize_tab_name(name: str, used: set[str]) -> str:
    clean = INVALID_TAB_CHARS.sub("_", name.strip())
    if not clean:
        clean = "Sheet"
    clean = clean[:100]
    candidate = clean
    idx = 2
    while candidate in used:
        suffix = f"_{idx}"
        candidate = f"{clean[:100 - len(suffix)]}{suffix}"
        idx += 1
    used.add(candidate)
    return candidate


def is_unknown_category_value(value: str) -> bool:
    text = str(value).strip().lower()
    if text in UNKNOWN_TERMS:
        return True
    if text.startswith("unidentified"):
        return True
    return False


def is_rate_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "429" in text
        or "rate limit" in text
        or "quota exceeded" in text
        or "writer requests per minute" in text
        or "writerequestsperminuteperuser" in text
    )


def run_with_write_retry(action_name: str, func):
    attempts = max(1, WRITE_RETRY_ATTEMPTS)
    for attempt in range(1, attempts + 1):
        try:
            return func()
        except Exception as exc:
            if not is_rate_limit_error(exc) or attempt >= attempts:
                raise
            print(
                f"{action_name} hit write quota (attempt {attempt}/{attempts}). "
                f"Sleeping {WRITE_RETRY_DELAY_SEC:.0f}s before retry..."
            )
            time.sleep(max(1.0, WRITE_RETRY_DELAY_SEC))
    raise RuntimeError(f"{action_name} failed after retries.")


def split_source_values(value: object) -> List[str]:
    parts: List[str] = []
    for piece in str(value).split(";"):
        clean = piece.strip()
        if clean and clean.lower() not in UNKNOWN_TERMS:
            parts.append(clean)
    return parts


def normalize_space(value: str) -> str:
    return " ".join(str(value).split())


def parse_xml_tag(xml_text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(xml_text)
    if not match:
        return ""
    return normalize_space(match.group(1))


def read_zip_text(zf: zipfile.ZipFile, member_name: str) -> str:
    if member_name not in zf.namelist():
        return ""
    with zf.open(member_name) as handle:
        return handle.read().decode("utf-8", "replace")


def parse_zip_external_sources(zip_path: Path) -> Dict[str, object]:
    out: Dict[str, object] = {
        "zip_name": zip_path.name,
        "title": "",
        "pub_date": "",
        "is_gbif_download": False,
        "citation_count": 0,
        "rights_dataset_count": 0,
        "dois": [],
        "citation_examples": [],
    }
    if not zip_path.exists():
        out["missing"] = True
        return out

    with zipfile.ZipFile(zip_path, "r") as zf:
        metadata_xml = read_zip_text(zf, "metadata.xml")
        citations_txt = read_zip_text(zf, "citations.txt")
        rights_txt = read_zip_text(zf, "rights.txt")

    title = parse_xml_tag(metadata_xml, TAG_TITLE_RE)
    pub_date = parse_xml_tag(metadata_xml, TAG_PUBDATE_RE)
    out["title"] = title
    out["pub_date"] = pub_date
    out["is_gbif_download"] = (
        "gbif occurrence download" in title.lower()
        or "gbif.org" in metadata_xml.lower()
        or "gbif.org" in citations_txt.lower()
    )

    citation_lines = [
        normalize_space(line)
        for line in citations_txt.replace("\r", "").split("\n")
        if normalize_space(line)
        and not line.lower().startswith("when using this dataset please use")
    ]
    out["citation_count"] = len(citation_lines)
    out["citation_examples"] = citation_lines[:5]

    doi_values: List[str] = []
    seen_dois: set[str] = set()
    for line in citation_lines:
        for doi in DOI_RE.findall(line):
            cleaned = doi.rstrip(".,;")
            key = cleaned.lower()
            if key in seen_dois:
                continue
            seen_dois.add(key)
            doi_values.append(cleaned)
    out["dois"] = doi_values

    rights_datasets = [
        normalize_space(line)
        for line in rights_txt.replace("\r", "").split("\n")
        if normalize_space(line).startswith("Dataset:")
    ]
    out["rights_dataset_count"] = len(rights_datasets)
    return out


def load_external_source_summaries() -> List[Dict[str, object]]:
    zip_names: set[str] = set()
    if RAW_NUMS_PATH.exists():
        raw_df = pd.read_csv(RAW_NUMS_PATH, usecols=["Occurrence Source File"])
        for value in raw_df["Occurrence Source File"].fillna(""):
            zip_names.update(split_source_values(value))

    if not zip_names:
        zip_names = {"animal data.zip", "extinct animal data.zip"}

    summaries: List[Dict[str, object]] = []
    for zip_name in sorted(zip_names):
        zip_path = DATA_DIR / zip_name
        if not zip_path.exists():
            zip_path = SCRIPT_DIR / zip_name
        summaries.append(parse_zip_external_sources(zip_path))
    return summaries


def load_diet_data_source_summary() -> Dict[str, object]:
    path = SCRIPT_DIR / "animal_diet_info_specific_names.csv"
    if not path.exists():
        return {"exists": False}

    df = pd.read_csv(path)
    out: Dict[str, object] = {"exists": True, "rows": int(len(df))}
    if "Model Used" in df.columns:
        values = sorted(
            {
                normalize_space(v)
                for v in df["Model Used"].fillna("").astype(str).tolist()
                if normalize_space(v)
            }
        )
        out["model_used"] = values
    else:
        out["model_used"] = []

    if "Notes" in df.columns:
        notes = df["Notes"].fillna("").astype(str)
        out["gbif_note_rows"] = int(notes.str.contains("GBIF", case=False, na=False).sum())
        out["globi_note_rows"] = int(notes.str.contains("GloBI", case=False, na=False).sum())
    return out


def escape_pdf_text(value: str) -> str:
    safe = value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return safe.encode("latin-1", "replace").decode("latin-1")


def wrap_text_line(value: str, max_chars: int = 98) -> List[str]:
    text = " ".join(str(value).strip().split())
    if not text:
        return [""]

    words = text.split(" ")
    wrapped: List[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            wrapped.append(current)
        current = word
    if current:
        wrapped.append(current)
    return wrapped


def build_simple_text_pdf_bytes(pages: List[List[str]]) -> bytes:
    if not pages:
        pages = [["No source details found."]]

    object_count = 3 + (len(pages) * 2)
    font_obj_id = object_count
    page_obj_start = 3
    content_obj_start = page_obj_start + len(pages)

    objects: Dict[int, bytes] = {}
    objects[1] = b"<< /Type /Catalog /Pages 2 0 R >>"

    kids = " ".join(f"{page_obj_start + idx} 0 R" for idx in range(len(pages)))
    objects[2] = f"<< /Type /Pages /Count {len(pages)} /Kids [{kids}] >>".encode("ascii")

    for idx, page_lines in enumerate(pages):
        page_obj_id = page_obj_start + idx
        content_obj_id = content_obj_start + idx
        objects[page_obj_id] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> "
            f"/Contents {content_obj_id} 0 R >>"
        ).encode("ascii")

        content_lines = [
            "BT",
            "/F1 11 Tf",
            "14 TL",
            "50 760 Td",
        ]
        for line_num, line in enumerate(page_lines):
            escaped = escape_pdf_text(line)
            if line_num == 0:
                content_lines.append(f"({escaped}) Tj")
            else:
                content_lines.append("T*")
                content_lines.append(f"({escaped}) Tj")
        content_lines.append("ET")
        stream_text = "\n".join(content_lines) + "\n"
        stream_bytes = stream_text.encode("latin-1", "replace")
        objects[content_obj_id] = (
            f"<< /Length {len(stream_bytes)} >>\nstream\n".encode("ascii")
            + stream_bytes
            + b"endstream"
        )

    objects[font_obj_id] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    pdf = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n"
    offsets = [0]
    for obj_id in range(1, object_count + 1):
        obj_bytes = objects[obj_id]
        offsets.append(len(pdf))
        pdf += f"{obj_id} 0 obj\n".encode("ascii")
        pdf += obj_bytes
        pdf += b"\nendobj\n"

    xref_start = len(pdf)
    pdf += f"xref\n0 {object_count + 1}\n".encode("ascii")
    pdf += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        pdf += f"{offset:010d} 00000 n \n".encode("ascii")
    pdf += f"trailer\n<< /Size {object_count + 1} /Root 1 0 R >>\n".encode("ascii")
    pdf += f"startxref\n{xref_start}\n%%EOF\n".encode("ascii")
    return pdf


def build_sources_report_lines() -> List[str]:
    lines: List[str] = []
    lines.append("Extinct Animal Impact Analysis - Sources Used")
    lines.append(
        "Generated UTC: "
        + datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    )
    lines.append("")
    lines.append("External data providers and datasets")
    source_summaries = load_external_source_summaries()
    if source_summaries:
        any_gbif = any(bool(item.get("is_gbif_download")) for item in source_summaries)
        if any_gbif:
            lines.append("- GBIF occurrence downloads (Darwin Core Archives)")
        for item in source_summaries:
            zip_name = str(item.get("zip_name", "(unknown zip)"))
            if item.get("missing"):
                lines.append(f"- {zip_name}: archive not found on disk")
                continue
            lines.append(f"- {zip_name}")
            title = str(item.get("title", "")).strip()
            if title:
                lines.append(f"  title: {title}")
            pub_date = str(item.get("pub_date", "")).strip()
            if pub_date:
                lines.append(f"  published: {pub_date}")
            lines.append(
                "  dataset citations: "
                + str(int(item.get("citation_count", 0)))
                + ", rights entries: "
                + str(int(item.get("rights_dataset_count", 0)))
            )
            dois = item.get("dois", [])
            if isinstance(dois, list) and dois:
                sample = ", ".join(str(v) for v in dois[:5])
                lines.append(f"  DOI sample: {sample}")

    diet_summary = load_diet_data_source_summary()
    if bool(diet_summary.get("exists")):
        lines.append("- GBIF species match API + GloBI interaction API (diet enrichment)")
        model_used = diet_summary.get("model_used", [])
        if isinstance(model_used, list) and model_used:
            lines.append("  model label(s): " + ", ".join(str(v) for v in model_used))
        lines.append("  rows using GBIF notes: " + str(int(diet_summary.get("gbif_note_rows", 0))))
        lines.append("  rows using GloBI notes: " + str(int(diet_summary.get("globi_note_rows", 0))))
    lines.append("")

    lines.append("Raw source trace fields in pipeline")
    lines.append("- raw nums.csv -> Specific Source File, Occurrence Source File")
    lines.append("- data/*.zip -> citations.txt, rights.txt, metadata.xml")
    lines.append("")

    lines.append("Internal pipeline files (for reproducibility)")
    lines.append(f"- {RAW_NUMS_PATH.name}")
    if (SCRIPT_DIR / "year_equivalences.csv").exists():
        lines.append("- year_equivalences.csv")
    if (SCRIPT_DIR / "animal_diet_info_specific_names.csv").exists():
        lines.append("- animal_diet_info_specific_names.csv")
    if (SCRIPT_DIR / "whights" / "animal_final_impact_percentages.csv").exists():
        lines.append("- whights/animal_final_impact_percentages.csv")
    lines.append("")

    lines.append("Occurrence source contribution summary")
    if OCCURRENCE_CONTRIB_PATH.exists():
        contrib_df = pd.read_csv(OCCURRENCE_CONTRIB_PATH)
        required_cols = {"Occurrence Source File", "Animal Name", "Source Contribution Amount"}
        if required_cols.issubset(contrib_df.columns):
            summary = (
                contrib_df.groupby("Occurrence Source File", dropna=False)
                .agg(
                    Animals=("Animal Name", "nunique"),
                    Rows=("Animal Name", "count"),
                    Total_Contribution=("Source Contribution Amount", "sum"),
                )
                .reset_index()
            )
            summary["Total_Contribution"] = pd.to_numeric(
                summary["Total_Contribution"], errors="coerce"
            ).fillna(0.0)
            summary = summary.sort_values(
                by=["Total_Contribution", "Animals"],
                ascending=[False, False],
            )
            for _, row in summary.iterrows():
                source = str(row["Occurrence Source File"]).strip() or "(blank)"
                animals = int(row["Animals"])
                rows = int(row["Rows"])
                total = float(row["Total_Contribution"])
                lines.append(
                    f"- {source}: animals={animals}, rows={rows}, total_contribution={total:.4f}"
                )
        else:
            lines.append("- occurrence_source_contributions.csv is missing expected columns")
    else:
        lines.append("- occurrence_source_contributions.csv not found")

    return lines


def write_sources_pdf(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = build_sources_report_lines()

    # Letter page with 11pt text and 14pt leading.
    max_lines_per_page = 48
    wrapped: List[str] = []
    for line in report_lines:
        wrapped.extend(wrap_text_line(line))

    pages: List[List[str]] = []
    for idx in range(0, len(wrapped), max_lines_per_page):
        pages.append(wrapped[idx : idx + max_lines_per_page])

    pdf_bytes = build_simple_text_pdf_bytes(pages)
    output_path.write_bytes(pdf_bytes)
    return output_path


def write_sources_text(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_lines = build_sources_report_lines()
    output_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return output_path


def max_average_in_table(df: pd.DataFrame) -> float:
    if AVERAGE_COL not in df.columns:
        return float("-inf")
    values = pd.to_numeric(df[AVERAGE_COL], errors="coerce")
    if not values.notna().any():
        return float("-inf")
    return float(values.max())


def sort_table_rows_by_average(df: pd.DataFrame) -> pd.DataFrame:
    if AVERAGE_COL not in df.columns:
        return df

    work = df.copy()
    work[AVERAGE_COL] = pd.to_numeric(work[AVERAGE_COL], errors="coerce")
    sort_cols = [AVERAGE_COL]
    ascending = [False]
    if ANIMAL_COUNT_COL in work.columns:
        work[ANIMAL_COUNT_COL] = pd.to_numeric(work[ANIMAL_COUNT_COL], errors="coerce")
        sort_cols.append(ANIMAL_COUNT_COL)
        ascending.append(False)
    if CATEGORY_VALUE_COL in work.columns:
        sort_cols.append(CATEGORY_VALUE_COL)
        ascending.append(True)
    return work.sort_values(by=sort_cols, ascending=ascending, na_position="last").reset_index(drop=True)


def load_csv_tables(groups_dir: Path) -> List[Tuple[str, pd.DataFrame]]:
    csv_paths = sorted(groups_dir.glob("*.csv"))
    if EXPORT_ORDER_ONLY:
        keep_names = {
            "group_order.csv",
            "group_order_anova.csv",
            "group_order_levene.csv",
            "group_order_tukey_hsd.csv",
        }
        csv_paths = [p for p in csv_paths if p.name in keep_names]
    tables: List[Tuple[str, pd.DataFrame]] = []
    used_names: set[str] = set()

    for path in csv_paths:
        df = sort_table_rows_by_average(pd.read_csv(path))
        tab = sanitize_tab_name(path.stem, used_names)
        tables.append((tab, df))
    tables.sort(key=lambda item: (-max_average_in_table(item[1]), item[0]))
    return tables


def to_grid_values(df: pd.DataFrame) -> List[List[str]]:
    safe_df = df.copy().where(pd.notna(df), None)
    rows: List[List[object]] = [safe_df.columns.tolist()]
    for raw_row in safe_df.values.tolist():
        row: List[object] = []
        for value in raw_row:
            if value is None:
                row.append("")
                continue
            if hasattr(value, "item"):
                value = value.item()
            row.append(value)
        rows.append(row)
    return rows


def write_table(spreadsheet: gspread.Spreadsheet, tab_name: str, df: pd.DataFrame) -> int:
    try:
        ws = spreadsheet.worksheet(tab_name)
    except gspread.WorksheetNotFound:
        rows = max(len(df) + 1, 10)
        cols = max(len(df.columns), 5)
        ws = run_with_write_retry(
            f"add_worksheet({tab_name})",
            lambda: spreadsheet.add_worksheet(title=tab_name, rows=rows, cols=cols),
        )

    grid = to_grid_values(df)
    rows = max(len(grid), 1)
    cols = max(len(grid[0]) if grid else 1, 1)
    run_with_write_retry(
        f"resize({tab_name})",
        lambda: ws.resize(rows=rows, cols=cols),
    )
    run_with_write_retry(
        f"update({tab_name})",
        lambda: ws.update(range_name="A1", values=grid, value_input_option="RAW"),
    )
    return ws.id


def col_index_to_a1(col_index_1_based: int) -> str:
    idx = int(col_index_1_based)
    if idx <= 0:
        raise ValueError("Column index must be >= 1")
    chars: List[str] = []
    while idx > 0:
        idx, rem = divmod(idx - 1, 26)
        chars.append(chr(ord("A") + rem))
    return "".join(reversed(chars))


def build_category_chart_window_df(df: pd.DataFrame) -> pd.DataFrame:
    cols = [CATEGORY_VALUE_COL]
    if AVERAGE_COL in df.columns:
        cols.append(AVERAGE_COL)
    if STDDEV_COL in df.columns:
        cols.append(STDDEV_COL)
    work = df[cols].copy()
    work = work[~work[CATEGORY_VALUE_COL].map(is_unknown_category_value)].copy()
    for num_col in [
        AVERAGE_COL,
        STDDEV_COL,
    ]:
        if num_col in work.columns:
            work[num_col] = pd.to_numeric(work[num_col], errors="coerce").fillna(0.0)
    if AVERAGE_COL in work.columns:
        work = work.sort_values(
            by=[AVERAGE_COL, CATEGORY_VALUE_COL],
            ascending=[False, True],
            na_position="last",
        ).reset_index(drop=True)

    limit = max(CHART_TOP_N, 0) + max(CHART_BOTTOM_N, 0)
    if limit <= 0 or len(work) <= limit:
        return work.reset_index(drop=True)

    top = work.head(max(CHART_TOP_N, 0))
    bottom = work.tail(max(CHART_BOTTOM_N, 0))
    return pd.concat([top, bottom], ignore_index=True)


def write_chart_window_block(
    spreadsheet: gspread.Spreadsheet,
    tab_name: str,
    source_col_count: int,
    chart_window_df: pd.DataFrame,
) -> Dict[str, int | None]:
    ws = spreadsheet.worksheet(tab_name)
    block_cols = list(chart_window_df.columns)
    block_rows = [block_cols] + chart_window_df.values.tolist()
    for row in block_rows:
        for idx, value in enumerate(row):
            if value is None:
                row[idx] = ""
            elif hasattr(value, "item"):
                row[idx] = value.item()

    start_col_1_based = int(source_col_count) + 2
    end_col_1_based = start_col_1_based + len(block_cols) - 1
    needed_rows = max(len(block_rows), ws.row_count)
    needed_cols = max(end_col_1_based, ws.col_count)

    run_with_write_retry(
        f"resize_for_chart_window({tab_name})",
        lambda: ws.resize(rows=needed_rows, cols=needed_cols),
    )

    start_col_a1 = col_index_to_a1(start_col_1_based)
    end_col_a1 = col_index_to_a1(end_col_1_based)
    range_name = f"{start_col_a1}1:{end_col_a1}{len(block_rows)}"
    run_with_write_retry(
        f"update_chart_window({tab_name})",
        lambda: ws.update(range_name=range_name, values=block_rows, value_input_option="RAW"),
    )

    average_col_idx_0_based: int | None = None
    if AVERAGE_COL in chart_window_df.columns:
        average_col_idx_0_based = start_col_1_based + chart_window_df.columns.get_loc(AVERAGE_COL) - 1

    stddev_col_idx_0_based: int | None = None
    if STDDEV_COL in chart_window_df.columns:
        stddev_col_idx_0_based = start_col_1_based + chart_window_df.columns.get_loc(STDDEV_COL) - 1

    return {
        "label_col_idx": start_col_1_based + chart_window_df.columns.get_loc(CATEGORY_VALUE_COL) - 1,
        "value_col_idx": average_col_idx_0_based,
        "average_col_idx": average_col_idx_0_based,
        "stddev_col_idx": stddev_col_idx_0_based,
        "data_rows": len(block_rows),
    }


def build_chart_source_df(groups_dir: Path) -> pd.DataFrame:
    order_path = groups_dir / "group_order.csv"
    if order_path.exists():
        df = pd.read_csv(order_path)
        required_order = {CATEGORY_VALUE_COL, AVERAGE_COL, STDDEV_COL}
        if required_order.issubset(df.columns):
            work = df.copy()
            work[CATEGORY_VALUE_COL] = work[CATEGORY_VALUE_COL].astype(str)
            work[AVERAGE_COL] = pd.to_numeric(work[AVERAGE_COL], errors="coerce").fillna(0.0)
            work[STDDEV_COL] = pd.to_numeric(work[STDDEV_COL], errors="coerce").fillna(0.0)
            work = work.sort_values(
                by=[AVERAGE_COL, CATEGORY_VALUE_COL],
                ascending=[False, True],
                na_position="last",
            ).reset_index(drop=True)
            return pd.DataFrame(
                {
                    "Label": "order - " + work[CATEGORY_VALUE_COL],
                    AVERAGE_COL: work[AVERAGE_COL],
                    STDDEV_COL: work[STDDEV_COL],
                }
            )

    return pd.DataFrame(columns=["Label", AVERAGE_COL, STDDEV_COL])


def get_sheet_id_map(service, spreadsheet_id: str) -> Dict[str, int]:
    meta = run_with_write_retry(
        "spreadsheets.get",
        lambda: service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute(),
    )
    mapping: Dict[str, int] = {}
    for s in meta.get("sheets", []):
        props = s.get("properties", {})
        mapping[props.get("title", "")] = int(props.get("sheetId"))
    return mapping


def recreate_chart_tab(service, spreadsheet_id: str, chart_tab_name: str) -> int:
    id_map = get_sheet_id_map(service, spreadsheet_id)
    requests = []
    if chart_tab_name in id_map:
        requests.append({"deleteSheet": {"sheetId": id_map[chart_tab_name]}})
    requests.append({"addSheet": {"properties": {"title": chart_tab_name}}})
    run_with_write_retry(
        f"recreate_chart_tab({chart_tab_name})",
        lambda: service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": requests}
        ).execute(),
    )
    return get_sheet_id_map(service, spreadsheet_id)[chart_tab_name]


def delete_tabs_not_in_keep_set(service, spreadsheet_id: str, keep_titles: set[str]) -> List[str]:
    id_map = get_sheet_id_map(service, spreadsheet_id)
    requests = []
    removed_titles: List[str] = []
    for title, sheet_id in id_map.items():
        if title not in keep_titles:
            requests.append({"deleteSheet": {"sheetId": sheet_id}})
            removed_titles.append(title)

    if requests:
        run_with_write_retry(
            "delete_tabs_not_in_keep_set",
            lambda: service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id, body={"requests": requests}
            ).execute(),
        )
    return removed_titles


def reorder_tabs_by_title(
    service,
    spreadsheet_id: str,
    ordered_titles: List[str],
) -> None:
    id_map = get_sheet_id_map(service, spreadsheet_id)
    requests: List[Dict[str, object]] = []
    seen: set[str] = set()
    idx = 0
    for title in ordered_titles:
        if title in seen:
            continue
        sheet_id = id_map.get(title)
        if sheet_id is None:
            continue
        requests.append(
            {
                "updateSheetProperties": {
                    "properties": {"sheetId": sheet_id, "index": idx},
                    "fields": "index",
                }
            }
        )
        seen.add(title)
        idx += 1

    if requests:
        run_with_write_retry(
            "reorder_tabs_by_title",
            lambda: service.spreadsheets().batchUpdate(
                spreadsheetId=spreadsheet_id, body={"requests": requests}
            ).execute(),
        )


def add_column_chart(
    service,
    spreadsheet_id: str,
    source_sheet_id: int,
    chart_sheet_id: int,
    data_rows: int,
    chart_title: str,
    x_axis_title: str,
    y_axis_title: str,
    label_col_idx: int = 0,
    value_col_idx: int | None = 1,
    average_col_idx: int | None = None,
    stddev_col_idx: int | None = None,
    anchor_row_idx: int = 0,
    anchor_col_idx: int = 0,
    width_pixels: int = 1200,
    height_pixels: int = 700,
) -> None:
    if data_rows <= 1:
        return

    end_row = data_rows
    safe_anchor_row_idx = min(max(anchor_row_idx, 0), max(end_row - 1, 0))
    safe_anchor_col_idx = max(anchor_col_idx, 0)
    bar_value_col_idx = value_col_idx if value_col_idx is not None else average_col_idx
    if bar_value_col_idx is None:
        return
    bar_series: Dict[str, object] = {
        "series": {
            "sourceRange": {
                "sources": [
                    {
                        "sheetId": source_sheet_id,
                        "startRowIndex": 0,
                        "endRowIndex": end_row,
                        "startColumnIndex": bar_value_col_idx,
                        "endColumnIndex": bar_value_col_idx + 1,
                    }
                ]
            }
        },
        "targetAxis": "LEFT_AXIS",
    }
    series: List[Dict[str, object]] = [bar_series]
    if stddev_col_idx is not None:
        series.append(
            {
                "series": {
                    "sourceRange": {
                        "sources": [
                            {
                                "sheetId": source_sheet_id,
                                "startRowIndex": 0,
                                "endRowIndex": end_row,
                                "startColumnIndex": stddev_col_idx,
                                "endColumnIndex": stddev_col_idx + 1,
                            }
                        ]
                    }
                },
                "targetAxis": "LEFT_AXIS",
            }
        )
    if average_col_idx is not None and average_col_idx != bar_value_col_idx:
        series.append(
            {
                "series": {
                    "sourceRange": {
                        "sources": [
                            {
                                "sheetId": source_sheet_id,
                                "startRowIndex": 0,
                                "endRowIndex": end_row,
                                "startColumnIndex": average_col_idx,
                                "endColumnIndex": average_col_idx + 1,
                            }
                        ]
                    }
                },
                "targetAxis": "LEFT_AXIS",
            }
        )

    request = {
        "addChart": {
            "chart": {
                "spec": {
                    "title": chart_title,
                    "basicChart": {
                        "chartType": "COLUMN",
                        "legendPosition": "RIGHT_LEGEND" if len(series) > 1 else "NO_LEGEND",
                        "headerCount": 1,
                        "axis": [
                            {"position": "BOTTOM_AXIS", "title": x_axis_title},
                            {"position": "LEFT_AXIS", "title": y_axis_title},
                        ],
                        "domains": [
                            {
                                "domain": {
                                    "sourceRange": {
                                        "sources": [
                                            {
                                                "sheetId": source_sheet_id,
                                                "startRowIndex": 0,
                                                "endRowIndex": end_row,
                                                "startColumnIndex": label_col_idx,
                                                "endColumnIndex": label_col_idx + 1,
                                            }
                                        ]
                                    }
                                }
                            }
                        ],
                        "series": series,
                    },
                },
                "position": {
                    "overlayPosition": {
                        "anchorCell": {
                            "sheetId": chart_sheet_id,
                            "rowIndex": safe_anchor_row_idx,
                            "columnIndex": safe_anchor_col_idx,
                        },
                        "offsetXPixels": 20,
                        "offsetYPixels": 20,
                        "widthPixels": width_pixels,
                        "heightPixels": height_pixels,
                    }
                },
            }
        }
    }

    run_with_write_retry(
        f"add_column_chart(sheet_id={chart_sheet_id})",
        lambda: service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id, body={"requests": [request]}
        ).execute(),
    )


def build_category_chart_specs(
    tables: List[Tuple[str, pd.DataFrame]],
) -> List[Dict[str, object]]:
    specs: List[Dict[str, object]] = []
    required_cols = {CATEGORY_VALUE_COL, AVERAGE_COL, STDDEV_COL}

    for tab_name, df in tables:
        if not tab_name.startswith("group_"):
            continue
        if df.empty:
            continue
        if not required_cols.issubset(df.columns):
            continue

        value_series = pd.to_numeric(df[AVERAGE_COL], errors="coerce")
        if not value_series.notna().any():
            continue

        if (
            "Category Type" in df.columns
            and not df["Category Type"].dropna().empty
            and df["Category Type"].dropna().astype(str).str.strip().nunique() == 1
        ):
            category_title = str(df["Category Type"].dropna().iloc[0]).strip() or tab_name
        else:
            category_title = tab_name.replace("group_", "").strip() or tab_name

        specs.append(
            {
                "source_tab_name": tab_name,
                "source_col_count": int(len(df.columns)),
                "chart_window_df": build_category_chart_window_df(df),
                "chart_title": f"{category_title} - Average / Std Dev Impact",
            }
        )

    return specs


def extract_auth_code(auth_response: str) -> str:
    value = auth_response.strip()
    if not value:
        raise ValueError("Missing OAuth authorization code.")

    if "://" in value:
        parsed = urlparse(value)
        code = parse_qs(parsed.query).get("code", [None])[0]
        if code:
            return code

    return value


def delete_existing_embedded_charts(
    service,
    spreadsheet_id: str,
    target_sheet_ids: set[int],
) -> int:
    meta = run_with_write_retry(
        "spreadsheets.get(charts)",
        lambda: service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute(),
    )
    requests: List[Dict[str, object]] = []
    for sheet in meta.get("sheets", []):
        props = sheet.get("properties", {})
        sheet_id = int(props.get("sheetId", -1))
        if sheet_id not in target_sheet_ids:
            continue
        for chart in sheet.get("charts", []) or []:
            chart_id = chart.get("chartId")
            if chart_id is None:
                continue
            requests.append({"deleteEmbeddedObject": {"objectId": int(chart_id)}})

    if not requests:
        return 0

    run_with_write_retry(
        "delete_existing_embedded_charts",
        lambda: service.spreadsheets().batchUpdate(
            spreadsheetId=spreadsheet_id,
            body={"requests": requests},
        ).execute(),
    )
    return len(requests)


def get_google_credentials():
    auth_mode = AUTH_MODE if AUTH_MODE in {"auto", "service_account", "oauth"} else "auto"

    if auth_mode in {"auto", "service_account"} and os.path.exists(SERVICE_ACCOUNT_FILE):
        creds = ServiceAccountCredentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
        return creds, "service_account"

    if auth_mode == "service_account":
        raise FileNotFoundError(
            f"Missing service account file: {SERVICE_ACCOUNT_FILE}\n"
            "Set GOOGLE_SERVICE_ACCOUNT_FILE to your JSON key path."
        )

    if not os.path.exists(OAUTH_CLIENT_SECRET_FILE):
        raise FileNotFoundError(
            f"Missing OAuth client secret file: {OAUTH_CLIENT_SECRET_FILE}\n"
            "Set GOOGLE_OAUTH_CLIENT_SECRET_FILE to your OAuth Desktop client JSON."
        )

    creds = None
    if os.path.exists(OAUTH_TOKEN_FILE):
        creds = UserCredentials.from_authorized_user_file(OAUTH_TOKEN_FILE, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as exc:
                print(f"Stored OAuth token refresh failed ({exc}). Re-running OAuth login flow.")
                creds = None
        else:
            pass
        if not creds or not creds.valid:
            flow = InstalledAppFlow.from_client_secrets_file(OAUTH_CLIENT_SECRET_FILE, SCOPES)
            if OAUTH_FLOW_MODE == "console":
                flow.redirect_uri = os.getenv("GOOGLE_OAUTH_REDIRECT_URI", "http://localhost").strip()
                auth_url, _ = flow.authorization_url(
                    access_type="offline",
                    include_granted_scopes="true",
                    prompt="consent",
                )
                print("\nOpen this URL in your browser and approve access:")
                print(auth_url)
                print(
                    "\nAfter approval, copy either:\n"
                    "  1) the raw authorization code, or\n"
                    "  2) the full redirected URL from your address bar."
                )
                auth_response = os.getenv("GOOGLE_OAUTH_AUTH_CODE", "").strip()
                if not auth_response:
                    auth_response = input(
                        "\nPaste the authorization code or redirected URL here: "
                    ).strip()
                auth_code = extract_auth_code(auth_response)
                flow.fetch_token(code=auth_code)
                creds = flow.credentials
            else:
                creds = flow.run_local_server(port=0)
        with open(OAUTH_TOKEN_FILE, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return creds, "oauth"


def main() -> None:
    if not GROUPS_DIR.exists():
        raise FileNotFoundError(f"Missing directory: {GROUPS_DIR}")
    sources_pdf_path = write_sources_pdf(SOURCES_PDF_PATH)
    sources_text_path = write_sources_text(SOURCES_TEXT_PATH)
    creds, auth_mode_used = get_google_credentials()
    gc = gspread.authorize(creds)
    service = build("sheets", "v4", credentials=creds)

    if SPREADSHEET_ID:
        spreadsheet = gc.open_by_key(SPREADSHEET_ID)
    else:
        spreadsheet = gc.create(SPREADSHEET_TITLE)

    if SHARE_EMAIL:
        spreadsheet.share(SHARE_EMAIL, perm_type="user", role="writer", notify=False)

    tables = load_csv_tables(GROUPS_DIR)
    sheet_ids: Dict[str, int] = {}
    for tab_name, df in tables:
        sheet_ids[tab_name] = write_table(spreadsheet, tab_name, df)

    chart_source_df = build_chart_source_df(GROUPS_DIR)
    chart_source_tab = sanitize_tab_name(CHART_SOURCE_TAB, set(sheet_ids.keys()))
    chart_source_id = write_table(spreadsheet, chart_source_tab, chart_source_df)
    category_chart_specs = build_category_chart_specs(tables)

    removed_tabs: List[str] = []
    if PRUNE_UNUSED_TABS:
        keep_titles = set(sheet_ids.keys())
        keep_titles.add(chart_source_tab)
        keep_titles.add(CHART_TAB)
        keep_titles.add(CATEGORY_CHARTS_TAB)
        removed_tabs = delete_tabs_not_in_keep_set(service, spreadsheet.id, keep_titles)

    chart_tab_id = recreate_chart_tab(service, spreadsheet.id, CHART_TAB)
    add_column_chart(
        service=service,
        spreadsheet_id=spreadsheet.id,
        source_sheet_id=chart_source_id,
        chart_sheet_id=chart_tab_id,
        data_rows=len(chart_source_df) + 1,
        chart_title="Top Group Average / Std Dev Impact",
        x_axis_title="Group",
        y_axis_title="Average Final Impact Percentage (Std Dev labels)",
        label_col_idx=0,
        value_col_idx=1,
        average_col_idx=None,
        stddev_col_idx=2,
    )

    category_charts_tab_id = recreate_chart_tab(service, spreadsheet.id, CATEGORY_CHARTS_TAB)
    source_chart_target_ids = {sheet_ids[str(spec["source_tab_name"])] for spec in category_chart_specs if str(spec["source_tab_name"]) in sheet_ids}
    removed_source_charts = delete_existing_embedded_charts(
        service=service,
        spreadsheet_id=spreadsheet.id,
        target_sheet_ids=source_chart_target_ids,
    )
    category_chart_count = 0
    for idx, spec in enumerate(category_chart_specs):
        source_tab_name = str(spec["source_tab_name"])
        source_sheet_id = sheet_ids.get(source_tab_name)
        if source_sheet_id is None:
            continue
        chart_window_meta = write_chart_window_block(
            spreadsheet=spreadsheet,
            tab_name=source_tab_name,
            source_col_count=int(spec["source_col_count"]),
            chart_window_df=spec["chart_window_df"],  # type: ignore[arg-type]
        )

        # 1) dedicated charts dashboard tab
        grid_row = idx // 2
        grid_col = idx % 2
        add_column_chart(
            service=service,
            spreadsheet_id=spreadsheet.id,
            source_sheet_id=source_sheet_id,
            chart_sheet_id=category_charts_tab_id,
            data_rows=int(chart_window_meta["data_rows"]),
            chart_title=str(spec["chart_title"]),
            x_axis_title="Category Value",
            y_axis_title="Average Final Impact Percentage (Std Dev labels)",
            label_col_idx=int(chart_window_meta["label_col_idx"]),
            value_col_idx=int(chart_window_meta["value_col_idx"]) if chart_window_meta["value_col_idx"] is not None else None,
            average_col_idx=None,
            stddev_col_idx=int(chart_window_meta["stddev_col_idx"]) if chart_window_meta["stddev_col_idx"] is not None else None,
            anchor_row_idx=grid_row * 22,
            anchor_col_idx=grid_col * 8,
            width_pixels=620,
            height_pixels=360,
        )
        # 2) chart on each source category tab (requested)
        add_column_chart(
            service=service,
            spreadsheet_id=spreadsheet.id,
            source_sheet_id=source_sheet_id,
            chart_sheet_id=source_sheet_id,
            data_rows=int(chart_window_meta["data_rows"]),
            chart_title=str(spec["chart_title"]),
            x_axis_title="Category Value",
            y_axis_title="Average Final Impact Percentage (Std Dev labels)",
            label_col_idx=int(chart_window_meta["label_col_idx"]),
            value_col_idx=int(chart_window_meta["value_col_idx"]) if chart_window_meta["value_col_idx"] is not None else None,
            average_col_idx=None,
            stddev_col_idx=int(chart_window_meta["stddev_col_idx"]) if chart_window_meta["stddev_col_idx"] is not None else None,
            anchor_row_idx=max(int(chart_window_meta["data_rows"]) + 1, 2),
            anchor_col_idx=0,
            width_pixels=900,
            height_pixels=420,
        )
        category_chart_count += 1

    ordered_titles = [tab_name for tab_name, _ in tables]
    ordered_titles.extend([chart_source_tab, CHART_TAB, CATEGORY_CHARTS_TAB])
    reorder_tabs_by_title(
        service=service,
        spreadsheet_id=spreadsheet.id,
        ordered_titles=ordered_titles,
    )

    print(f"Spreadsheet ID: {spreadsheet.id}")
    print(f"Spreadsheet URL: {spreadsheet.url}")
    print(f"Auth mode used: {auth_mode_used}")
    print(f"Uploaded tabs: {len(tables) + 1}")
    print(f"Chart source tab: {chart_source_tab}")
    print(f"Chart tab: {CHART_TAB}")
    print(f"Category charts tab: {CATEGORY_CHARTS_TAB}")
    print(f"Category charts created: {category_chart_count}")
    print(f"Chart cap for long categories: top {CHART_TOP_N} + bottom {CHART_BOTTOM_N}")
    print(f"Removed existing charts from category tabs: {removed_source_charts}")
    print(f"Sources PDF: {sources_pdf_path}")
    print(f"Sources Text: {sources_text_path}")
    if removed_tabs:
        print("Removed stale tabs: " + ", ".join(sorted(removed_tabs)))


if __name__ == "__main__":
    main()
