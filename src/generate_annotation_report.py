#!/usr/bin/env python3
"""
Generate an annotation quality report from streamlit_app_output.xlsx.

For each sheet × language group, computes precision / recall / Jaccard
for three vectors (keywords, fields, index) in two settings:

  before  – original model output vs. gold standard
             (orig_kw_ids      vs. gold_kw_ids;
              orig_field_ids   vs. gold_field_ids;
              orig_index_terms vs. gold_index_terms)
  after   – original model output vs. annotator-approved set
             (orig_kw_ids      vs. kw_kept_ids;
              orig_field_ids   vs. field_kept_ids ∪ field_miss_agreed_ids;
              orig_index_terms vs. index_kept_terms ∪ index_miss_agreed_terms)

Language is joined from data/LUR_annotations.csv on ref_id ↔ Reference.

Outputs (in results/annotation_report/)
-------
  metrics_by_vector.csv  – one row per (sheet, language, vector)
  metrics_avg.csv        – averages across vectors per (sheet, language)
"""

import datetime
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, jaccard_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE    = Path("../data/streamlit_app_output.xlsx")
LUR_FILE      = Path("../data/LUR_annotations.csv")
KEYWORDS_FILE = Path("../data/Keywords.csv")
TOPICS_FILE   = Path("../data/Topics.csv")
OUTPUT_DIR    = Path("../results/annotation_report")
SHEETS = [
    "claude_opus4_6",
    "w_en_claude_opus4_6",
    "gemini_3_pro",
    "w_en_gemini_3_pro",
    "qwen_3_5",
    "w_en_qwen_3_5",
]


# ---------------------------------------------------------------------------
# Language lookup  (ref_id → language)
# ---------------------------------------------------------------------------

def build_language_map(lur_path: Path) -> dict[str, str]:
    """Return {Reference: Language} from the LUR annotations file."""
    lur = pd.read_csv(lur_path)
    # Drop rows without a Reference; keep first language when duplicates exist
    lur = lur.dropna(subset=["Reference"])
    lur["Reference"] = lur["Reference"].str.strip()
    return dict(zip(lur["Reference"], lur["Language"]))


# ---------------------------------------------------------------------------
# Set parsing
# ---------------------------------------------------------------------------

def parse_set(value) -> set | None:
    """Parse a comma-separated cell value into a set of lowercase strings.

    Returns None for datetime objects (Excel date-corruption artefact) —
    the caller treats the corresponding metric as NaN.
    Returns an empty set for NaN / empty cells.
    """
    if isinstance(value, (datetime.datetime, pd.Timestamp)):
        return None
    if value is None:
        return set()
    if isinstance(value, float):
        if np.isnan(value):
            return set()
        return {str(int(value))}  # single numeric ID stored as float
    if isinstance(value, int):
        return {str(value)}
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return set()
    return {item.strip().lower() for item in re.split(r",\s*", s) if item.strip()}


def compute_metrics_per_sample(pred: set | None, gold: set | None) -> tuple[float, float, float]:
    """Return (precision, recall, jaccard).

    Returns (NaN, NaN, NaN) when either set is None (corrupted data)
    or both are empty (undefined metrics).
    """
    if pred is None or gold is None:
        return np.nan, np.nan, np.nan
    if not pred and not gold:
        return np.nan, np.nan, np.nan
    # Create a unified vocabulary for the current sample
    vocab = list(pred | gold)
    # Build binary indicator arrays based on presence in the vocabulary
    y_true = [1 if x in gold else 0 for x in vocab]
    y_pred = [1 if x in pred else 0 for x in vocab]
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    j = jaccard_score(y_true, y_pred, zero_division=0)
    return p, r, j


# ---------------------------------------------------------------------------
# Per-sample metric computation
# ---------------------------------------------------------------------------

def compute_sample_metrics(row: pd.Series) -> dict:
    """Compute before/after P/R/J for keywords, fields, and index."""
    # --- keywords ---
    pred_kw = parse_set(row["orig_kw_ids"])
    gold_kw = parse_set(row["gold_kw_ids"])
    kept_kw = parse_set(row["kw_kept_ids"])
    kw_b = compute_metrics_per_sample(pred_kw, gold_kw)
    kw_a = compute_metrics_per_sample(pred_kw, kept_kw)

    # --- fields ---
    pred_fi = parse_set(row["orig_field_ids"])
    gold_fi = parse_set(row["gold_field_ids"])
    kept_fi = parse_set(row["field_kept_ids"])
    miss_fi = parse_set(row["field_miss_agreed_ids"])
    gold_fi_aft = (kept_fi | miss_fi) if (kept_fi is not None and miss_fi is not None) else None
    fi_b = compute_metrics_per_sample(pred_fi, gold_fi)
    fi_a = compute_metrics_per_sample(pred_fi, gold_fi_aft)

    # --- index ---
    pred_ix = parse_set(row["orig_index_terms"])
    gold_ix = parse_set(row["gold_index_terms"])
    kept_ix = parse_set(row["index_kept_terms"])
    miss_ix = parse_set(row["index_miss_agreed_terms"])
    gold_ix_aft = (kept_ix | miss_ix) if (kept_ix is not None and miss_ix is not None) else None
    ix_b = compute_metrics_per_sample(pred_ix, gold_ix)
    ix_a = compute_metrics_per_sample(pred_ix, gold_ix_aft)

    return {
        "kw_p_before": kw_b[0], "kw_r_before": kw_b[1], "kw_j_before": kw_b[2],
        "kw_p_after": kw_a[0], "kw_r_after": kw_a[1], "kw_j_after": kw_a[2],
        "fi_p_before": fi_b[0], "fi_r_before": fi_b[1], "fi_j_before": fi_b[2],
        "fi_p_after": fi_a[0], "fi_r_after": fi_a[1], "fi_j_after": fi_a[2],
        "ix_p_before": ix_b[0], "ix_r_before": ix_b[1], "ix_j_before": ix_b[2],
        "ix_p_after": ix_a[0], "ix_r_after": ix_a[1], "ix_j_after": ix_a[2],
    }


# ---------------------------------------------------------------------------
# Sheet processing
# ---------------------------------------------------------------------------

def load_and_deduplicate(sheet_name: str) -> tuple[pd.DataFrame, int]:
    """Load sheet, keep last annotation per name (latest date wins).
    Returns (deduplicated DataFrame, original row count).
    """
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    # n_total = len(df)
    df = df.drop_duplicates(subset="name", keep="last").reset_index(drop=True)
    return df


def process_sheet(sheet_name: str, lang_map: dict) -> pd.DataFrame:
    """Return per-sample DataFrame with computed metrics and language label."""
    df = load_and_deduplicate(sheet_name)

    names = sorted(df["name"].dropna().astype(str).tolist())
    # print(f"  [{sheet_name}] {len(names)} unique names:")
    # for n in names:
    #     print(f"    {n}")

    df["language"] = df["name"].map(lang_map).fillna("Unknown")

    records = []
    for _, row in df.iterrows():
        m = compute_sample_metrics(row)
        m["ref_id"] = row["name"]
        m["language"] = row["language"]
        records.append(m)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

VECTORS = [
    ("keywords", "kw"),
    ("fields", "fi"),
    ("index", "ix"),
]


def aggregate_group(grp: pd.DataFrame, sheet: str, language: str,
                    n_samples: int) -> list[dict]:
    """One summary row per vector for a (sheet, language) group."""
    rows = []
    for vec_name, prefix in VECTORS:
        row = {
            "sheet": sheet,
            "language": language,
            "n_samples": n_samples,
            "vector": vec_name,
        }
        for metric, letter in [("precision", "p"), ("recall", "r"), ("jaccard", "j")]:
            b = float(np.nanmean(grp[f"{prefix}_{letter}_before"]))
            a = float(np.nanmean(grp[f"{prefix}_{letter}_after"]))
            row[f"{metric}_before"] = round(b, 4)
            row[f"{metric}_after"] = round(a, 4)
            row[f"delta_{metric}"] = round(a - b, 4)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Model coverage analysis
# ---------------------------------------------------------------------------

MODEL_SHEETS = {
    "gemini": ["gemini_3_pro", "w_en_gemini_3_pro"],
    "qwen":   ["qwen_3_5", "w_en_qwen_3_5"],
    "claude": ["claude_opus4_6", "w_en_claude_opus4_6"],
}


def _safe_set(value) -> set:
    """parse_set wrapper that treats None (datetime corruption) as empty set."""
    result = parse_set(value)
    return result if result is not None else set()


def build_gold_annotations(lur_path: Path, excel_path: Path) -> pd.DataFrame:
    """Return a DataFrame [ref_id, text, new_gold_keywords, new_gold_fields, new_gold_index].

    new_gold_keywords  = kw_kept_ids  → resolved to keyword names
    new_gold_fields    = field_kept_ids ∪ field_miss_agreed_ids  → resolved to topic names
    new_gold_index     = index_kept_terms ∪ index_miss_agreed_terms  (already text)

    Values are unioned across all sheets (MODEL_SHEETS) per ref_id.
    Rows follow LUR order and are restricted to Analyzed == 'y' records
    that appear in at least one sheet.
    """
    kw_names = pd.read_csv(KEYWORDS_FILE).set_index("Id")["Keyword"].to_dict()
    fi_names = pd.read_csv(TOPICS_FILE).set_index("Id")["Topic"].to_dict()

    lur = pd.read_csv(lur_path)
    lur = lur[lur["Analyzed [y/n]"].str.strip().str.lower() == "y"].reset_index(drop=True)

    gold_kw: dict[str, set] = {}
    gold_fi: dict[str, set] = {}
    gold_ix: dict[str, set] = {}
    ref_text: dict[str, str] = {}

    all_sheets = [s for sheets in MODEL_SHEETS.values() for s in sheets]
    for sheet in all_sheets:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df["name"] = df["name"].str.strip()
        df = df.drop_duplicates(subset="name", keep="last")
        for _, row in df.iterrows():
            ref_id = row["name"]
            if pd.isna(ref_id):
                continue
            gold_kw[ref_id] = gold_kw.get(ref_id, set()) | _safe_set(row["kw_kept_ids"])
            gold_fi[ref_id] = gold_fi.get(ref_id, set()) | _safe_set(row["field_kept_ids"]) | _safe_set(row["field_miss_agreed_ids"])
            gold_ix[ref_id] = gold_ix.get(ref_id, set()) | _safe_set(row["index_kept_terms"]) | _safe_set(row["index_miss_agreed_terms"])
            if ref_id not in ref_text and pd.notna(row.get("text")):
                ref_text[ref_id] = str(row["text"])

    covered = set(gold_kw.keys())

    def resolve(ids: set, lookup: dict) -> str:
        names = sorted(lookup.get(int(i), i) for i in ids if str(i).strip())
        return ", ".join(names)

    records = []
    for _, row in lur.iterrows():
        ref_id = row.get("Reference")
        if pd.isna(ref_id) or ref_id not in covered:
            continue
        records.append({
            "ref_id":            ref_id,
            "text":              ref_text.get(ref_id, ""),
            "new_gold_keywords": resolve(gold_kw.get(ref_id, set()), kw_names),
            "new_gold_fields":   resolve(gold_fi.get(ref_id, set()), fi_names),
            "new_gold_index":    ", ".join(sorted(gold_ix.get(ref_id, set()))),
        })

    return pd.DataFrame(records)


def build_coverage_dataframe(lur_path: Path, excel_path: Path) -> pd.DataFrame:
    """Return a DataFrame [ref_id, gemini, qwen, claude] in LUR row order.

    Each boolean column is True when that model has an annotation for the record
    in any of its associated sheets.  w_en_claude_opus4_6 is excluded.
    """
    lur = pd.read_csv(lur_path)
    lur = lur[lur["Analyzed [y/n]"].str.strip().str.lower() == "y"]

    model_ref_ids: dict[str, set] = {}
    for model, sheets in MODEL_SHEETS.items():
        covered: set = set()
        for sheet in sheets:
            df = pd.read_excel(excel_path, sheet_name=sheet)
            covered.update(df["name"].dropna().unique())
        model_ref_ids[model] = covered

    records = []
    for _, row in lur.iterrows():
        ref_id = row["Reference"] if pd.notna(row.get("Reference")) else None
        record: dict = {"ref_id": ref_id}
        for model, covered in model_ref_ids.items():
            record[model] = (ref_id in covered) if ref_id is not None else False
        records.append(record)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Model comparison by vector
# ---------------------------------------------------------------------------

def build_model_comparison(vector_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank all sheets by jaccard_after within each vector (language == 'All').

    Returns:
        ranking_df  – all sheets ranked per vector (by jaccard_after desc)
        best_df     – one row per vector: the top-ranked sheet and its metrics
    """
    all_rows = vector_df[vector_df["language"] == "All"].copy()

    metric_cols = [
        "precision_before", "recall_before", "jaccard_before",
        "precision_after",  "recall_after",  "jaccard_after",
        "delta_precision",  "delta_recall",   "delta_jaccard",
    ]

    ranking_rows = []
    for vec in ["keywords", "fields", "index"]:
        grp = all_rows[all_rows["vector"] == vec].copy()
        grp = grp.sort_values("jaccard_after", ascending=False).reset_index(drop=True)
        grp["rank"] = grp.index + 1
        ranking_rows.append(grp[["rank", "sheet", "n_samples", "vector"] + metric_cols])

    ranking_df = pd.concat(ranking_rows, ignore_index=True)

    best_df = ranking_df[ranking_df["rank"] == 1][
        ["vector", "sheet", "n_samples"] + metric_cols
    ].reset_index(drop=True)

    return ranking_df, best_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_FILE.exists():
        sys.exit(f"Input file not found: {INPUT_FILE}")
    if not LUR_FILE.exists():
        sys.exit(f"LUR annotations file not found: {LUR_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lang_map = build_language_map(LUR_FILE)

    all_vector_rows: list[dict] = []

    print("Processing sheets...")
    for sheet in SHEETS:
        per_sample = process_sheet(sheet, lang_map)
        n_dedup = len(per_sample)

        # All records combined (no language split)
        all_vector_rows.extend(aggregate_group(per_sample, sheet, "All", n_dedup))

        # Then grouped by language
        for lang, grp in per_sample.groupby("language", sort=True):
            all_vector_rows.extend(aggregate_group(grp, sheet, lang, len(grp)))

        langs = per_sample["language"].value_counts().to_dict()
        print(f"  {sheet}: {n_dedup} samples | {langs}")

    # ---- per-vector table ----
    vector_df = pd.DataFrame(all_vector_rows)
    vec_path = OUTPUT_DIR / "metrics_by_vector.csv"
    vector_df.to_csv(vec_path, index=False, float_format="%.4f")
    print(f"\nSaved: {vec_path}")

    # ---- averages across all three vectors per (sheet, language) ----
    metric_cols = [
        "precision_before", "recall_before", "jaccard_before",
        "precision_after", "recall_after", "jaccard_after",
        "delta_precision", "delta_recall", "delta_jaccard",
    ]
    avg_rows = []
    for (sheet, lang), grp in vector_df.groupby(["sheet", "language"], sort=False):
        avg_row = {
            "sheet": sheet,
            "language": lang,
            "n_samples": int(grp["n_samples"].iloc[0]),
        }
        for col in metric_cols:
            avg_row[f"avg_{col}"] = round(float(grp[col].mean()), 4)
        avg_rows.append(avg_row)

    avg_df = pd.DataFrame(avg_rows)
    avg_path = OUTPUT_DIR / "metrics_avg.csv"
    avg_df.to_csv(avg_path, index=False, float_format="%.4f")
    print(f"Saved: {avg_path}")

    # ---- updated gold annotations ----
    print("\nBuilding updated gold annotations...")
    gold_df = build_gold_annotations(LUR_FILE, INPUT_FILE)
    gold_path = OUTPUT_DIR / "gold_annotations.csv"
    gold_df.to_csv(gold_path, index=False)
    print(f"Saved: {gold_path} ({len(gold_df)} records)")

    # ---- model coverage dataframe ----
    print("\nBuilding model coverage dataframe...")
    coverage_df = build_coverage_dataframe(LUR_FILE, INPUT_FILE)
    coverage_path = OUTPUT_DIR / "model_coverage.csv"
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Saved: {coverage_path}")
    covered_any = coverage_df[["gemini", "qwen", "claude"]].any(axis=1).sum()
    print(f"  {covered_any} / {len(coverage_df)} records covered by at least one model")

    # ---- model comparison by vector ----
    ranking_df, best_df = build_model_comparison(vector_df)
    ranking_path = OUTPUT_DIR / "model_ranking_by_vector.csv"
    ranking_df.to_csv(ranking_path, index=False, float_format="%.4f")
    print(f"\nSaved: {ranking_path}")

    # ---- console preview ----
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n=== Metrics by Vector ===")
    print(vector_df.to_string(index=False))
    print("\n=== Average Across Vectors ===")
    print(avg_df.to_string(index=False))
    print("\n=== Model Ranking by Vector (language=All, sorted by jaccard_after) ===")
    print(ranking_df.to_string(index=False))
    print("\n=== Best Model per Vector ===")
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()
