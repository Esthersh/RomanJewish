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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
INPUT_FILE = Path("data/streamlit_app_output.xlsx")
LUR_FILE   = Path("data/LUR_annotations.csv")
OUTPUT_DIR = Path("results/annotation_report")
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
        return {str(int(value))}   # single numeric ID stored as float
    if isinstance(value, int):
        return {str(value)}
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none"):
        return set()
    return {item.strip().lower() for item in re.split(r",\s*", s) if item.strip()}


def prec_rec_jac(pred: set | None, gold: set | None) -> tuple[float, float, float]:
    """Return (precision, recall, jaccard).

    Returns (NaN, NaN, NaN) when either set is None (corrupted data)
    or both are empty (undefined metrics).
    """
    if pred is None or gold is None:
        return np.nan, np.nan, np.nan
    if not pred and not gold:
        return np.nan, np.nan, np.nan
    inter = len(pred & gold)
    union = len(pred | gold)
    p = inter / len(pred) if pred else 0.0
    r = inter / len(gold) if gold else 0.0
    j = inter / union    if union  else 0.0
    return p, r, j

# ---------------------------------------------------------------------------
# Per-sample metric computation
# ---------------------------------------------------------------------------

def compute_sample_metrics(row: pd.Series) -> dict:
    """Compute before/after P/R/J for keywords, fields, and index."""
    # --- keywords ---
    orig_kw = parse_set(row["orig_kw_ids"])
    gold_kw = parse_set(row["gold_kw_ids"])
    kept_kw = parse_set(row["kw_kept_ids"])
    kw_b    = prec_rec_jac(orig_kw, gold_kw)
    kw_a    = prec_rec_jac(orig_kw, kept_kw)

    # --- fields ---
    orig_fi      = parse_set(row["orig_field_ids"])
    gold_fi      = parse_set(row["gold_field_ids"])
    kept_fi      = parse_set(row["field_kept_ids"])
    miss_fi      = parse_set(row["field_miss_agreed_ids"])
    gold_fi_aft  = (kept_fi | miss_fi) if (kept_fi is not None and miss_fi is not None) else None
    fi_b         = prec_rec_jac(orig_fi, gold_fi)
    fi_a         = prec_rec_jac(orig_fi, gold_fi_aft)

    # --- index ---
    orig_ix      = parse_set(row["orig_index_terms"])
    gold_ix      = parse_set(row["gold_index_terms"])
    kept_ix      = parse_set(row["index_kept_terms"])
    miss_ix      = parse_set(row["index_miss_agreed_terms"])
    gold_ix_aft  = (kept_ix | miss_ix) if (kept_ix is not None and miss_ix is not None) else None
    ix_b         = prec_rec_jac(orig_ix, gold_ix)
    ix_a         = prec_rec_jac(orig_ix, gold_ix_aft)

    return {
        "kw_p_before": kw_b[0], "kw_r_before": kw_b[1], "kw_j_before": kw_b[2],
        "kw_p_after":  kw_a[0], "kw_r_after":  kw_a[1], "kw_j_after":  kw_a[2],
        "fi_p_before": fi_b[0], "fi_r_before": fi_b[1], "fi_j_before": fi_b[2],
        "fi_p_after":  fi_a[0], "fi_r_after":  fi_a[1], "fi_j_after":  fi_a[2],
        "ix_p_before": ix_b[0], "ix_r_before": ix_b[1], "ix_j_before": ix_b[2],
        "ix_p_after":  ix_a[0], "ix_r_after":  ix_a[1], "ix_j_after":  ix_a[2],
    }

# ---------------------------------------------------------------------------
# Sheet processing
# ---------------------------------------------------------------------------

def load_and_deduplicate(sheet_name: str) -> tuple[pd.DataFrame, int]:
    """Load sheet, keep last annotation per ref_id (latest date wins).
    Returns (deduplicated DataFrame, original row count).
    """
    df = pd.read_excel(INPUT_FILE, sheet_name=sheet_name)
    n_total = len(df)
    df["_date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values(["ref_id", "_date"], na_position="first")
    df = df.drop_duplicates(subset="ref_id", keep="last").reset_index(drop=True)
    return df.drop(columns=["_date"]), n_total


def process_sheet(sheet_name: str, lang_map: dict) -> pd.DataFrame:
    """Return per-sample DataFrame with computed metrics and language label."""
    df, n_total = load_and_deduplicate(sheet_name)

    df["language"] = df["ref_id"].map(lang_map).fillna("Unknown")

    records = []
    for _, row in df.iterrows():
        m = compute_sample_metrics(row)
        m["ref_id"]        = row["ref_id"]
        m["language"]      = row["language"]
        m["n_total_rows"]  = n_total
        records.append(m)

    return pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

VECTORS = [
    ("keywords", "kw"),
    ("fields",   "fi"),
    ("index",    "ix"),
]

def aggregate_group(grp: pd.DataFrame, sheet: str, language: str,
                    n_samples: int, n_dup_removed: int) -> list[dict]:
    """One summary row per vector for a (sheet, language) group."""
    rows = []
    for vec_name, prefix in VECTORS:
        row = {
            "sheet":         sheet,
            "language":      language,
            "n_samples":     n_samples,
            "n_dup_removed": n_dup_removed,
            "vector":        vec_name,
        }
        for metric, letter in [("precision", "p"), ("recall", "r"), ("jaccard", "j")]:
            b = float(np.nanmean(grp[f"{prefix}_{letter}_before"]))
            a = float(np.nanmean(grp[f"{prefix}_{letter}_after"]))
            row[f"{metric}_before"] = round(b, 4)
            row[f"{metric}_after"]  = round(a, 4)
            row[f"delta_{metric}"]  = round(a - b, 4)
        rows.append(row)
    return rows

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
        n_total    = int(per_sample["n_total_rows"].iloc[0])
        n_dedup    = len(per_sample)
        n_dup      = n_total - n_dedup

        for lang, grp in per_sample.groupby("language", sort=True):
            rows = aggregate_group(grp, sheet, lang, len(grp), n_dup)
            all_vector_rows.extend(rows)

        langs = per_sample["language"].value_counts().to_dict()
        print(f"  {sheet}: {n_dedup} samples ({n_dup} dups removed) | {langs}")

    # ---- per-vector table ----
    vector_df = pd.DataFrame(all_vector_rows)
    vec_path  = OUTPUT_DIR / "metrics_by_vector.csv"
    vector_df.to_csv(vec_path, index=False, float_format="%.4f")
    print(f"\nSaved: {vec_path}")

    # ---- averages across all three vectors per (sheet, language) ----
    metric_cols = [
        "precision_before", "recall_before", "jaccard_before",
        "precision_after",  "recall_after",  "jaccard_after",
        "delta_precision",  "delta_recall",  "delta_jaccard",
    ]
    avg_rows = []
    for (sheet, lang), grp in vector_df.groupby(["sheet", "language"], sort=False):
        avg_row = {
            "sheet":         sheet,
            "language":      lang,
            "n_samples":     int(grp["n_samples"].iloc[0]),
            "n_dup_removed": int(grp["n_dup_removed"].iloc[0]),
        }
        for col in metric_cols:
            avg_row[f"avg_{col}"] = round(float(grp[col].mean()), 4)
        avg_rows.append(avg_row)

    avg_df   = pd.DataFrame(avg_rows)
    avg_path = OUTPUT_DIR / "metrics_avg.csv"
    avg_df.to_csv(avg_path, index=False, float_format="%.4f")
    print(f"Saved: {avg_path}")

    # ---- console preview ----
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    print("\n=== Metrics by Vector ===")
    print(vector_df.to_string(index=False))
    print("\n=== Average Across Vectors ===")
    print(avg_df.to_string(index=False))


if __name__ == "__main__":
    main()
