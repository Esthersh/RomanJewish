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

import argparse
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


def _ids_to_names(ids: set | None, lookup: dict) -> set[str] | None:
    """Resolve a set of ID-like strings to lowercase name strings via ``lookup``.

    Falls back to the original token (lowercased) when not in the lookup or
    not parseable as int. Returns None if ``ids`` is None.
    """
    if ids is None:
        return None
    out: set[str] = set()
    for i in ids:
        i_str = str(i).strip()
        if not i_str:
            continue
        try:
            resolved = lookup.get(int(i_str), i_str)
        except (ValueError, TypeError):
            resolved = i_str
        out.add(str(resolved).strip().lower())
    return out


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

def compute_sample_metrics(row: pd.Series,
                            kw_names: dict | None = None,
                            fi_names: dict | None = None,
                            external_gold: dict[str, dict[str, set]] | None = None) -> dict:
    """Compute before/after P/R/J for keywords, fields, and index.

    When ``external_gold`` is provided, the "after" comparison uses the
    pre-built gold (sets of lowercase names) keyed by ``row["name"]``, and
    keyword/field predictions are resolved from IDs to names via
    ``kw_names`` / ``fi_names``. Otherwise, "after" uses the per-sheet
    kept / miss-agreed columns (the original behavior).
    """
    use_external = external_gold is not None
    ref_id = row["name"]

    # --- keywords ---
    pred_kw = parse_set(row["orig_kw_ids"])
    gold_kw = parse_set(row["gold_kw_ids"])
    kw_b = compute_metrics_per_sample(pred_kw, gold_kw)
    if use_external:
        pred_kw_names = _ids_to_names(pred_kw, kw_names or {})
        kept_kw = external_gold.get(ref_id, {}).get("kw")
        kw_a = compute_metrics_per_sample(pred_kw_names, kept_kw)
    else:
        kept_kw = parse_set(row["kw_kept_ids"])
        kw_a = compute_metrics_per_sample(pred_kw, kept_kw)

    # --- fields ---
    pred_fi = parse_set(row["orig_field_ids"])
    gold_fi = parse_set(row["gold_field_ids"])
    fi_b = compute_metrics_per_sample(pred_fi, gold_fi)
    if use_external:
        pred_fi_names = _ids_to_names(pred_fi, fi_names or {})
        gold_fi_aft = external_gold.get(ref_id, {}).get("fi")
        fi_a = compute_metrics_per_sample(pred_fi_names, gold_fi_aft)
    else:
        kept_fi = parse_set(row["field_kept_ids"])
        miss_fi = parse_set(row["field_miss_agreed_ids"])
        gold_fi_aft = (kept_fi | miss_fi) if (kept_fi is not None and miss_fi is not None) else None
        fi_a = compute_metrics_per_sample(pred_fi, gold_fi_aft)

    # --- index ---
    pred_ix = parse_set(row["orig_index_terms"])
    gold_ix = parse_set(row["gold_index_terms"])
    ix_b = compute_metrics_per_sample(pred_ix, gold_ix)
    if use_external:
        gold_ix_aft = external_gold.get(ref_id, {}).get("ix")
        ix_a = compute_metrics_per_sample(pred_ix, gold_ix_aft)
    else:
        kept_ix = parse_set(row["index_kept_terms"])
        miss_ix = parse_set(row["index_miss_agreed_terms"])
        gold_ix_aft = (kept_ix | miss_ix) if (kept_ix is not None and miss_ix is not None) else None
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


def process_sheet(sheet_name: str, lang_map: dict,
                  external_gold: dict | None = None,
                  kw_names: dict | None = None,
                  fi_names: dict | None = None) -> pd.DataFrame:
    """Return per-sample DataFrame with computed metrics and language label."""
    df = load_and_deduplicate(sheet_name)

    names = sorted(df["name"].dropna().astype(str).tolist())
    # print(f"  [{sheet_name}] {len(names)} unique names:")
    # for n in names:
    #     print(f"    {n}")

    df["language"] = df["name"].map(lang_map).fillna("Unknown")

    records = []
    for _, row in df.iterrows():
        m = compute_sample_metrics(row, kw_names=kw_names, fi_names=fi_names,
                                    external_gold=external_gold)
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


def build_gold_sets(excel_path: Path, model: str | None,
                    kw_names: dict, fi_names: dict) -> dict[str, dict[str, set]]:
    """Return {ref_id: {'kw': set, 'fi': set, 'ix': set}} of lowercase name strings.

    'kw'  ← kw_kept_ids resolved via ``kw_names``
    'fi'  ← (field_kept_ids ∪ field_miss_agreed_ids) resolved via ``fi_names``
    'ix'  ← (index_kept_terms ∪ index_miss_agreed_terms), already text

    When ``model`` is None, the gold is the union across all model sheets;
    otherwise it is the union across MODEL_SHEETS[model].
    """
    if model is None:
        sheets = [s for sheets in MODEL_SHEETS.values() for s in sheets]
    elif model in MODEL_SHEETS:
        sheets = MODEL_SHEETS[model]
    else:
        raise ValueError(f"Unknown model {model!r}. Choose from {sorted(MODEL_SHEETS)}.")

    raw_kw: dict[str, set] = {}
    raw_fi: dict[str, set] = {}
    raw_ix: dict[str, set] = {}

    for sheet in sheets:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df["name"] = df["name"].str.strip()
        df = df.drop_duplicates(subset="name", keep="last")
        for _, row in df.iterrows():
            ref_id = row["name"]
            if pd.isna(ref_id):
                continue
            raw_kw[ref_id] = raw_kw.get(ref_id, set()) | _safe_set(row["kw_kept_ids"])
            raw_fi[ref_id] = (raw_fi.get(ref_id, set())
                              | _safe_set(row["field_kept_ids"])
                              | _safe_set(row["field_miss_agreed_ids"]))
            raw_ix[ref_id] = (raw_ix.get(ref_id, set())
                              | _safe_set(row["index_kept_terms"])
                              | _safe_set(row["index_miss_agreed_terms"]))

    gold: dict[str, dict[str, set]] = {}
    for ref_id in set(raw_kw) | set(raw_fi) | set(raw_ix):
        gold[ref_id] = {
            "kw": _ids_to_names(raw_kw.get(ref_id, set()), kw_names),
            "fi": _ids_to_names(raw_fi.get(ref_id, set()), fi_names),
            "ix": {str(t).strip().lower() for t in raw_ix.get(ref_id, set()) if str(t).strip()},
        }
    return gold


def build_gold_annotations(lur_path: Path, excel_path: Path,
                           model: str | None = None) -> pd.DataFrame:
    """Return a DataFrame [ref_id, text, new_gold_keywords, new_gold_fields, new_gold_index].

    new_gold_keywords  = kw_kept_ids  → resolved to keyword names
    new_gold_fields    = field_kept_ids ∪ field_miss_agreed_ids  → resolved to topic names
    new_gold_index     = index_kept_terms ∪ index_miss_agreed_terms  (already text)

    When ``model`` is given, values are unioned across the sheets belonging to
    that model (MODEL_SHEETS[model]). When ``model`` is None, values are unioned
    across all model sheets. Rows follow LUR order and are restricted to
    Analyzed == 'y' records that appear in at least one of the relevant sheets.
    """
    if model is None:
        sheets = [s for sheets in MODEL_SHEETS.values() for s in sheets]
    elif model in MODEL_SHEETS:
        sheets = MODEL_SHEETS[model]
    else:
        raise ValueError(f"Unknown model {model!r}. Choose from {sorted(MODEL_SHEETS)}.")

    kw_names = pd.read_csv(KEYWORDS_FILE).set_index("Id")["Keyword"].to_dict()
    fi_names = pd.read_csv(TOPICS_FILE).set_index("Id")["Topic"].to_dict()

    lur = pd.read_csv(lur_path)
    lur = lur[lur["Analyzed [y/n]"].str.strip().str.lower() == "y"].reset_index(drop=True)

    gold_kw: dict[str, set] = {}
    gold_fi: dict[str, set] = {}
    gold_ix: dict[str, set] = {}
    gold_kw_new: dict[str, set] = {}
    ref_text: dict[str, str] = {}
    has_en_translation: set[str] = set()

    for sheet in sheets:
        df = pd.read_excel(excel_path, sheet_name=sheet)
        df["name"] = df["name"].str.strip()
        df = df.drop_duplicates(subset="name", keep="last")
        is_en_sheet = sheet.startswith("w_en_")
        for _, row in df.iterrows():
            ref_id = row["name"]
            if pd.isna(ref_id):
                continue
            gold_kw[ref_id] = gold_kw.get(ref_id, set()) | _safe_set(row["kw_kept_ids"])
            gold_fi[ref_id] = gold_fi.get(ref_id, set()) | _safe_set(row["field_kept_ids"]) | _safe_set(row["field_miss_agreed_ids"])
            gold_ix[ref_id] = gold_ix.get(ref_id, set()) | _safe_set(row["index_kept_terms"]) | _safe_set(row["index_miss_agreed_terms"])
            gold_kw_new[ref_id] = gold_kw_new.get(ref_id, set()) | _safe_set(row.get("kw_accepted_new"))
            if ref_id not in ref_text and pd.notna(row.get("text")):
                ref_text[ref_id] = str(row["text"])
            if is_en_sheet:
                has_en_translation.add(ref_id)

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
            "ref_id":                  ref_id,
            "text":                    ref_text.get(ref_id, ""),
            "language":                row.get("Language", ""),
            "english":                 row.get("English", ""),
            "has_english_translation": ref_id in has_en_translation,
            "new_gold_keywords":       resolve(gold_kw.get(ref_id, set()), kw_names),
            "new_gold_fields":         resolve(gold_fi.get(ref_id, set()), fi_names),
            "new_gold_index":          ", ".join(sorted(gold_ix.get(ref_id, set()))),
            "new_keyword_suggestions": ", ".join(sorted(gold_kw_new.get(ref_id, set()))),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SHEETS),
        default=None,
        help="Model whose annotator decisions define the gold standard. "
             "If omitted (and --flex_gold is not set), gold is the union of all models.",
    )
    parser.add_argument(
        "--flex_gold",
        action="store_true",
        help="Build a separate gold annotations file per model "
             "(in addition to any --model / union gold).",
    )
    args = parser.parse_args()

    if not INPUT_FILE.exists():
        sys.exit(f"Input file not found: {INPUT_FILE}")
    if not LUR_FILE.exists():
        sys.exit(f"LUR annotations file not found: {LUR_FILE}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    lang_map = build_language_map(LUR_FILE)

    kw_names = pd.read_csv(KEYWORDS_FILE).set_index("Id")["Keyword"].to_dict()
    fi_names = pd.read_csv(TOPICS_FILE).set_index("Id")["Topic"].to_dict()

    # Determine which gold each sheet's "after" comparison should use.
    sheet_to_gold: dict[str, dict | None] = {sheet: None for sheet in SHEETS}
    if args.flex_gold:
        print("Building per-model golds for 'after' comparison...")
        per_model_gold = {
            m: build_gold_sets(INPUT_FILE, m, kw_names, fi_names) for m in MODEL_SHEETS
        }
        for m, sheets in MODEL_SHEETS.items():
            for sheet in sheets:
                sheet_to_gold[sheet] = per_model_gold[m]
    elif args.model:
        print(f"Building '{args.model}' gold for 'after' comparison...")
        shared_gold = build_gold_sets(INPUT_FILE, args.model, kw_names, fi_names)
        for sheet in SHEETS:
            sheet_to_gold[sheet] = shared_gold

    all_vector_rows: list[dict] = []

    print("Processing sheets...")
    for sheet in SHEETS:
        per_sample = process_sheet(sheet, lang_map,
                                    external_gold=sheet_to_gold.get(sheet),
                                    kw_names=kw_names, fi_names=fi_names)
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

    gold_targets: list[str | None] = []
    if args.flex_gold:
        gold_targets.extend(MODEL_SHEETS.keys())
    if args.model and args.model not in gold_targets:
        gold_targets.append(args.model)
    if not gold_targets:
        gold_targets.append(None)  # union of all models

    for target in gold_targets:
        gold_df = build_gold_annotations(LUR_FILE, INPUT_FILE, target)
        filename = f"gold_annotations_{target}.csv" if target else "gold_annotations.csv"
        gold_path = OUTPUT_DIR / filename
        gold_df.to_csv(gold_path, index=False)
        scope = target if target else "union"
        print(f"  Saved: {gold_path} ({len(gold_df)} records, scope={scope})")

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
