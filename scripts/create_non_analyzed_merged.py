"""
Creates a merged JSON file from a non-analyzed CSV for testing the non-analyzed UI.
Fabricated model predictions are added using real taxonomy IDs so the app can render labels.

Usage:
    python scripts/create_non_analyzed_merged.py \
        --csv data/non_analyzed_fabricated.csv \
        --output_dir results/prioritized/tosefta
"""

import json
import sys
import os
import random
import argparse

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))

from data_loader import DataLoader


def _safe(val):
    """Convert NaN / float NaN to None for JSON serialisation."""
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    return val


def create_merged_from_csv(csv_path, output_dir, keywords_file=None, fields_file=None, seed=42):
    random.seed(seed)

    df = pd.read_csv(csv_path)

    # Forward-fill Group / Name so rows without them get the parent value
    for col in ("Group", "Name"):
        if col in df.columns:
            df[col] = df[col].ffill()

    # Load taxonomy for realistic fabricated predictions
    loader = DataLoader()
    kw_ids, field_ids = [], []

    if keywords_file and os.path.exists(keywords_file):
        kws = loader.load_keywords(keywords_file)
        kw_ids = [str(k.id) for k in kws]

    if fields_file and os.path.exists(fields_file):
        fields = loader.load_judicial_fields(fields_file)
        field_ids = [str(f.id) for f in fields]

    results = []
    for _, row in df.iterrows():
        ref_code = str(row.get("ref Code") or row.get("Calculated refCode") or "")
        name = str(row.get("Reference") or row.get("Name") or "")
        text = str(row.get("Text") or "")
        english = str(row.get("English") or "")

        matched_ids = random.sample(kw_ids, min(5, len(kw_ids))) if kw_ids else []
        matched_field_ids = random.sample(field_ids, min(2, len(field_ids))) if field_ids else []
        index_terms = [w for w in text.split() if len(w) > 3][:5] or ["term_a", "term_b"]
        suggested_kws = []

        original_row = {k: _safe(v) for k, v in row.to_dict().items()}

        item = {
            "ref_id": ref_code,
            "source_id": str(_safe(row.get("SourceID")) or ""),
            "group": str(_safe(row.get("Group")) or ""),
            "name": name,
            "text": text,
            "original_row": original_row,
            "original_res": {},
            "matched_ids": matched_ids,
            "matched_keywords": [],
            "suggested_kws": suggested_kws,
            "matched_field_ids": matched_field_ids,
            "suggested_fields": [],
            "matched_index_ids": [],
            "index_terms": index_terms,
            "origin_file": csv_path,
        }
        results.append(item)

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(output_dir, f"merged_{base}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"Created {output_file} with {len(results)} items.")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a merged JSON from a non-analyzed CSV.")
    parser.add_argument("--csv", required=True, help="Path to the non-analyzed CSV file")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/prioritized/tosefta)")
    parser.add_argument("--keywords_file", default=None)
    parser.add_argument("--fields_file", default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "results", "prioritized", "tosefta")

    if args.keywords_file is None:
        kw_path = os.path.join(project_root, "data", "Keywords.csv")
        args.keywords_file = kw_path if os.path.exists(kw_path) else None

    if args.fields_file is None:
        tp_path = os.path.join(project_root, "data", "Topics.csv")
        args.fields_file = tp_path if os.path.exists(tp_path) else None

    create_merged_from_csv(
        args.csv, args.output_dir,
        args.keywords_file, args.fields_file,
        args.seed,
    )
