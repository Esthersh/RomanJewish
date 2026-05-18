"""
Creates a merged JSON file from a non-analyzed CSV for testing the non-analyzed UI.
Predictions are taken from a reference merged JSON (matched by name), so the app
renders real labels instead of random IDs.

Usage:
    python scripts/create_non_analyzed_merged.py \
        --csv data/non_analyzed_fabricated.csv \
        --predictions results/prioritized/tosefta/merged_gemini_3_pro.json \
        --output_dir results/prioritized/tosefta
"""

import json
import sys
import os
import argparse

import pandas as pd

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.insert(0, os.path.join(project_root, "src"))


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


def create_merged_from_csv(csv_path, output_dir, predictions_file):
    df = pd.read_csv(csv_path)

    for col in ("Group", "Name"):
        if col in df.columns:
            df[col] = df[col].ffill()

    with open(predictions_file, encoding="utf-8") as f:
        pred_data = json.load(f)
    pred_by_name = {item["name"].strip(): item for item in pred_data}

    results = []
    for _, row in df.iterrows():
        ref_code = str(row.get("ref Code") or row.get("Calculated refCode") or "")
        name = str(row.get("Reference") or row.get("Name") or "").strip()
        text = str(row.get("Text") or "")

        pred = pred_by_name.get(name, {})
        matched_ids = pred.get("matched_ids", [])
        matched_field_ids = pred.get("matched_field_ids", [])
        index_terms = pred.get("index_terms", [])
        suggested_kws = pred.get("suggested_kws", [])
        matched_keywords = pred.get("matched_keywords", [])

        original_row = {k: _safe(v) for k, v in row.to_dict().items()}

        item = {
            "ref_id": ref_code,
            "source_id": str(_safe(row.get("SourceID")) or ""),
            "group": str(_safe(row.get("Group")) or ""),
            "name": name,
            "text": text,
            "original_row": original_row,
            "original_res": pred.get("original_res", {}),
            "matched_ids": matched_ids,
            "matched_keywords": matched_keywords,
            "suggested_kws": suggested_kws,
            "matched_field_ids": matched_field_ids,
            "suggested_fields": pred.get("suggested_fields", []),
            "matched_index_ids": pred.get("matched_index_ids", []),
            "index_terms": index_terms,
            "origin_file": csv_path,
        }
        results.append(item)
        if not pred:
            print(f"  WARNING: no prediction found for {name!r}")

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
    parser.add_argument("--predictions", required=True,
                        help="Merged JSON whose predictions to use (matched by name)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/prioritized/tosefta)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "results", "prioritized", "tosefta")

    create_merged_from_csv(args.csv, args.output_dir, args.predictions)
