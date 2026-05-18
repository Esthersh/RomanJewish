"""
Creates a merged JSON that combines non-analyzed CSV rows with predictions from a
reference merged JSON (matched by name).

Two modes:
  1. Default: output only the rows from --csv (14 records), with predictions
     taken from --predictions matched by name.
  2. --merge_analyzed: output all records from --predictions (89), replacing
     the records whose name matches a CSV row with their non-analyzed version
     (Analyzed [y/n] = 'n') and keeping the rest unchanged.

Usage:
    # 14 non-analyzed records only
    python scripts/create_non_analyzed_merged.py \
        --csv data/non_analyzed_fabricated.csv \
        --predictions results/prioritized/tosefta/merged_gemini_3_pro.json \
        --output_dir results/prioritized/tosefta

    # 89 records: 14 non-analyzed + 75 original analyzed
    python scripts/create_non_analyzed_merged.py \
        --csv data/non_analyzed_fabricated.csv \
        --predictions results/prioritized/tosefta/merged_gemini_3_pro.json \
        --output_dir results/prioritized/tosefta \
        --merge_analyzed
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


def _non_analyzed_item(row, pred):
    """Build one result item from a CSV row, using predictions from pred dict."""
    ref_code = str(row.get("ref Code") or row.get("Calculated refCode") or "")
    name = str(row.get("Reference") or row.get("Name") or "").strip()
    original_row = {k: _safe(v) for k, v in row.to_dict().items()}
    return {
        "ref_id": ref_code,
        "source_id": str(_safe(row.get("SourceID")) or ""),
        "group": str(_safe(row.get("Group")) or ""),
        "name": name,
        "text": str(row.get("Text") or ""),
        "original_row": original_row,
        "original_res": pred.get("original_res", {}),
        "matched_ids": pred.get("matched_ids", []),
        "matched_keywords": pred.get("matched_keywords", []),
        "suggested_kws": pred.get("suggested_kws", []),
        "matched_field_ids": pred.get("matched_field_ids", []),
        "suggested_fields": pred.get("suggested_fields", []),
        "matched_index_ids": pred.get("matched_index_ids", []),
        "index_terms": pred.get("index_terms", []),
        "origin_file": pred.get("origin_file", ""),
    }


def create_merged_from_csv(csv_path, output_dir, predictions_file, merge_analyzed=False):
    df = pd.read_csv(csv_path)
    for col in ("Group", "Name"):
        if col in df.columns:
            df[col] = df[col].ffill()

    with open(predictions_file, encoding="utf-8") as f:
        pred_data = json.load(f)
    pred_by_name = {item["name"].strip(): item for item in pred_data}

    # Build map from name -> CSV row for quick lookup
    csv_by_name = {}
    for _, row in df.iterrows():
        name = str(row.get("Reference") or row.get("Name") or "").strip()
        csv_by_name[name] = row

    if merge_analyzed:
        # Walk the predictions file in order; replace overlapping records with
        # their non-analyzed version, keep everything else unchanged.
        results = []
        for item in pred_data:
            name = item["name"].strip()
            if name in csv_by_name:
                results.append(_non_analyzed_item(csv_by_name[name], item))
            else:
                results.append(item)
    else:
        # Output only the CSV rows (non-analyzed), predictions matched by name.
        results = []
        for _, row in df.iterrows():
            name = str(row.get("Reference") or row.get("Name") or "").strip()
            pred = pred_by_name.get(name, {})
            if not pred:
                print(f"  WARNING: no prediction found for {name!r}")
            results.append(_non_analyzed_item(row, pred))

    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]
    output_file = os.path.join(output_dir, f"merged_{base}.json")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    n_non = sum(1 for r in results if r.get("original_row", {}).get("Analyzed [y/n]") == "n")
    print(f"Created {output_file}: {len(results)} records ({n_non} non-analyzed, {len(results)-n_non} analyzed).")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to the non-analyzed CSV file")
    parser.add_argument("--predictions", required=True,
                        help="Merged JSON used as prediction source (matched by name)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: results/prioritized/tosefta)")
    parser.add_argument("--merge_analyzed", action="store_true",
                        help="Include all analyzed records from --predictions alongside the non-analyzed ones")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(project_root, "results", "prioritized", "tosefta")

    create_merged_from_csv(args.csv, args.output_dir, args.predictions, args.merge_analyzed)
