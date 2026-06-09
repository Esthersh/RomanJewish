import json
import os
import re
import sys
import argparse
import pandas as pd
from typing import List, Dict, Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scripts.sheets_loader import ID_COLUMN, load_samples_tab


RESULTS_DIR = "/Users/davidfram/RomanJewish/results/results_0_1"


NAME_COLUMN = "Reference"
TEXT_COLUMN = "Text"
LANGUAGE_COLUMN = "Language"


def _cell(row, col: str) -> str:
    val = row.get(col)
    if pd.isna(val):
        return ""
    return str(val)


def load_corpus() -> tuple[Dict[str, int], Dict[str, Dict[str, str]]]:
    """Return (order_map, metadata_map) keyed by Id from the Samples2Update tab.

    order_map:    {Id → row index}
    metadata_map: {Id → {language, name, text}}
    """
    df = load_samples_tab()
    for required in (ID_COLUMN, NAME_COLUMN, TEXT_COLUMN, LANGUAGE_COLUMN):
        if required not in df.columns:
            raise ValueError(f"Samples2Update is missing required column {required!r}. Found: {list(df.columns)}")

    order: Dict[str, int] = {}
    meta: Dict[str, Dict[str, str]] = {}
    for idx, row in df.iterrows():
        raw_id = row.get(ID_COLUMN)
        if pd.isna(raw_id):
            continue
        key = str(raw_id).strip()
        order[key] = idx
        meta[key] = {
            "language": _cell(row, LANGUAGE_COLUMN),
            "name": _cell(row, NAME_COLUMN),
            "text": _cell(row, TEXT_COLUMN),
        }
    return order, meta


def parse_response(raw) -> Any:
    """Extract and parse JSON from a response that may include prose around a code block."""
    if not raw:
        return None
    text = str(raw)
    # Try to find a ```json ... ``` block anywhere in the text (model sometimes adds prose before it)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
    else:
        candidate = text.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        print(f"  Warning: could not parse response JSON for candidate: {e}")
        print(f"    Candidate preview: {repr(candidate[:120])}")
        return None


def load_files(paths: List[str]) -> Dict[str, Any]:
    """Load multiple JSON files and index by ref_id (last write wins on collision)."""
    by_ref: Dict[str, Any] = {}
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: file not found: {p}")
            continue
        with open(p, encoding="utf-8") as f:
            records = json.load(f)
        for rec in records:
            ref_id = str(rec.get("ref_id", "")).strip()
            if ref_id:
                by_ref[ref_id] = rec
        print(f"  Loaded {len(records)} records from {os.path.basename(p)}")
    return by_ref


def merge(results_dir: str, output_file: str, keyword_files=None, topics_files=None, index_files=None) -> None:
    print("Fetching corpus from Samples2Update tab...")
    order_map, meta_map = load_corpus()
    print(f"  {len(order_map)} ids loaded (with language/name/text metadata)")

    def _resolve(names):
        # filenames are taken relative to results_dir unless already absolute
        return [n if os.path.isabs(n) else os.path.join(results_dir, n) for n in names]

    # Defaults reproduce the original 0.1 batch layout; override to merge other runs
    # (e.g. a sequential keyword pass: keyword_files=["gemini_KEYWORDS_0_2_SEQ.json"]).
    kw_names = keyword_files or ["gemini_KEYWORDS_0_1_JTWC.json", "gemini_KEYWORDS_0_1_PI.json"]
    topics_names = topics_files or ["gemini_TOPICS_0_1_JTWC.json", "gemini_TOPICS_0_1_PI.json"]
    index_names = index_files or ["gemini_INDEX_0_1.json"]

    print("\nLoading KEYWORDS files...")
    kw_data = load_files(_resolve(kw_names))

    print("\nLoading TOPICS files...")
    topics_data = load_files(_resolve(topics_names))

    print("\nLoading INDEX file...")
    index_data = load_files(_resolve(index_names))

    all_refs = set(kw_data) | set(topics_data) | set(index_data)
    print(f"\nTotal unique ref_ids across all files: {len(all_refs)}")

    merged = []
    for ref_id in all_refs:
        meta = meta_map.get(ref_id, {})
        record: Dict[str, Any] = {
            "ref_id": ref_id,
            "name": meta.get("name", ""),
            "language": meta.get("language", ""),
            "text": meta.get("text", ""),
        }

        if ref_id in kw_data:
            parsed = parse_response(kw_data[ref_id].get("response", "[]"))
            record["keywords"] = parsed if parsed is not None else []
        else:
            record["keywords"] = []

        if ref_id in topics_data:
            parsed = parse_response(topics_data[ref_id].get("response", "[]"))
            record["topics"] = parsed if parsed is not None else []
        else:
            record["topics"] = []

        if ref_id in index_data:
            parsed = parse_response(index_data[ref_id].get("response", "[]"))
            record["index_terms"] = parsed if parsed is not None else []
        else:
            record["index_terms"] = []

        merged.append(record)

    merged.sort(key=lambda x: order_map.get(x["ref_id"], float("inf")))

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"\nWrote {len(merged)} merged records to {output_file}")

    missing_kw = sum(1 for r in merged if not r["keywords"])
    missing_topics = sum(1 for r in merged if not r["topics"])
    missing_index = sum(1 for r in merged if not r["index_terms"])
    print(f"  Missing keywords:    {missing_kw}")
    print(f"  Missing topics:      {missing_topics}")
    print(f"  Missing index_terms: {missing_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge per-vector result JSONs into a single merged output.")
    parser.add_argument("--results_dir", default=RESULTS_DIR)
    parser.add_argument("--output", default=os.path.join(RESULTS_DIR, "merged_0_1.json"))
    parser.add_argument("--keywords-files", nargs="+", default=None,
                        help="Keyword result file(s), relative to --results_dir or absolute. "
                             "Default: gemini_KEYWORDS_0_1_JTWC.json + _PI.json. "
                             "Sequential run: gemini_KEYWORDS_0_2_SEQ.json")
    parser.add_argument("--topics-files", nargs="+", default=None,
                        help="Topics result file(s). Default: gemini_TOPICS_0_1_JTWC.json + _PI.json")
    parser.add_argument("--index-files", nargs="+", default=None,
                        help="Index result file(s). Default: gemini_INDEX_0_1.json")
    args = parser.parse_args()

    merge(args.results_dir, args.output,
          keyword_files=args.keywords_files,
          topics_files=args.topics_files,
          index_files=args.index_files)
