import json
import os
import argparse
import pandas as pd
from typing import List


def get_corpus_order(csv_path: str):
    try:
        df = pd.read_csv(csv_path, dtype=str)
        return {str(row['ref Code']).strip(): idx for idx, row in df.iterrows() if pd.notna(row.get('ref Code'))}
    except Exception as e:
        print(f"Error reading corpus CSV: {e}")
        return {}


def get_key(item):
    k = item.get('original_row', {}).get('ref Code') or item.get('ref_id') or item.get('source_id')
    return str(k).strip() if k else None


def load_and_merge(paths: List[str]) -> List[dict]:
    combined = []
    for p in paths:
        if not os.path.exists(p):
            print(f"Warning: File not found {p}")
            continue
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                combined.extend(data)
        except Exception as e:
            print(f"Error loading {p}: {e}")
    return combined


def do_horizontal_merge(kw_data, fields_data, index_data, output_file, order_map):
    merged_data = {}

    # Merge KW
    for item in kw_data:
        key = get_key(item)
        if key:
            merged_data[key] = item

    # Merge Fields
    for item in fields_data:
        key = get_key(item)
        if not key: continue
        if key in merged_data:
            merged_data[key].update({
                'matched_field_ids': item.get('matched_field_ids', []),
                'suggested_fields': item.get('suggested_fields', [])
            })
        else:
            merged_data[key] = item

    # Merge Index
    for item in index_data:
        key = get_key(item)
        if not key: continue
        if key in merged_data:
            merged_data[key].update({
                'matched_index_ids': item.get('matched_index_ids', []),
                'index_terms': item.get('index_terms', [])
            })
        else:
            merged_data[key] = item

    final_list = list(merged_data.values())
    if not final_list:
        print(f"Skipping {output_file} - No data to merge.")
        return

    # Sort according to corpus order
    final_list.sort(key=lambda x: order_map.get(str(get_key(x)), float('inf')))

    # Create dir if not exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(final_list)} items into {output_file}")


def merge_all_results(results_dir: str, corpus_csv: str, en_only: bool = False):
    order_map = get_corpus_order(corpus_csv)
    # models = ["gemini_3_pro", "qwen_3_5", "claude_opus4_6"]
    models = ["gemini", "qwen", "claude"]

    for m in models:
        print(f"\n--- Processing Model: {m} ---")

        # 1. Base files (Plain + Context) -> merged_{model}.json
        if not en_only:
            kw_base = [
                os.path.join(results_dir, f"keywords_{m}.json"),
                os.path.join(results_dir, f"keywords_context_{m}.json")
            ]
            fields_base = [
                os.path.join(results_dir, f"fields_{m}.json"),
                os.path.join(results_dir, f"fields_context_{m}.json")
            ]
            index_base = [
                os.path.join(results_dir, f"index_v1_{m}.json"),
                os.path.join(results_dir, f"index_v1_context_{m}.json")
            ]

            print(f"Merging Base (Plain + Context) for {m}...")
            do_horizontal_merge(
                load_and_merge(kw_base),
                load_and_merge(fields_base),
                load_and_merge(index_base),
                os.path.join(results_dir, f"merged_{m}.json"),
                order_map
            )
        else:
            print(f"Skipping Base files for {m} (--en_only flag active)")

        # 2. English translated versions -> merged_w_en_{model}.json
        # kw_wen = [os.path.join(results_dir, f"keywords_w_en_context_{m}.json")]
        # fields_wen = [os.path.join(results_dir, f"fields_w_en_context_{m}.json")]
        # index_wen = [os.path.join(results_dir, f"index_w_en_v1_context_{m}.json")]
        kw_wen = [os.path.join(results_dir, f"tosefta_keywords_w_en_context_{m}.json")]
        fields_wen = [os.path.join(results_dir, f"tosefta_fields_w_en_context_{m}.json")]
        index_wen = [os.path.join(results_dir, f"tosefta_index_w_en_v1_context_{m}.json")]

        print(f"Merging W_EN (Translated versions) for {m}...")
        do_horizontal_merge(
            load_and_merge(kw_wen),
            load_and_merge(fields_wen),
            load_and_merge(index_wen),
            os.path.join(results_dir, f"merged_w_en_{m}.json"),
            order_map
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge 3-vector result files across batch subsets.")
    parser.add_argument("--results_dir", default="/Users/davidfram/RomanJewish/results",
                        help="Directory containing result JSON files")
    parser.add_argument("--corpus_csv", default="'/Users/davidfram/RomanJewish/data/LUR sample corpus.xlsx - Samples.csv'",
                        help="Original Corpus CSV for ordering")
    parser.add_argument("--en_only", action="store_true",
                        help="Skip base files and merge ONLY the English translated files")

    args = parser.parse_args()
    merge_all_results(args.results_dir, args.corpus_csv, args.en_only)

