import json
import os
import argparse
from typing import List, Dict, Any

def merge_files(kw_file: str, fields_file: str, index_file: str, output_file: str):
    print(f"Merging files:\n  KW: {kw_file}\n  Fields: {fields_file}\n  Index: {index_file}")

    def load_json(path):
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    kw_data = load_json(kw_file)
    fields_data = load_json(fields_file)
    index_data = load_json(index_file)

    # Use 'ref Code' from original_row as the unique key for merging text segments
    def get_key(item):
        return item.get('original_row', {}).get('ref Code') or item.get('ref_id') or item.get('source_id')

    merged_data = {}

    # Initial load from KW (usually contains the most data or is our primary reference)
    for item in kw_data:
        key = get_key(item)
        merged_data[key] = item

    # Merge Fields
    for item in fields_data:
        key = get_key(item)
        if key in merged_data:
            # Add fields specific data
            merged_data[key].update({
                'matched_field_ids': item.get('matched_field_ids', []),
                'suggested_fields': item.get('suggested_fields', []) # If available
            })
        else:
            merged_data[key] = item

    # Merge Index
    for item in index_data:
        key = get_key(item)
        if key in merged_data:
            # Add index specific data
            merged_data[key].update({
                'matched_index_ids': item.get('matched_index_ids', []),
                'index_terms': item.get('index_terms', [])
            })
        else:
            merged_data[key] = item

    # Convert back to list
    final_list = list(merged_data.values())

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, indent=2, ensure_ascii=False)

    print(f"Merged {len(final_list)} items into {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge 3-vector result files.")
    parser.add_argument("--model", required=True, help="Model name (e.g., gemini_3pro)")
    parser.add_argument("--results_dir", default="results", help="Directory containing result files")
    
    args = parser.parse_args()
    
    kw_path = os.path.join(args.results_dir, f"kw_{args.model}.json")
    fields_path = os.path.join(args.results_dir, f"fields_{args.model}.json")
    index_path = os.path.join(args.results_dir, f"index_{args.model}.json")
    output_path = os.path.join(args.results_dir, f"merged_{args.model}.json")
    
    merge_files(kw_path, fields_path, index_path, output_path)
