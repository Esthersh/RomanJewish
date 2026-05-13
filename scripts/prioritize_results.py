import json
import os
import pandas as pd

results_dir = "/home/esther/PycharmProjects/RomanJewish/results"
final_dir = os.path.join(results_dir, "final_merged")
os.makedirs(final_dir, exist_ok=True)

csv_path = "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv"


def get_corpus_order(csv_path: str):
    try:
        df = pd.read_csv(csv_path, dtype=str)
        return {str(row['ref Code']).strip(): idx for idx, row in df.iterrows() if pd.notna(row.get('ref Code'))}
    except Exception as e:
        print(f"Error reading corpus CSV: {e}")
        return {}


models = ["claude_opus4_6"]  # ["gemini_3_pro", "qwen_3_5", "claude_opus4_6"]


def merge_json_files_with_origin(w_en_path, other_path, output_path):
    # Load both JSON files
    with open(w_en_path, 'r', encoding='utf-8') as f:
        w_en_data = json.load(f)

    with open(other_path, 'r', encoding='utf-8') as f:
        other_data = json.load(f)

    # Dictionary to store the unique samples by their ref_id
    merged_records = {}

    # 1. Load all records from the second JSON file (lower priority)
    for item in other_data:
        ref_id = str(item.get('ref_id'))
        # Mark where this record came from
        item['origin_file'] = other_path
        merged_records[ref_id] = item

    # 2. Load records from the 'w_en' JSON file (higher priority).
    # Overwrites existing records, including updating the 'origin_file' tag.
    for item in w_en_data:
        ref_id = str(item.get('ref_id'))
        # Mark where this record came from
        item['origin_file'] = w_en_path
        merged_records[ref_id] = item

    # Convert the dictionary values back to a list
    final_list = list(merged_records.values())

    # Save to the third JSON file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(final_list, f, ensure_ascii=False, indent=2)

    print(f"Merge complete! Saved {len(final_list)} unique samples to '{output_path}'.")


if __name__ == '__main__':
    order_map = get_corpus_order(csv_path)
    for m in models:
        w_en_file = f"../results/merged_w_en_{m}.json"
        other_file = f"../results/merged_{m}.json"
        output_file = f"../results/prioritized/merged_{m}.json"

        merge_json_files_with_origin(w_en_file, other_file, output_file)
