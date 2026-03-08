import csv
import os

def fix_csv(filename):
    if not os.path.exists(filename):
        print(f"File {filename} not found.")
        return

    # Modern columns we want to ensure exist
    metrics_cols = [
        'results_filename', 'annotator', 'date', 'ref_id', 'source_id', 'text', 
        'group', 'name', 'original_matched', 'original_matched_ids', 'kept_ids', 
        'added_existing_ids', 'gold_ids', 'original_suggested', 'accepted_new_keywords', 
        'kept_keywords', 'added_keywords', 'orig_precision', 'orig_recall', 
        'orig_jaccard', 'mod_precision', 'mod_recall', 'mod_jaccard'
    ]

    rows = []
    with open(filename, mode='r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # We'll try to map existing columns to the new ones
        # This is a bit heuristics-based because some versions had different column orders
        for i, row in enumerate(reader):
            # Line 5 Saw 15, expected 13.
            # Row 5 in the view_file: mkw_gemini_3fp.json,Esther User,2026-03-04,000200330008000100000000,2,...
            # This row has 15 columns. Let's try to parse it.
            
            d = {}
            if len(row) == 13:
                # results_filename,ref_id,source_id,text,group,name,original_matched,kept_ids,added_existing_ids,original_suggested,accepted_new_keywords,kept_keywords,added_keywords
                cols = ['results_filename', 'ref_id', 'source_id', 'text', 'group', 'name', 'original_matched', 'kept_ids', 'added_existing_ids', 'original_suggested', 'accepted_new_keywords', 'kept_keywords', 'added_keywords']
                d = dict(zip(cols, row))
            elif len(row) == 15:
                # results_filename, annotator, date, ref_id, source_id, text, group, name, original_matched, kept_ids, added_existing_ids, original_suggested, accepted_new_keywords, kept_keywords, added_keywords
                cols = ['results_filename', 'annotator', 'date', 'ref_id', 'source_id', 'text', 'group', 'name', 'original_matched', 'kept_ids', 'added_existing_ids', 'original_suggested', 'accepted_new_keywords', 'kept_keywords', 'added_keywords']
                d = dict(zip(cols, row))
            elif len(row) >= 23:
                # Latest version (approximately)
                d = dict(zip(metrics_cols, row))
            else:
                # Unknown format, just pad it
                d = {f"col_{j}": val for j, val in enumerate(row)}
            
            rows.append(d)

    # Standardize all rows to have metrics_cols
    output_rows = []
    for d in rows:
        standardized = {col: d.get(col, '') for col in metrics_cols}
        output_rows.append(standardized)

    # Overwrite the file with the standardized format
    with open(filename, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=metrics_cols)
        writer.writeheader()
        writer.writerows(output_rows)
    
    print(f"Successfully fixed {len(output_rows)} rows in {filename}.")

if __name__ == "__main__":
    fix_csv("annotated_results.csv")
