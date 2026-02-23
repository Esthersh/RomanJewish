import pandas as pd
import argparse
import ast
import numpy as np
import os


def parse_list_string(s):
    """Parses a string representation of a list into a set of strings."""
    if pd.isna(s) or s == '':
        return set()

    # Handle simple comma-separated strings (common in CSVs)
    if not s.strip().startswith('['):
        # Remove potential surrounding quotes/spaces and split
        return set(item.strip().strip("'").strip('"') for item in str(s).split(',') if item.strip())

    try:
        # Handle python list string representation e.g. "['1', '2']"
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return set(str(item).strip() for item in parsed)
    except (ValueError, SyntaxError):
        pass

    # Fallback: simple split if eval fails
    return set(item.strip().strip("'").strip('"') for item in str(s).replace('[', '').replace(']', '').split(',') if
               item.strip())


def eval_col(pred_col, gt_col, merged):
    precisions = []
    recalls = []
    f1s = []
    jaccard_indices = []
    if pred_col not in merged.columns:
        print(f"Warning: '{pred_col}' not found. Using 'matched_ids' if available.")

    for index, row in merged.iterrows():
        gt_ids = parse_list_string(row[gt_col])
        pred_ids = parse_list_string(row[pred_col])

        # Filter out NaN/None/'nan' strings
        gt_ids = {x for x in gt_ids if x and x.lower() != 'nan'}
        pred_ids = {x for x in pred_ids if x and x.lower() != 'nan'}

        intersection = gt_ids.intersection(pred_ids)
        union = gt_ids.union(pred_ids)

        tp = len(intersection)
        fp = len(pred_ids - gt_ids)
        fn = len(gt_ids - pred_ids)

        # Sample micro-metrics
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0

        # Jaccard
        jaccard = len(intersection) / len(union) if len(union) > 0 else 0.0

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        jaccard_indices.append(jaccard)

    # Add metrics to dataframe
    merged['precision'] = precisions
    merged['recall'] = recalls
    merged['f1_score'] = f1s
    merged['jaccard_index'] = jaccard_indices

    # Macro Average
    macro_p = np.mean(precisions) if precisions else 0
    macro_r = np.mean(recalls) if recalls else 0
    macro_f1 = np.mean(f1s) if f1s else 0
    mean_jaccard = np.mean(jaccard_indices) if jaccard_indices else 0

    print(f"Total Samples: {len(merged)}")
    print(f"Macro Precision: {macro_p:.4f}")
    print(f"Macro Recall:    {macro_r:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Mean Jaccard:    {mean_jaccard:.4f}")


def evaluate_keywords(results_path, gold_path):
    print(f"Loading results from: {results_path}")
    print(f"Loading gold standard from: {gold_path}")

    # Load DataFrames
    try:
        df_res = pd.read_csv(results_path)
        # Rename Text to text if needed
        if 'Text' in df_res.columns:
            df_res = df_res.rename(columns={'Text': 'text'})

        df_res['original_row'] = df_res['original_row'].apply(ast.literal_eval)

        # We NO LONGER extract 'Refference' here because it contains 9 corrupted IDs in your file.
        # We will extract 'Text' using .get() to avoid KeyErrors
        df_res["text"] = df_res['original_row'].apply(lambda x: x.get('Text', ''))

        df_gold = pd.read_csv(gold_path)
        if 'Text' in df_gold.columns:
            df_gold = df_gold.rename(columns={'Text': 'text'})

        # In df_gold drop rows with the value: "NULL" (safest way in newer Pandas)
        df_gold.dropna(subset=['Keywords'], inplace=True)

    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Join DataFrames
    print(f"Normalizing texts and joining on the 'text' column...")

    # FIX: Flatten ALL hidden \r\n and whitespace discrepancies using split() & join()
    df_res['text'] = df_res['text'].astype(str).apply(lambda x: ' '.join(x.split()))
    df_gold['text'] = df_gold['text'].astype(str).apply(lambda x: ' '.join(x.split()))

    missing_samples = df_gold[~df_gold['text'].isin(df_res['text'])]
    if len(missing_samples) > 0:
        print(f"Warning: Missing {len(missing_samples)} sample(s) during matching.")

    # We merge exclusively on 'text'.
    # The gold standard still contains your perfectly preserved, clean 'ref Code'.
    merged = pd.merge(
        df_res,
        df_gold[['text', 'KW Ids', 'Keywords', 'ref Code']],
        on='text',
        how='inner'
    )

    print(f"Matched {len(merged)} rows out of {len(df_res)} results.")

    if len(merged) == 0:
        print("No matches found. Check ID formatting.")
        return

    print("\n--- Evaluation (IDs) ---")

    pred_col = 'matched_ids'
    gt_col = 'KW Ids'

    eval_col(pred_col, gt_col, merged)

    print("\n--- Evaluation (names) ---")
    pred_col = 'matched_keywords'
    gt_col = 'Keywords'
    eval_col(pred_col, gt_col, merged)

    # Save to CSV
    output_filename = os.path.basename(results_path).replace('.csv', '_validated.csv')
    output_path = os.path.join(os.path.dirname(results_path), output_filename)

    try:
        merged.set_index("ref Code", inplace=True)
        cols = ["source_id", "group", "name", "text",
                "pred_ids", "pred_kwords", "suggested_kws",
                "gold_ids", "gold_kwords",
                "precision", "recall", "f1_score", "jaccard_index",
                ]
        cols_rename = {"matched_ids": "pred_ids",
                       "matched_keywords": "pred_kwords",
                       "suggested_kws": "suggested_kws",
                       "KW Ids": "gold_ids",
                       "Keywords": "gold_kwords"
                       }
        merged.rename(columns=cols_rename, inplace=True)
        merged = merged[cols]
        merged.to_csv(output_path)
        print(f"\nSaved validated results to: {output_path}")
    except Exception as e:
        print(f"Error saving output CSV: {e}")


if __name__ == "__main__":
    """
    --results_file
    "/home/esther/PycharmProjects/RomanJewish/batch_results_gpt4.csv"
        --results_file
    "/home/esther/PycharmProjects/RomanJewish/results/batch_results_gpt4_5shot.csv"
    
    --results_file "/home/esther/PycharmProjects/RomanJewish/results/annotated_5shot_gpt_keywords_stable.csv"  
    --gold_file
    "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv"
    """
    parser = argparse.ArgumentParser(description="Validate Keyword Classification Results")
    parser.add_argument("--results_file", required=True, help="Path to the results CSV file")
    parser.add_argument("--gold_file", required=True, help="Path to the gold standard CSV file")

    args = parser.parse_args()

    evaluate_keywords(args.results_file, args.gold_file)