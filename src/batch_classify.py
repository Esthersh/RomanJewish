import argparse
import json
import os
import sys
from tqdm import tqdm
from src.data_loader import DataLoader
from src.classifier import Classifier
from src.keyword_manager import KeywordManager
from src.data_loader import Keyword
import pandas as pd


def main():
    """
--prompt_file
"/home/esther/PycharmProjects/RomanJewish/prompts/default.py"
--prompt_k
"CLASSIFICATION_PROMPT"
--keywords_csv
"/home/esther/antigravity/RomanJewish/Keywords_05022026.csv"
--corpus_xlsx "/home/esther/antigravity/RomanJewish/LUR sample corpus.xlsx"
--provider
openai
--api_key
"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"
--model
"gpt-4.1-2025-04-14"
--temperature
0
--output_file
"batch_results_gpt4.json"

 :return:
    """
    parser = argparse.ArgumentParser(description="Run RomanJewish Classification Batch")
    parser.add_argument("--provider", required=True, choices=["gemini", "openai", "qwen"], help="LLM Provider")
    parser.add_argument("--api_key", required=True, help="API Key for the provider")
    parser.add_argument("--prompt_file", default="prompts/default.py", help="Path to prompt file")
    parser.add_argument("--prompt_k", default="CLASSIFICATION_PROMPT", help="Name of the prompt variable to use")
    parser.add_argument("--keywords_csv", default="Keywords_05022026.csv", help="Path to keywords CSV")
    parser.add_argument("--corpus_xlsx", default="LUR sample corpus.xlsx", help="Path to corpus Excel")
    parser.add_argument("--output_file", default="batch_results.json", help="Output JSON file for results")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    # Model config args
    parser.add_argument("--model", type=str, help="Model signature/name (e.g. gpt-4-turbo)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-P for generation")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    # Load Data
    print("Loading data...")
    loader = DataLoader()
    keywords = loader.load_keywords(args.keywords_csv)
    corpus = loader.load_corpus(args.corpus_xlsx)

    if args.limit:
        corpus = corpus[:args.limit]
        print(f"Limiting to {args.limit} samples.")

    print(f"Loaded {len(keywords)} keywords and {len(corpus)} samples.")

    # Init Classifier
    try:
        classifier = Classifier(
            provider=args.provider,
            api_key=args.api_key,
            prompt_path=args.prompt_file,
            prompt_name=args.prompt_k,
            model_name=args.model,
            temperature=args.temperature,
            top_p=args.top_p,
            debug=args.debug
        )
    except Exception as e:
        print(f"Error initializing classifier: {e}")
        sys.exit(1)

    # Keyword Manager (for tracking new suggestions globally, though batch usually just records them)
    # in this phase we just record what the LLM says.

    results = []
    processed_ids = set()

    # Check if output file exists and load existing results
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, 'r') as f:
                existing_results = json.load(f)
                if isinstance(existing_results, list):
                    results = existing_results
                    processed_ids = {str(item.get("source_id")) for item in results if "source_id" in item}
                    print(f"Loaded {len(results)} existing results. Skipping {len(processed_ids)} already processed samples.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {args.output_file}. Starting fresh.")
        except Exception as e:
            print(f"Warning: Error reading {args.output_file}: {e}. Starting fresh.")

    print("Starting classification...")
    for sample in tqdm(corpus):
        if str(sample.source_id) in processed_ids:
            continue

        try:
            metadata = {
                "source_name": sample.source_name,
                "group": sample.group,
                "name": sample.name,
                "language": sample.language
            }
            matched_ids, suggested_kws, full_res = classifier.classify(sample.text_en, keywords, metadata)

            # Resolve IDs to Names
            # specific helper to find name by id
            kw_map = {str(k.id): k.name for k in keywords}
            matched_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in matched_ids]

            result_entry = {
                "source_id": sample.source_id,
                "group": sample.group,
                "name": sample.name,
                "text_en": sample.text_en,
                "original_row": sample.original_row,  # Keep original metadata
                "matched_ids": matched_ids,
                "matched_keywords": matched_names,
                "suggested_kws": suggested_kws,
                "original_res": full_res,
            }
            results.append(result_entry)

            # Save results iteratively
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)

            # update the prompt to include the suggested keywords
            if suggested_kws:
                # Find max ID to assign new IDs
                max_id = max([k.id for k in keywords]) if keywords else 0
                new_keywords_added = []
                # Find or create "Additional Keywords" root category
                additional_root = next((k for k in keywords if k.name == "Additional Keywords" and k.level == 0), None)
                if not additional_root:
                    max_id += 1
                    additional_root = Keyword(
                        id=max_id,
                        name="Additional Keywords",
                        level=0,
                        parent_id=None,
                        full_path="Additional Keywords",
                        indented_name="Additional Keywords"
                    )
                    keywords.append(additional_root)
                    new_keywords_added.append(additional_root)
                
                # Add new keywords under "Additional Keywords"
                for new_kw_name in suggested_kws:
                    # Check if already exists to avoid duplicates
                    if not any(k.name.lower() == new_kw_name.lower() for k in keywords):
                        max_id += 1
                        new_keyword = Keyword(
                            id=max_id,
                            name=new_kw_name,
                            level=1,
                            parent_id=additional_root.id,
                            full_path=f"Additional Keywords > {new_kw_name}",
                            indented_name=f"  {new_kw_name}"
                        )
                        keywords.append(new_keyword)
                        new_keywords_added.append(new_keyword)
                        if args.debug:
                            print(f"[DEBUG] Added new keyword: {new_kw_name} (ID: {max_id})")

                # save the additional keywords in the CSV files with the original list
                if new_keywords_added:
                    new_rows = []
                    for kw in new_keywords_added:
                        new_rows.append({
                            'Id': kw.id,
                            'Keyword': kw.name,
                            'Parent KW Id': kw.parent_id,
                            'Indented Keywords': kw.indented_name,
                            'Full Path': kw.full_path,
                            'Level': kw.level
                        })
                    
                    df_new = pd.DataFrame(new_rows)
                    try:
                        df_new.to_csv(args.keywords_csv, mode='a', header=False, index=False)
                    except Exception as e:
                        print(f"Error saving new keywords to CSV: {e}")

        except Exception as e:
            print(f"Error processing sample {sample.source_id}: {e}")
            # Continue to next sample? Or break? Let's continue and log error.
            results.append({
                "source_id": sample.source_id,
                "error": str(e)
            })

    # Final Save (redundant but safe)
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
