import argparse
import json
import os
import sys
from tqdm import tqdm
from data_loader import DataLoader
from classifier import Classifier


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
    parser.add_argument("--prompt_file", default="/home/esther/PycharmProjects/RomanJewish/prompts/default.py",
                        help="Path to prompt file")
    parser.add_argument("--prompt_k", default="CLASSIFICATION_PROMPT", help="Name of the prompt variable to use")
    parser.add_argument("--keywords_csv", default="Keywords_05022026.csv", help="Path to keywords CSV")
    parser.add_argument("--corpus_csv", default="LUR sample corpus.csv", help="Path to corpus CSV")
    parser.add_argument("--output_file", default="batch_results.json", help="Output JSON file for results")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    # Model config args
    parser.add_argument("--model", type=str, help="Model signature/name (e.g. gpt-4-turbo)")
    parser.add_argument("--temperature", type=float, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, help="Top-P for generation")
    parser.add_argument("--thinking_level", type=str, help="thinking_level for generation")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    parser.add_argument("--expand_kwords", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    # Load Data
    print("Loading data...")
    loader = DataLoader()
    keywords = loader.load_keywords(args.keywords_csv)
    corpus = loader.load_corpus(args.corpus_csv)

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
            debug=args.debug,
            thinking_level=args.thinking_level
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
                    print(
                        f"Loaded {len(results)} existing results. Skipping {len(processed_ids)} already processed samples.")
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
                "ref_id": sample.ref_id,
                "language": sample.language,
                "translation": sample.original_row.get('translation', '')
            }
            matched_ids, suggested_kws, full_res = classifier.classify(sample.text, keywords, metadata)

            # Resolve IDs to Names
            # specific helper to find name by id
            kw_map = {str(k.id): k.name for k in keywords}
            matched_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in matched_ids]

            result_entry = {
                "ref_id": sample.ref_id,
                "source_id": sample.source_id,
                "group": sample.group,
                "name": sample.source_name,
                "text": sample.text,
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

        except Exception as e:
            print(f"Error processing sample {sample.ref_id}: {e}")
            # Continue to next sample? Or break? Let's continue and log error.
            # results.append({
            #     "source_id": sample.source_id,
            #     "error": str(e)
            # })
            continue

    # Final Save (redundant but safe)
    print(f"Saving results to {args.output_file}...")
    with open(args.output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
