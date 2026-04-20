import argparse
import json
import os
import sys
from argparse import Namespace
from tqdm import tqdm
from data_loader import DataLoader
from classifier import Classifier


def main():
    """
    :return:
    """
    args = parse_run_args()

    # Determine which prompt type we're running
    prompt_type = args.prompt_k
    needs_keywords = not (prompt_type.startswith("FIELDS") or prompt_type.startswith("INDEX"))

    # Load Data
    print("Loading data...")
    loader = DataLoader()
    keywords = []
    if needs_keywords:
        if not args.keywords_csv:
            print("Error: --keywords_csv is required for KEYWORDS/MATCH_KEYWORDS prompts.")
            sys.exit(1)
        keywords = loader.load_keywords(args.keywords_csv)
    corpus = loader.load_corpus(args.corpus_csv, include_non_english=args.include_non_english,
                                # include_unannotated=args.include_unannotated,
                                analyzed_only=args.analyzed_only,
                                context_filter=args.context_filter)

    if args.limit:
        corpus = corpus[:args.limit]
        print(f"Limiting to {args.limit} samples.")

    if needs_keywords:
        print(f"Loaded {len(keywords)} keywords and {len(corpus)} samples.")
    else:
        print(f"Loaded {len(corpus)} samples.")

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
            thinking_level=args.thinking_level,
            topics_csv=args.topics_csv,
            keywords_csv=args.keywords_csv
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
                    processed_ids = {str(item.get("ref_id")) for item in results if "ref_id" in item}
                    print(
                        f"Loaded {len(results)} existing results. Skipping {len(results)} already processed samples.")
        except json.JSONDecodeError:
            print(f"Warning: Could not decode {args.output_file}. Starting fresh.")
        except Exception as e:
            print(f"Warning: Error reading {args.output_file}: {e}. Starting fresh.")

    print("Starting classification...")
    for sample in tqdm(corpus):
        if str(sample.ref_id) in processed_ids:
            continue

        try:
            metadata = {
                "source_name": sample.source_name,
                "group": sample.group,
                "ref_id": sample.ref_id,
                "language": sample.language,
                "translation": sample.original_row.get('translation', ''),
                "broader_context": sample.context_text
            }
            matched_ids, suggested_kws, full_res = classifier.classify(sample.text, metadata)

            if not full_res or not full_res.strip():
                print(f"Skipping sample {sample.ref_id} due to empty LLM response.")
                continue

            result_entry = {
                "ref_id": sample.ref_id,
                "source_id": sample.source_id,
                "group": sample.group,
                "name": sample.source_name,
                "text": sample.text,
                "original_row": sample.original_row,
                "original_res": full_res,
            }

            if prompt_type.startswith("INDEX"):
                result_entry["index_terms"] = matched_ids  # list of strings for INDEX
            elif prompt_type.startswith("FIELDS"):
                result_entry["matched_field_ids"] = matched_ids
            else:
                # KEYWORDS / MATCH_KEYWORDS
                kw_map = {str(k.id): k.name for k in keywords}
                matched_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in matched_ids]
                result_entry["matched_ids"] = matched_ids
                result_entry["matched_keywords"] = matched_names
                result_entry["suggested_kws"] = suggested_kws

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


def parse_run_args() -> Namespace:
    parser = argparse.ArgumentParser(description="Run RomanJewish Classification Batch")
    parser.add_argument("--provider", type=str, required=True,
                        choices=['openai', 'google', 'gemini', 'anthropic', 'claude',
                                 'together', 'qwen', 'dashscope'],
                        help="LLM provider name")
    parser.add_argument("--api_key", required=True, help="API Key for the provider")
    parser.add_argument("--prompt_file",
                        default="/home/esther/PycharmProjects/RomanJewish/prompts/default.py",
                        help="Path to prompt file")
    parser.add_argument("--prompt_k", default="CLASSIFICATION_PROMPT",
                        help="Name of the prompt variable to use")
    parser.add_argument("--keywords_csv", default=None,
                        help="Path to keywords CSV (required for KEYWORDS/MATCH_KEYWORDS prompts)")
    parser.add_argument("--topics_csv", default=None,
                        help="Path to topics CSV (required for FIELDS prompts)")
    parser.add_argument("--corpus_csv", default="LUR sample corpus.csv",
                        help="Path to corpus CSV")
    parser.add_argument("--output_file", default="batch_results.json",
                        help="Output JSON file for results")
    parser.add_argument("--limit", type=int, help="Limit number of samples for testing")
    # Model config args
    parser.add_argument("--model", type=str, help="Model signature")
    parser.add_argument("--temperature", type=float, help="Temperature for generation")
    parser.add_argument("--top_p", type=float, help="Top-P for generation")
    parser.add_argument("--thinking_level", type=str, help="thinking_level for generation")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")
    # only relevant if we decide the keyword could grow on the fly
    parser.add_argument("--expand_kwords", action="store_true")
    parser.add_argument("--include_non_english", action="store_true", help="Include samples without English translations")
    # parser.add_argument("--include_unannotated", action="store_true", help="Include samples without keyword annotations")
    parser.add_argument("--analyzed_only", action="store_true", help="Only include samples where Analyzed [y/n] == y")
    parser.add_argument("--context_filter", type=str, default='any', choices=['with', 'without', 'any'],
                        help="Filter samples by context: 'with' = only samples with context, 'without' = only without, 'any' = no filter")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main()
