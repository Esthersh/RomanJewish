import argparse
import json
import os
import sys
from tqdm import tqdm
from src.data_loader import DataLoader
from src.classifier import Classifier
from src.keyword_manager import KeywordManager

def main():
    parser = argparse.ArgumentParser(description="Run RomanJewish Classification Batch")
    parser.add_argument("--provider", required=True, choices=["gemini", "openai", "qwen"], help="LLM Provider")
    parser.add_argument("--api_key", required=True, help="API Key for the provider")
    parser.add_argument("--prompt_file", default="prompts/default.py", help="Path to prompt file")
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
    
    print("Starting classification...")
    for sample in tqdm(corpus):
        try:
            metadata = {
                "source_name": sample.source_name,
                "group": sample.group,
                "name": sample.name
            }
            matched_ids, suggested_kws = classifier.classify(sample.text_en, keywords, metadata)
            
            # Resolve IDs to Names
            # specific helper to find name by id
            kw_map = {str(k.id): k.name for k in keywords}
            matched_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in matched_ids]

            result_entry = {
                "source_id": sample.source_id,
                "group": sample.group,
                "name": sample.name,
                "text_en": sample.text_en,
                "original_row": sample.original_row, # Keep original metadata
                "matched_ids": matched_ids,
                "matched_keywords": matched_names,
                "suggested_kws": suggested_kws
            }
            results.append(result_entry)
            
            # Save results iteratively
            with open(args.output_file, 'w') as f:
                json.dump(results, f, indent=2)
                
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
