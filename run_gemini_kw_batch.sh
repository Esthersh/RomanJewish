#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run batch classification
# Model: Interpreted "GPT4.1" as "gpt-4-turbo" (or "gpt-4o" is also an option).
# Temperature: 0
# Top P: Default (omitted flag)
# API Key: Provided by user

export PYTHONPATH=$PYTHONPATH:.

python3 src/batch_classify.py \
  --provider gemini \
  --api_key "AIzaSyDWMa8WRtJ5N1r7C8TTvr0e5y8wRVXCRq8" \
  --model "gemini-3-flash-preview" \
  --thinking_level "HIGH" \
  --prompt_file "/home/esther/PycharmProjects/RomanJewish/prompts/default.py" \
  --output_file "/home/esther/PycharmProjects/RomanJewish/results/mkw_gemini_3fp.json" \
  --corpus_csv "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv" \
  --keywords_csv "/home/esther/PycharmProjects/RomanJewish/data/Keywords.csv" \
  --prompt_k "MATCH_KEYWORDS" \
  --debug \
  --limit 5 && \
echo "Batch classification complete. Results saved to batch_results_gemini.json."
