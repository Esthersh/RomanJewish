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
  --provider openai \
  --api_key "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" \
  --model "gpt-5.1-2025-11-13" \
  --thinking_level "high" \
  --prompt_file "/home/esther/PycharmProjects/RomanJewish/prompts/default.py" \
  --output_file "/home/esther/PycharmProjects/RomanJewish/results/mkw_gpt5_1.json" \
  --corpus_csv "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv" \
  --keywords_csv "/home/esther/PycharmProjects/RomanJewish/data/Keywords.csv" \
  --prompt_k "MATCH_KEYWORDS" \
  --debug \
  --limit 5 && \
echo "Batch classification complete. Results saved to batch_results_gpt4.json."
