#!/bin/bash

# Activate virtual environment
source venv/bin/activate

export PYTHONPATH=$PYTHONPATH:.

# Qwen (DashScope) API Key
# export QWEN_API_KEY="your-key-here"

python3 src/batch_classify.py \
  --provider qwen \
  --api_key "$QWEN_API_KEY" \
  --model "qwen3-max-2026-01-23" \
  --prompt_file "/home/esther/PycharmProjects/RomanJewish/prompts/default.py" \
  --output_file "/home/esther/PycharmProjects/RomanJewish/results/mkw_qwen3_max.json" \
  --corpus_csv "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv" \
  --keywords_csv "/home/esther/PycharmProjects/RomanJewish/data/Keywords.csv" \
  --prompt_k "MATCH_KEYWORDS" \
  --limit 10 \
  --debug

echo "Batch classification complete. Results saved to results/mkw_qwen3_max.json."
