#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Run batch classification
# Temperature: 0
# Top P: Default (omitted flag)
# API Key: Provided by user

export PYTHONPATH=$PYTHONPATH:.

python3 src/batch_classify.py \
  --provider gemini \
  --api_key "AIzaSyDWMa8WRtJ5N1r7C8TTvr0e5y8wRVXCRq8" \
  --prompt_k "MINIMAL_CLASSIFY_5_SHOT" \
  --model "gemini-3-flash-preview" \
  --temperature 0 \
  --top_p 1 \
  --thinking_level "HIGH" \
  --corpus_csv "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv" \
  --keywords_csv "/home/esther/antigravity/RomanJewish/Keywords_05022026.csv" \
  --output_file "batch_results_gemini3.json" && \
  echo "Batch classification complete. Results saved to batch_results_gemini3.json."
