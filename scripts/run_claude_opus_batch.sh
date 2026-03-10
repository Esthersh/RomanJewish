#!/bin/bash

# Activate virtual environment
source venv/bin/activate

# Anthropic API Key - PLEASE SET THIS
# export ANTHROPIC_API_KEY="your-key-here"

export PYTHONPATH=$PYTHONPATH:.

python3 src/batch_classify.py \
  --provider anthropic \
  --api_key "$ANTHROPIC_API_KEY" \
  --model "claude-opus-4-6" \
  --temperature 0. \
  --top_p 1.  \
  --prompt_file "/home/esther/PycharmProjects/RomanJewish/prompts/default.py" \
  --output_file "/home/esther/PycharmProjects/RomanJewish/results/mkw_claude_opus.json" \
  --corpus_csv "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv" \
  --keywords_csv "/home/esther/PycharmProjects/RomanJewish/data/Keywords.csv" \
  --prompt_k "MATCH_KEYWORDS" \
  --debug \
  --limit 10

echo "Batch classification complete. Results saved to results/mkw_claude_opus.json."
