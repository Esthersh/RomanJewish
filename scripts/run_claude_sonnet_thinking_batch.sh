#!/bin/bash

# Activate virtual environment
source venv/bin/activate

export PYTHONPATH=$PYTHONPATH:.

# Anthropic API Key
# API_KEY="your-key-here"

python3 src/batch_classify.py \
  --provider anthropic \
  --api_key "$ANTHROPIC_API_KEY" \
  --model "claude-3-7-sonnet-20250219" \
  --thinking_level high \
  --prompt_file "/home/esther/PycharmProjects/RomanJewish/prompts/default.py" \
  --output_file "/home/esther/PycharmProjects/RomanJewish/results/mkw_claude_sonnet_thinking.json" \
  --corpus_csv "/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv" \
  --keywords_csv "/home/esther/PycharmProjects/RomanJewish/data/Keywords.csv" \
  --prompt_k "MATCH_KEYWORDS" \
  --limit 10 \
  --debug

echo "Batch classification complete. Results saved to results/mkw_claude_sonnet_thinking.json."
