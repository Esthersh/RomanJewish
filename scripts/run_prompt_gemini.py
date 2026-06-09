#!/usr/bin/env python3
"""
Read a prompt text file and print the response from Gemini.

Usage:
    python scripts/run_prompt_gemini.py <prompt_file> [--api_key KEY] [--model MODEL]

API key precedence: --api_key flag > GOOGLE_API_KEY env var.

Example:
    python scripts/run_prompt_gemini.py data/prompt_inputs/INDEX_W_EN_V1_CONTEXT/Mishnah_Bava_batra_8_1.txt --api_key YOUR_KEY
"""

import argparse
import os
import sys
import tomllib
from pathlib import Path

SECRETS_FILE = Path(__file__).parent.parent / "src" / ".streamlit" / "secrets.toml"


def load_api_key_from_secrets() -> str:
    if not SECRETS_FILE.exists():
        return ""
    with open(SECRETS_FILE, "rb") as f:
        secrets = tomllib.load(f)
    return secrets.get("api_keys", {}).get("gemini", "")


def main():
    parser = argparse.ArgumentParser(description="Run a prompt file through Gemini and print the output")
    parser.add_argument("prompt_file", help="Path to the prompt text file")
    parser.add_argument("--api_key", default=None, help="Google API key (overrides secrets.toml and GOOGLE_API_KEY)")
    parser.add_argument("--model", default="gemini-2.5-pro", help="Gemini model name")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY") or load_api_key_from_secrets()
    if not api_key:
        print("Error: no API key found. Add it to src/.streamlit/secrets.toml, set GOOGLE_API_KEY, or pass --api_key.", file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(args.prompt_file):
        print(f"Error: file not found: {args.prompt_file}", file=sys.stderr)
        sys.exit(1)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        print("Error: google-genai package not installed. Run: pip install google-genai", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.0,
        top_p=1.0,
    )

    print(f"Model:  {args.model}", file=sys.stderr)
    print(f"File:   {args.prompt_file}", file=sys.stderr)
    print(file=sys.stderr)

    response = client.models.generate_content(
        model=args.model,
        contents=prompt,
        config=config,
    )

    print(response.text)


if __name__ == "__main__":
    main()
