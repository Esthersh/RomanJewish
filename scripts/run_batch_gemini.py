#!/usr/bin/env python3
"""
Run a set of prompt files through Gemini and save results to JSON.

Saves after each sample so progress is never lost — safe to re-run if interrupted.

Usage:
    python scripts/run_batch_gemini.py --prompt-type KEYWORDS_0_1_JTWC
    python scripts/run_batch_gemini.py --prompt-type TOPICS_0_1_PI --model gemini-2.5-flash
    python scripts/run_batch_gemini.py --prompt-type INDEX_0_1_JTWC --output results/my_run.json

Available prompt types:
    KEYWORDS_0_1_JTWC   KEYWORDS_0_1_PI
    TOPICS_0_1_JTWC     TOPICS_0_1_PI
    INDEX_0_1_JTWC      INDEX_0_1
"""

import argparse
import json
import os
import sys
import time
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
SECRETS_FILE = PROJECT_ROOT / "src" / ".streamlit" / "secrets.toml"

PROMPT_TYPES = [
    "KEYWORDS_0_1_JTWC",
    "KEYWORDS_0_1_PI",
    "TOPICS_0_1_JTWC",
    "TOPICS_0_1_PI",
    "INDEX_0_1",
]


def load_api_key() -> str:
    if not SECRETS_FILE.exists():
        return ""
    with open(SECRETS_FILE, "rb") as f:
        secrets = tomllib.load(f)
    return secrets.get("api_keys", {}).get("gemini", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompt-type", required=True, choices=PROMPT_TYPES,
        help="Which prompt folder to run (e.g. KEYWORDS_0_1_JTWC)",
    )
    parser.add_argument("--model",  default="gemini-2.5-pro")
    parser.add_argument("--output", default=None,
                        help="Output JSON path (default: results/gemini_{prompt_type}.json)")
    parser.add_argument("--delay",  type=float, default=1.0,
                        help="Seconds to wait between requests (default: 1)")
    args = parser.parse_args()

    prompt_dir  = PROJECT_ROOT / "data" / "prompt_inputs" / args.prompt_type
    default_out = PROJECT_ROOT / "results" / f"gemini_{args.prompt_type}.json"
    out_path    = Path(args.output) if args.output else default_out

    api_key = os.environ.get("GOOGLE_API_KEY") or load_api_key()
    if not api_key:
        sys.exit("Error: no Gemini API key found in secrets.toml or GOOGLE_API_KEY.")

    if not prompt_dir.exists():
        sys.exit(f"Error: prompt directory not found: {prompt_dir}\nRun generate_prompt_inputs.py first.")

    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Error: run  pip install google-genai")

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(temperature=0.0, top_p=1.0)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results so we can resume
    results: list[dict] = []
    if out_path.exists():
        with open(out_path) as f:
            results = json.load(f)
    done = {r["ref_id"] for r in results if r["response"] is not None}
    results = [r for r in results if r["response"] is not None]

    prompt_files = sorted(prompt_dir.glob("*.txt"))
    remaining = [p for p in prompt_files if p.stem not in done]

    print(f"Prompt type: {args.prompt_type}")
    print(f"Model:       {args.model}")
    print(f"Output:      {out_path}")
    print(f"Total:     {len(prompt_files)}  |  Done: {len(done)}  |  Remaining: {len(remaining)}")
    print()

    for i, prompt_file in enumerate(remaining, 1):
        ref_id = prompt_file.stem
        print(f"[{i}/{len(remaining)}] {ref_id} ... ", end="", flush=True)

        prompt = prompt_file.read_text(encoding="utf-8")

        raw = None
        for attempt in range(1, 6):
            try:
                response = client.models.generate_content(
                    model=args.model,
                    contents=prompt,
                    config=config,
                )
                raw = response.text
                print("OK")
                break
            except Exception as e:
                msg = str(e)
                if "503" in msg or "UNAVAILABLE" in msg:
                    wait = 10 * attempt
                    print(f"503 (attempt {attempt}/5), retrying in {wait}s ... ", end="", flush=True)
                    time.sleep(wait)
                else:
                    print(f"ERROR: {e}")
                    break
        else:
            print("FAILED after 5 attempts")

        results.append({"ref_id": ref_id, "response": raw})

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        if i < len(remaining):
            time.sleep(args.delay)

    print(f"\nDone. {len(results)} results saved to {out_path}")


if __name__ == "__main__":
    main()
