#!/usr/bin/env python3
"""
Generate model-ready prompt input text files from either the 89-sample gold set or the
full LUR table (~400 sources).

Prompt selection is based on the "context level" column in LUR_annotations.csv:
  - Sources WITH a context level  → JTWC prompts (includes broader_context)
  - Sources WITHOUT a context level → P&I prompts (no context)

Within each branch, English translation is included when available.

Usage:
    python scripts/generate_prompt_inputs.py            # gold set (89 samples, default)
    python scripts/generate_prompt_inputs.py --dataset all   # full LUR table

Output: data/prompt_inputs/{PROMPT_TYPE}/{sanitized_ref_id}.txt
"""

import argparse
import os
import re
import sys

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# NOTE: KEYWORDS prompts are no longer pre-rendered here. Their keyword list grows
# as sources are processed, so they are rendered on the fly by
# scripts/run_keywords_sequential.py. This script handles TOPICS + INDEX only.
from prompts.all_vectors import (
    INDEX_0_1,
    TOPICS_0_1_JTWC,
    TOPICS_0_1_PI,
)
from scripts.sheets_loader import ID_COLUMN, load_samples_tab

GOLD_CSV = os.path.join(PROJECT_ROOT, "results", "annotation_report", "gold_annotations.csv")
LUR_CSV  = os.path.join(PROJECT_ROOT, "data", "LUR_annotations.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "prompt_inputs")


def sanitize_filename(ref_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", ref_id).strip("_")


def load_rows(dataset: str) -> list[dict]:
    """Return a list of dicts with keys: ref_id, text, language, translation, has_english, has_context, broader_context."""
    if dataset == "samples2update":
        # Pulled from the Samples2Update tab in Drive; no LUR_CSV needed.
        pass
    else:
        lur = pd.read_csv(LUR_CSV)

    if dataset == "gold":
        gold = pd.read_csv(GOLD_CSV)
        context_text_map = dict(zip(lur["Reference"], lur["context text"]))
        context_level_map = {
            ref: (pd.notna(level) and str(level).strip() not in ("", "nan"))
            for ref, level in zip(lur["Reference"], lur["context level"])
        }
        rows = []
        for _, row in gold.iterrows():
            ref_id = str(row["ref_id"])
            ctx_raw = context_text_map.get(ref_id, "")
            rows.append({
                "ref_id": ref_id,
                "text": str(row["text"]),
                "language": str(row["language"]),
                "translation": str(row["english"]) if pd.notna(row["english"]) else "",
                "has_english": bool(row["has_english_translation"]),
                "has_context": context_level_map.get(ref_id, False),
                "broader_context": str(ctx_raw) if pd.notna(ctx_raw) else "",
            })
        return rows

    if dataset == "samples2update":
        df = load_samples_tab()
        rows = []
        for _, row in df.iterrows():
            raw_id = row.get(ID_COLUMN)
            if pd.isna(raw_id) or str(raw_id).strip() in ("", "nan"):
                continue
            ref_id = str(raw_id).strip()
            translation = str(row["English"]) if pd.notna(row.get("English")) else ""
            ctx_raw = row.get("context text")
            broader_context = str(ctx_raw) if pd.notna(ctx_raw) else ""
            has_context = pd.notna(row.get("context level")) and str(row.get("context level")).strip() not in ("", "nan")
            rows.append({
                "ref_id": ref_id,
                "text": str(row["Text"]) if pd.notna(row.get("Text")) else "",
                "language": str(row["Language"]) if pd.notna(row.get("Language")) else "",
                "translation": translation,
                "has_english": bool(translation),
                "has_context": has_context,
                "broader_context": broader_context,
            })
        return rows

    # dataset == "all": read directly from LUR
    rows = []
    for _, row in lur.iterrows():
        if pd.isna(row["Reference"]):
            if pd.isna(row["Calculated refCode"]):
                continue
            ref_id = str(row["Calculated refCode"])
        else:
            ref_id = str(row["Reference"])
        translation = str(row["English"]) if pd.notna(row.get("English", None)) else ""
        ctx_raw = row["context text"]
        broader_context = str(ctx_raw) if pd.notna(ctx_raw) else ""
        has_context = pd.notna(row["context level"]) and str(row["context level"]).strip() not in ("", "nan")
        rows.append({
            "ref_id": ref_id,
            "text": str(row["Text"]) if pd.notna(row["Text"]) else "",
            "language": str(row["Language"]) if pd.notna(row["Language"]) else "",
            "translation": translation,
            "has_english": bool(translation),
            "has_context": has_context,
            "broader_context": broader_context,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=["gold", "all", "samples2update"], default="gold",
        help="'gold' = 89-sample gold set (default); 'all' = full LUR table; "
             "'samples2update' = pull from the Samples2Update tab in Drive (uses 'Id' as ref_id)",
    )
    args = parser.parse_args()

    print(f"Dataset:  {args.dataset}")
    rows = load_rows(args.dataset)
    print(f"Sources:  {len(rows)}")

    prompt_types = [
        "TOPICS_0_1_JTWC",
        "TOPICS_0_1_PI",
        "INDEX_0_1",
    ]
    for pt in prompt_types:
        pt_dir = os.path.join(OUTPUT_DIR, pt)
        os.makedirs(pt_dir, exist_ok=True)
        for f in os.listdir(pt_dir):
            os.remove(os.path.join(pt_dir, f))

    jtwc_count = 0
    pi_count = 0

    for row in rows:
        ref_id       = row["ref_id"]
        text         = row["text"]
        language     = row["language"]
        translation  = row["translation"]
        has_english  = row["has_english"]
        has_context  = row["has_context"]
        broader_context = row["broader_context"]

        safe_name = sanitize_filename(ref_id)
        index_prompt = INDEX_0_1.format(
            source_name=ref_id,
            language=language,
            text=text,
            context_section=f"Broader Context:\n{broader_context}\n\n" if has_context else "",
        )

        if has_context:
            jtwc_count += 1
            prompts = {
                "TOPICS_0_1_JTWC": TOPICS_0_1_JTWC.format(
                    reference_name=ref_id,
                    language=language,
                    text=text,
                    broader_context=broader_context,
                    translation=translation,
                ),
                "INDEX_0_1": index_prompt,
            }
        else:
            pi_count += 1
            prompts = {
                "TOPICS_0_1_PI": TOPICS_0_1_PI.format(
                    reference_name=ref_id,
                    language=language,
                    text=text,
                    translation=translation,
                ),
                "INDEX_0_1": index_prompt,
            }

        for prompt_type, prompt_text in prompts.items():
            out_path = os.path.join(OUTPUT_DIR, prompt_type, f"{safe_name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)

    total = jtwc_count + pi_count
    files_per_source = len(prompt_types) - 1  # each source gets its TOPICS variant + INDEX
    print(f"\nDone!")
    print(f"  JTWC (with context):  {jtwc_count:3d}  →  {jtwc_count * files_per_source} files")
    print(f"  P&I  (no context):    {pi_count:3d}  →  {pi_count * files_per_source} files")
    print(f"  Total files created:  {total * files_per_source}")
    print(f"  (KEYWORDS handled separately by run_keywords_sequential.py)")
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
