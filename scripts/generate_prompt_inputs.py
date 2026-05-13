#!/usr/bin/env python3
"""
Generate model-ready prompt input text files for all 89 samples in gold_annotations.csv.

For samples WITH an English translation:
    INDEX_W_EN_V1_CONTEXT, KEYWORDS_W_EN_CONTEXT, FIELDS_W_EN_CONTEXT

For samples WITHOUT an English translation:
    INDEX_V1_CONTEXT, KEYWORDS_CONTEXT, FIELDS_CONTEXT

Output: data/prompt_inputs/{PROMPT_TYPE}/{sanitized_ref_id}.txt
Run from project root: python scripts/generate_prompt_inputs.py
"""

import os
import re
import sys
import csv

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from prompts.all_vectors import (
    INDEX_V1_CONTEXT,
    INDEX_W_EN_V1_CONTEXT,
    KEYWORDS_CONTEXT,
    KEYWORDS_W_EN_CONTEXT,
    FIELDS_CONTEXT,
    FIELDS_W_EN_CONTEXT,
)

GOLD_CSV = os.path.join(PROJECT_ROOT, "results", "annotation_report", "gold_annotations.csv")
LUR_CSV = os.path.join(PROJECT_ROOT, "data", "LUR_annotations.csv")
KEYWORDS_CSV = os.path.join(PROJECT_ROOT, "data", "Keywords.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "prompt_inputs")


def load_keywords_hierarchy() -> str:
    df = pd.read_csv(KEYWORDS_CSV)
    categories = []
    children_map: dict = {}

    for _, row in df.iterrows():
        kid = int(row["Id"])
        name = str(row["Keyword"]).strip()
        level = int(row["Level"])
        parent_raw = row["Parent KW Id"]
        parent_id = int(parent_raw) if (pd.notna(parent_raw) and parent_raw != 0) else None

        if level == 0:
            categories.append((kid, name))
        elif level == 1 and parent_id is not None:
            children_map.setdefault(parent_id, []).append((kid, name))

    lines = []
    for cat_id, cat_name in categories:
        lines.append(f"Category: {cat_name} (id: {cat_id})")
        for child_id, child_name in children_map.get(cat_id, []):
            lines.append(f"  - {child_name} (id: {child_id})")
        lines.append("")

    return "\n".join(lines).rstrip()


def sanitize_filename(ref_id: str) -> str:
    return re.sub(r"[^\w\-]", "_", ref_id).strip("_")


def main():
    print("Loading keyword hierarchy...")
    keywords_hierarchy = load_keywords_hierarchy()

    print("Loading gold annotations...")
    gold = pd.read_csv(GOLD_CSV)

    print("Loading context text from LUR annotations...")
    lur = pd.read_csv(LUR_CSV)
    context_map = dict(zip(lur["Reference"], lur["context text"]))

    prompt_types = [
        "INDEX_W_EN_V1_CONTEXT",
        "KEYWORDS_W_EN_CONTEXT",
        "FIELDS_W_EN_CONTEXT",
        "INDEX_V1_CONTEXT",
        "KEYWORDS_CONTEXT",
        "FIELDS_CONTEXT",
    ]
    for pt in prompt_types:
        os.makedirs(os.path.join(OUTPUT_DIR, pt), exist_ok=True)

    en_count = 0
    no_en_count = 0

    for _, row in gold.iterrows():
        ref_id = str(row["ref_id"])
        text = str(row["text"])
        language = str(row["language"])
        english = str(row["english"]) if pd.notna(row["english"]) else ""
        has_english = bool(row["has_english_translation"])

        ctx_raw = context_map.get(ref_id, "")
        broader_context = str(ctx_raw) if pd.notna(ctx_raw) else ""

        safe_name = sanitize_filename(ref_id)

        if has_english:
            en_count += 1
            prompts = {
                "INDEX_W_EN_V1_CONTEXT": INDEX_W_EN_V1_CONTEXT.format(
                    source_name=ref_id,
                    language=language,
                    text=text,
                    translation=english,
                    broader_context=broader_context,
                ),
                "KEYWORDS_W_EN_CONTEXT": KEYWORDS_W_EN_CONTEXT.format(
                    hierarchy=keywords_hierarchy,
                    source_name=ref_id,
                    language=language,
                    text=text,
                    translation=english,
                    broader_context=broader_context,
                ),
                "FIELDS_W_EN_CONTEXT": FIELDS_W_EN_CONTEXT.format(
                    source_name=ref_id,
                    language=language,
                    text=text,
                    translation=english,
                    broader_context=broader_context,
                ),
            }
        else:
            no_en_count += 1
            prompts = {
                "INDEX_V1_CONTEXT": INDEX_V1_CONTEXT.format(
                    source_name=ref_id,
                    language=language,
                    text=text,
                    broader_context=broader_context,
                ),
                "KEYWORDS_CONTEXT": KEYWORDS_CONTEXT.format(
                    hierarchy=keywords_hierarchy,
                    source_name=ref_id,
                    language=language,
                    text=text,
                    broader_context=broader_context,
                ),
                "FIELDS_CONTEXT": FIELDS_CONTEXT.format(
                    source_name=ref_id,
                    language=language,
                    text=text,
                    broader_context=broader_context,
                ),
            }

        for prompt_type, prompt_text in prompts.items():
            out_path = os.path.join(OUTPUT_DIR, prompt_type, f"{safe_name}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(prompt_text)

    total_files = (en_count + no_en_count) * 3
    print(f"\nDone!")
    print(f"  Samples with English:    {en_count:3d}  →  {en_count * 3} files (INDEX/KEYWORDS/FIELDS _W_EN_CONTEXT)")
    print(f"  Samples without English: {no_en_count:3d}  →  {no_en_count * 3} files (INDEX/KEYWORDS/FIELDS _CONTEXT)")
    print(f"  Total files created:     {total_files}")
    print(f"  Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
