#!/usr/bin/env python3
"""
Tag sources with keywords *sequentially*, growing the keyword vocabulary as we go.

Unlike run_batch_gemini.py (stateless: every prompt is pre-rendered up front with a
fixed keyword list), this renders the keyword list fresh for each source from a
vocabulary that grows. Keywords the model suggests are folded back into the vocabulary
with new ids (>= 10000) so that *later* sources can reuse them instead of re-coining
near-identical variants — which is what was causing the explosion of distinct keywords.

For each source, every keyword the model returns is resolved *by name* against the
vocabulary; the id always comes from us, never from the model (model ids are unreliable
-- they come back as strings, or as -1 for every suggestion). A name already present
(seed or previously added) reuses its id; a genuinely new name mints the next id >= 10000.

Outputs are namespaced by prompt version (<ver> = 0_2 by default), saved after every
source -- safe to resume by re-running the same command (only failed/unseen sources rerun):
    results/keywords_sequential_<ver>/gemini_KEYWORDS_<ver>_SEQ.json   per-source results;
                                                `response` is the normalized JSON (real ids).
    results/keywords_sequential_<ver>/keyword_vocab_snapshot.json      resume state.
    data/Keywords_augmented_<ver>.csv                                  seed + added keywords
                                                (curated Keywords.csv is never modified).

Transient overload (503/429/5xx) uses fail-fast retries (--max-attempts, default 3), then
defers the source to the next resume pass rather than burning long per-source backoffs.

Usage:
    python scripts/run_keywords_sequential.py                      # 0.2, full samples2update, pro
    python scripts/run_keywords_sequential.py --prompt-version 0.1 --max-jtwc 30
    python scripts/run_keywords_sequential.py --dry-run            # render first prompt, no API call
"""

import argparse
import json
import os
import re
import sys
import time
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from prompts.all_vectors import (
    KEYWORDS_0_1_JTWC,
    KEYWORDS_0_1_PI,
    KEYWORDS_0_2_JTWC,
    KEYWORDS_0_2_PI,
)
from scripts.generate_prompt_inputs import load_rows
from src.keyword_manager import KeywordVocabulary

SECRETS_FILE = PROJECT_ROOT / "src" / ".streamlit" / "secrets.toml"
SEED_CSV = PROJECT_ROOT / "data" / "Keywords.csv"

# Prompt templates per version. 0.1 and 0.2 share the same {placeholders}, so the
# runner only needs to pick the pair; build_prompt is version-agnostic.
TEMPLATES = {
    "0.1": {"jtwc": KEYWORDS_0_1_JTWC, "pi": KEYWORDS_0_1_PI},
    "0.2": {"jtwc": KEYWORDS_0_2_JTWC, "pi": KEYWORDS_0_2_PI},
}


def load_api_key() -> str:
    if not SECRETS_FILE.exists():
        return ""
    with open(SECRETS_FILE, "rb") as f:
        secrets = tomllib.load(f)
    return secrets.get("api_keys", {}).get("gemini", "")


# ----------------------------------------------------------------- pure helpers
def select_sources(rows: list[dict], max_jtwc=None, max_pi=None, limit=None) -> list[dict]:
    """Pick a subset by variant, preserving corpus order.

    JTWC = sources with broader context; P&I = sources without. Each cap keeps the
    first N of that variant (in corpus order); `limit` then caps the merged result.
    The vocabulary still grows in corpus order across the interleaved subset.
    """
    jtwc = [(i, r) for i, r in enumerate(rows) if r["has_context"]]
    pi = [(i, r) for i, r in enumerate(rows) if not r["has_context"]]
    if max_jtwc is not None:
        jtwc = jtwc[:max_jtwc]
    if max_pi is not None:
        pi = pi[:max_pi]
    selected = [r for _, r in sorted(jtwc + pi, key=lambda x: x[0])]
    return selected[:limit] if limit else selected


def build_prompt(row: dict, vocab: KeywordVocabulary, templates: dict) -> str:
    """Render the keyword prompt for a source using the current vocabulary.

    JTWC (with broader context) vs P&I is chosen by `has_context`, mirroring
    generate_prompt_inputs.py exactly, plus the dynamic `keyword_list`. `templates`
    is one entry of TEMPLATES, e.g. TEMPLATES["0.2"].
    """
    keyword_list = vocab.render_list()
    if row["has_context"]:
        return templates["jtwc"].format(
            source_name=row["ref_id"],
            language=row["language"],
            text=row["text"],
            broader_context=row["broader_context"],
            translation_note=" and its English translation" if row["has_english"] else "",
            translation_section=f"\nEnglish Translation:\n{row['translation']}" if row["has_english"] else "",
            keyword_list=keyword_list,
        )
    return templates["pi"].format(
        reference_name=row["ref_id"],
        language=row["language"],
        text=row["text"],
        translation=row["translation"],
        keyword_list=keyword_list,
    )


def parse_keywords(raw: object) -> list[dict]:
    """Extract a list of keyword objects from a (possibly fenced) model response."""
    if not raw:
        return []
    text = str(raw)
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidate = match.group(1).strip() if match else text.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return []
    if isinstance(parsed, dict):
        if isinstance(parsed.get("keywords"), list):
            parsed = parsed["keywords"]
        elif "keyword" in parsed:
            parsed = [parsed]
        else:
            return []
    if not isinstance(parsed, list):
        return []
    return [obj for obj in parsed if isinstance(obj, dict) and obj.get("keyword")]


def normalize(parsed: list[dict], vocab: KeywordVocabulary) -> list[dict]:
    """Resolve each returned keyword against the vocabulary, growing it as needed.

    `suggested=True` marks a keyword *first introduced by this source*; a keyword an
    earlier source introduced is reused (suggested=False) but keeps its id >= 10000.
    """
    out = []
    for obj in parsed:
        kw, is_new = vocab.resolve(obj.get("keyword"), obj.get("category"), obj.get("category_id"))
        out.append({
            "category": kw.category_name,
            "keyword": kw.name,
            "suggested": is_new,
            "category_id": kw.category_id,
            "keyword_id": kw.id,
        })
    return out


# ----------------------------------------------------------------------- driver
def _is_transient(msg: str) -> bool:
    return any(t in msg for t in ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "500", "INTERNAL"))


def call_gemini(client, types, model: str, prompt: str, max_attempts: int = 3, base_wait: int = 5) -> object:
    """Call Gemini with *fail-fast* retries on transient overload (503/429/5xx).

    A few quick tries (base_wait*attempt seconds), then give up and return None so the
    source is recorded as a deferred failure and retried on the next resume pass. This
    is far more throughput-efficient under sustained overload than long per-source
    backoffs — a 401-source run captures the available successes quickly and you mop up
    the rest in later passes, rather than burning minutes of backoff on each failure.
    """
    config = types.GenerateContentConfig(temperature=0.0, top_p=1.0)
    for attempt in range(1, max_attempts + 1):
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config).text
        except Exception as e:  # noqa: BLE001 -- transient overload is retried; everything else defers
            msg = str(e)
            if _is_transient(msg) and attempt < max_attempts:
                wait = base_wait * attempt
                print(f"transient (attempt {attempt}/{max_attempts}), retry in {wait}s ... ", end="", flush=True)
                time.sleep(wait)
            else:
                print("deferred" if _is_transient(msg) else f"ERROR: {e}", end="", flush=True)
                return None
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["gold", "all", "samples2update"], default="samples2update")
    parser.add_argument("--prompt-version", choices=["0.1", "0.2"], default="0.2",
                        help="Which KEYWORDS_<ver> templates to use (default: 0.2)")
    parser.add_argument("--model", default="gemini-2.5-pro")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds between requests (default: 1)")
    parser.add_argument("--max-attempts", type=int, default=3, help="Retries per source on transient errors (default: 3)")
    parser.add_argument("--retry-wait", type=int, default=5, help="Base backoff seconds, grows per attempt (default: 5)")
    parser.add_argument("--output-dir", default=None,
                        help="Default: results/keywords_sequential_<ver>/")
    parser.add_argument("--augmented-out", default=None,
                        help="Default: data/Keywords_augmented_<ver>.csv")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N sources overall (testing)")
    parser.add_argument("--max-jtwc", type=int, default=None, help="Cap JTWC (with-context) sources to the first N")
    parser.add_argument("--max-pi", type=int, default=None, help="Cap P&I (no-context) sources to the first N")
    parser.add_argument("--dry-run", action="store_true", help="Render the first prompt and exit (no API call)")
    args = parser.parse_args()

    templates = TEMPLATES[args.prompt_version]
    vtag = args.prompt_version.replace(".", "_")  # 0.2 -> 0_2; namespaces all outputs by version
    out_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / "results" / f"keywords_sequential_{vtag}"
    augmented_out = args.augmented_out or str(PROJECT_ROOT / "data" / f"Keywords_augmented_{vtag}.csv")
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / f"gemini_KEYWORDS_{vtag}_SEQ.json"
    snapshot_path = out_dir / "keyword_vocab_snapshot.json"

    print(f"Dataset: {args.dataset}  |  Prompt version: {args.prompt_version}  |  Output: {out_dir}")
    rows = load_rows(args.dataset)
    rows = select_sources(rows, max_jtwc=args.max_jtwc, max_pi=args.max_pi, limit=args.limit)
    n_jtwc = sum(1 for r in rows if r["has_context"])
    print(f"Sources: {len(rows)}  (JTWC={n_jtwc}, P&I={len(rows) - n_jtwc})")

    # Resume vocabulary from snapshot if present, else seed fresh from the curated CSV.
    if snapshot_path.exists():
        vocab = KeywordVocabulary.load_snapshot(snapshot_path)
        print(f"Resumed vocabulary from snapshot: {len(vocab.keywords)} keywords, next id {vocab._next_id}")
    else:
        vocab = KeywordVocabulary.from_csv(SEED_CSV)
        print(f"Seeded vocabulary from {SEED_CSV.name}: {len(vocab.keywords)} keywords")

    if args.dry_run:
        print("\n--- DRY RUN: rendered prompt for first source ---\n")
        print(build_prompt(rows[0], vocab, templates))
        return

    results: list[dict] = []
    if results_path.exists():
        results = [r for r in json.loads(results_path.read_text(encoding="utf-8")) if r.get("response") is not None]
    done = {r["ref_id"] for r in results}

    api_key = os.environ.get("GOOGLE_API_KEY") or load_api_key()
    if not api_key:
        sys.exit("Error: no Gemini API key found in secrets.toml or GOOGLE_API_KEY.")
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        sys.exit("Error: run  pip install google-genai")
    client = genai.Client(api_key=api_key)

    remaining = [r for r in rows if r["ref_id"] not in done]
    print(f"Model: {args.model}  |  Done: {len(done)}  |  Remaining: {len(remaining)}\n")

    for i, row in enumerate(remaining, 1):
        ref_id = row["ref_id"]
        print(f"[{i}/{len(remaining)}] {ref_id} ... ", end="", flush=True)

        raw = call_gemini(client, types, args.model, build_prompt(row, vocab, templates),
                          max_attempts=args.max_attempts, base_wait=args.retry_wait)
        if raw is None:
            print("  [deferred to resume]")
            results.append({"ref_id": ref_id, "response": None, "response_raw": None})
            results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        parsed = parse_keywords(raw)
        # Save vocabulary state BEFORE results so a crash never leaves results that
        # reference ids missing from the snapshot (which would drift later ids).
        normalized = normalize(parsed, vocab)
        vocab.save_snapshot(snapshot_path)
        vocab.save_augmented_csv(augmented_out)

        new_here = sum(1 for k in normalized if k["suggested"])
        results.append({
            "ref_id": ref_id,
            "response": json.dumps(normalized, ensure_ascii=False),
            "response_raw": raw,
        })
        results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"OK ({len(normalized)} kw, {new_here} new; vocab now {len(vocab.keywords)})")

        if i < len(remaining):
            time.sleep(args.delay)

    ok = sum(1 for r in results if r.get("response") is not None)
    print(f"\nDone. {ok}/{len(rows)} sources succeeded -> {results_path}")
    print(f"Vocabulary: {len(vocab.keywords)} keywords ({len(vocab.added())} added) -> {augmented_out}")
    if ok < len(rows):
        print(f"  {len(rows) - ok} deferred/failed — re-run the same command to resume just those.")


if __name__ == "__main__":
    main()
