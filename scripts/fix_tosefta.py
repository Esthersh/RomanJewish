import json
import os

from pytz import reference

TRIPLETS = [
    (
        "results/prioritized/merged_claude_opus4_6.json",
        "results/tosefta/merged_w_en_claude.json",
        "results/prioritized/tosefta/merged_claude_opus4_6.json",
    ),
    (
        "results/prioritized/merged_gemini_3_pro.json",
        "results/tosefta/merged_w_en_gemini.json",
        "results/prioritized/tosefta/merged_gemini_3_pro.json",
    ),
    (
        "results/prioritized/merged_qwen_3_5.json",
        "results/tosefta/merged_w_en_qwen.json",
        "results/prioritized/tosefta/merged_qwen_3_5.json",
    ),
]

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for prio_path, tosefta_path, out_path in TRIPLETS:
    prio_full = os.path.join(BASE, prio_path)
    tosefta_full = os.path.join(BASE, tosefta_path)
    out_full = os.path.join(BASE, out_path)

    with open(prio_full, encoding="utf-8") as f:
        prio = json.load(f)
    with open(tosefta_full, encoding="utf-8") as f:
        tosefta = json.load(f)

    def ref_key(r):
        row = r.get("original_row", {})
        return row.get("Reference") or row.get("Refference")

    tosefta_by_id = {ref_key(r): r for r in tosefta if ref_key(r)}

    def fix_name_group(rec, name_fallback, group_fallback):
        if not rec.get("name"):
            rec["name"] = name_fallback
        if not rec.get("group") or rec["group"] == "nan":
            row_group = rec.get("original_row", {}).get("Group")
            rec["group"] = str(row_group) if row_group and str(row_group) != "nan" else group_fallback

    replaced = 0
    matched_keys = set()
    result = []
    for record in prio:
        key = ref_key(record)
        if key and key in tosefta_by_id:
            rec = dict(tosefta_by_id[key])
            rec["origin_file"] = tosefta_path
            fix_name_group(rec, name_fallback=key, group_fallback=record.get("group", ""))
            result.append(rec)
            matched_keys.add(key)
            replaced += 1
        else:
            result.append(record)

    added = 0
    for record in tosefta:
        key = ref_key(record)
        if key and key not in matched_keys:
            rec = dict(record)
            rec["origin_file"] = tosefta_path
            fix_name_group(rec, name_fallback=key, group_fallback="")
            result.append(rec)
            added += 1

    # sort by /home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv order
    import pandas as pd
    df = pd.read_csv("/home/esther/PycharmProjects/RomanJewish/data/LUR_annotations.csv")
    # sort by "name" column, which contains the reference
    name_order = {str(name): idx for idx, name in enumerate(df["Reference"])}
    result.sort(key=lambda r: name_order.get(str(ref_key(r)), float('inf')))


    with open(out_full, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"{os.path.basename(out_path)}: {len(prio)} records, {replaced} replaced, {added} added from tosefta")