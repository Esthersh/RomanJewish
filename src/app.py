import os
import sys
import string

# Add the project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import streamlit as st
import pandas as pd
import json
import yaml
from datetime import date
import unicodedata
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from classifier import format_keywords
from streamlit_gsheets import GSheetsConnection

from data_loader import DataLoader
from keyword_manager import KeywordManager

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1cb4Pmc7SFCZ3C5kJD8kkDFQsuJXdk16a1afoRElJ3L0/edit?gid=0#gid=0"

MODEL_NAMES_ALIASES = {
    "gemini_3_pro": "Gemini",
    "claude_opus4_6": "Claude",
    "qwen_3_5": "Qwen",
    "w_en_gemini_3_pro": "Gemini (Translated)",
    "w_en_claude_opus4_6": "Claude (Translated)",
    "w_en_qwen_3_5": "Qwen (Translated)"
}


# Function to parse arguments
def get_config(results_dir, project_root=None):
    # Ensure results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # List only merged JSON files in results directory
    json_files = sorted([f for f in os.listdir(results_dir) if f.startswith('merged_') and f.endswith('.json')])

    # Default to first file if available, otherwise None
    default_input = json_files[0] if json_files else None

    # Default keywords file
    keywords_file = None
    fields_file = None
    # If project_root not passed, walk up from results_dir until we find a data/ folder
    if project_root is None:
        candidate = results_dir
        for _ in range(5):
            candidate = os.path.dirname(candidate)
            if os.path.exists(os.path.join(candidate, "data", "Keywords.csv")):
                project_root = candidate
                break
        if project_root is None:
            project_root = os.path.dirname(os.path.dirname(results_dir))

    kw_path = os.path.join(project_root, "data", "Keywords.csv")
    if os.path.exists(kw_path):
        keywords_file = kw_path

    tp_path = os.path.join(project_root, "data", "Topics.csv")
    if os.path.exists(tp_path):
        fields_file = tp_path

    for i, arg in enumerate(sys.argv):
        if arg == "--input_file" and i + 1 < len(sys.argv):
            default_input = sys.argv[i + 1]
        if arg == "--keywords_file" and i + 1 < len(sys.argv):
            keywords_file = sys.argv[i + 1]
        if arg == "--fields_file" and i + 1 < len(sys.argv):
            fields_file = sys.argv[i + 1]

    return default_input, keywords_file, fields_file, json_files


def create_annotation(result, filename,
                      kw_kept_ids, kw_new_accepted,
                      field_kept_ids, field_miss_ids,
                      index_kept_terms, index_miss_terms,
                      annotator_comments="",
                      dup_keywords=None,
                      kw_fn_ids=None,
                      user_defined_topics=None,
                      user_defined_keywords=None,
                      user_defined_index_terms=None,
                      is_non_analyzed=False):
    """Creates the annotation dictionary for 3-Vector Review."""
    if dup_keywords is None:
        dup_keywords = []
    original_row = result.get('original_row', {})

    # Gold reference for meta-data and metrics
    def get_gold_list(key):
        val = original_row.get(key, '')
        if val and str(val).strip() and str(val).lower() != 'nan':
            return [v.strip() for v in str(val).split(',') if v.strip()]
        return []

    gold_kw_ids = get_gold_list('KW Ids')
    gold_field_ids = get_gold_list('Judicial Topic Ids')
    gold_index_terms = get_gold_list('Index Terms')

    return {
        "results_filename": result["origin_file"].split("/")[-1],
        "annotator": st.session_state.get('name', ''),
        "date": date.today().isoformat(),
        # "ref_id": original_row.get("Refference") or original_row.get("ref Code"),
        # convert ref_id to string at creation time
        "ref_id": str(original_row.get("Refference") or original_row.get("ref Code") or ""),
        "source_id": result.get('source_id'),
        "group": result.get("group"),
        "name": result.get("name"),
        "text": result.get("text"),

        # Keywords
        "orig_kw_ids": result.get('matched_ids', []),
        "kw_kept_ids": kw_kept_ids,
        # "kw_manually_added_ids": kw_man_ids,
        "kw_accepted_new": kw_new_accepted,
        "gold_kw_ids": gold_kw_ids,

        # Judicial Fields
        "orig_field_ids": result.get('matched_field_ids', []),
        "field_kept_ids": field_kept_ids,
        "field_miss_agreed_ids": field_miss_ids,
        "gold_field_ids": gold_field_ids,

        # Index Terms (Lists of strings)
        "orig_index_terms": result.get('index_terms', []),
        "index_kept_terms": index_kept_terms,
        "index_miss_agreed_terms": index_miss_terms,
        "gold_index_terms": gold_index_terms,
        "annotator_comments": annotator_comments,
        "dup_keywords": dup_keywords,

        # Non-analyzed specific fields
        "is_non_analyzed": is_non_analyzed,
        "kw_fn_ids": kw_fn_ids or [],
        "user_defined_topics": user_defined_topics or [],
        "user_defined_keywords": user_defined_keywords or [],
        "user_defined_index_terms": user_defined_index_terms or [],
    }


def add_anno(result, filename,
             kw_kept_ids,
             kw_new_accepted,
             field_kept_ids, field_miss_ids,
             index_kept_terms, index_miss_terms,
             annotator_comments="",
             dup_keywords=None,
             next_i=False,
             kw_fn_ids=None,
             user_defined_topics=None,
             user_defined_keywords=None,
             user_defined_index_terms=None,
             is_non_analyzed=False):
    """Adds the 3-vector annotation to the session state."""
    annotation = create_annotation(
        result, filename,
        kw_kept_ids,
        kw_new_accepted,
        field_kept_ids, field_miss_ids,
        index_kept_terms, index_miss_terms,
        annotator_comments,
        dup_keywords,
        kw_fn_ids=kw_fn_ids,
        user_defined_topics=user_defined_topics,
        user_defined_keywords=user_defined_keywords,
        user_defined_index_terms=user_defined_index_terms,
        is_non_analyzed=is_non_analyzed,
    )

    # Add to the session buffer
    st.session_state.annotations.append(annotation)

    # Update keyword manager (for new suggested keywords that were accepted)
    if kw_new_accepted:
        st.session_state.keyword_manager.update_keywords(kw_new_accepted)

    # Increment index
    if next_i:
        st.session_state.current_index += 1


def load_all_models(results_dir):
    """Load all merged JSON model files once and cache in session state."""
    if 'all_models_data' not in st.session_state:
        all_models_data = {}
        models_by_ref = {}  # ref_id -> set of filenames that contain it
        json_files = sorted([f for f in os.listdir(results_dir)
                             if f.startswith('merged_') and f.endswith('.json')])
        for fn in json_files:
            try:
                with open(os.path.join(results_dir, fn), 'r', encoding='utf-8') as f:
                    data = json.load(f)
                all_models_data[fn] = data
                for item in data:
                    rid = item.get('ref_id')
                    if rid is not None:
                        rid_key = str(rid)
                        models_by_ref.setdefault(rid_key, set()).add(fn)
            except Exception as e:
                st.error(f"Error loading {fn}: {e}")
        st.session_state.all_models_data = all_models_data
        st.session_state.models_by_ref = models_by_ref

    # Ensure keywords & fields are loaded once
    if not st.session_state.get('keywords'):
        try:
            loader = DataLoader()
            st.session_state.keywords = loader.load_keywords(st.session_state.keywords_file)
        except Exception as e:
            st.error(f"Error loading keywords: {e}")
            st.session_state.keywords = []
    if not st.session_state.get('fields'):
        try:
            if hasattr(st.session_state, 'fields_file') and st.session_state.fields_file:
                loader = DataLoader()
                st.session_state.fields = loader.load_judicial_fields(st.session_state.fields_file)
            else:
                st.session_state.fields = []
        except Exception as e:
            st.error(f"Error loading fields: {e}")
            st.session_state.fields = []


def switch_model(selected_file, all_models_data):
    """Switch the active model pointer and preserve index."""
    if not selected_file or st.session_state.get('input_file_basename') == selected_file:
        return

    new_results = all_models_data.get(selected_file, [])
    if not new_results: return

    # Find the current ref_id to preserve position
    current_ref_id = None
    old_results_file = st.session_state.get('input_file_basename')
    current_index = st.session_state.get('current_index', 0)

    if old_results_file:
        old_results = all_models_data.get(old_results_file, [])
        if current_index < len(old_results):
            current_ref_id = str(old_results[current_index].get('ref_id', ''))

    st.session_state.input_file_basename = selected_file

    new_index = 0
    if current_ref_id:
        for i, res in enumerate(new_results):
            if str(res.get('ref_id', '')) == current_ref_id:
                new_index = i
                break

    st.session_state.current_index = new_index


# def switch_model(selected_file):
#     """Switch to a different model, preserving the current ref_id position."""
#     if not selected_file:
#         return
#     if st.session_state.get('input_file_basename') == selected_file:
#         return  # Already on this model

#     new_results = st.session_state.all_models_data.get(selected_file, [])
#     if not new_results:
#         return

#     # Find the current ref_id to preserve position
#     current_ref_id = None
#     old_results = st.session_state.get('results', [])
#     current_index = st.session_state.get('current_index', 0)
#     if old_results and current_index < len(old_results):
#         current_ref_id = str(old_results[current_index].get('ref_id', ''))

#     # Set the new results
#     st.session_state.results = new_results
#     st.session_state.input_file_basename = selected_file

#     # Find matching ref_id in the new model's results
#     new_index = 0
#     if current_ref_id:
#         for i, res in enumerate(new_results):
#             if str(res.get('ref_id', '')) == current_ref_id:
#                 new_index = i
#                 break

#     st.session_state.current_index = new_index
#     st.session_state.data_loaded = True


def display_source_text(text_content, language=""):
    is_ltr = language.strip().lower() in ["latin", "greek", "english"]
    direction = "ltr" if is_ltr else "rtl"
    text_align = "left" if is_ltr else "right"

    st.markdown(
        f"""
        <div style="
            direction: {direction}; 
            text-align: {text_align}; 
            border: 1px solid #ccc; 
            padding: 10px; 
            border-radius: 5px; 
            height: auto; 
            min-height: fit-content;
            width: fit-content; 
            max-width: 100%;    
        ">
            {text_content}
        </div>
        """,
        unsafe_allow_html=True
    )


def compute_sample_metrics(gold_ids, pred_ids):
    """Compute precision, recall, and Jaccard index for a single sample."""
    gold = set(str(g).strip().lower() for g in gold_ids if str(g).strip())
    pred = set(str(p).strip().lower() for p in pred_ids if str(p).strip())
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    union = gold | pred
    jaccard = len(gold & pred) / len(union) if union else 0.0
    return precision, recall, jaccard


def _load_annotation_sheets(results_dir):
    """Loads and deduplicates annotation data from all Google Sheet tabs."""
    if 'conn' not in st.session_state:
        st.session_state.conn = st.connection("gsheets", type=GSheetsConnection)

    sheet_names = []
    if results_dir and os.path.exists(results_dir):
        json_files = sorted([f for f in os.listdir(results_dir)
                             if f.startswith('merged_') and f.endswith('.json')])
        base_names = [f.replace('merged_', '').replace('.json', '') for f in json_files]
        # w_en_ variants generated from the base names (canonical names used by save_results)
        w_en_names = [f"w_en_{name}" for name in base_names if not name.startswith('w_en_')]
        # Legacy short names (old saves used origin_file-derived names like w_en_gemini / w_en_claude)
        legacy_w_en = [f"w_en_{n[len('w_en_'):].split('_')[0]}" for n in w_en_names]
        all_names = base_names + w_en_names
        sheet_names = all_names + [n for n in legacy_w_en if n not in all_names]

    all_dfs = []
    for sheet_name in sheet_names:
        try:
            df = st.session_state.conn.read(worksheet=sheet_name, ttl=300)
            if df is None or df.empty:
                continue
            # Keep the last annotation per unique record
            # dedup_cols = [c for c in ['results_filename', 'ref_id', 'name'] if c in df.columns]
            # dedup_cols = [c for c in ['results_filename', 'name'] if c in df.columns]
            dedup_cols = [c for c in ['name'] if c in df.columns]
            if dedup_cols:
                df = df.drop_duplicates(subset=dedup_cols, keep='last')
            names = sorted(df["name"].dropna().astype(str).tolist()) if "name" in df.columns else []
            print(f"[{sheet_name}] {len(names)} unique names:")
            for n in names:
                print(f"  {n}")
            df['_sheet'] = MODEL_NAMES_ALIASES.get(sheet_name, sheet_name)
            all_dfs.append(df)
        except Exception:
            pass

    return pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()


def _compute_annotation_metrics(df, vector):
    """
    Recomputes orig/mod precision, recall, Jaccard per row from raw list columns.
    orig_*: LLM predictions vs initial gold
    mod_*:  LLM predictions vs updated gold (annotator's final output)
    """

    def parse_col(val):
        if pd.isna(val) or not str(val).strip() or str(val).strip() == 'nan':
            return []
        return [s.strip() for s in str(val).split(',') if s.strip()]

    col_map = {
        "Keywords": {
            "pred": "orig_kw_ids",
            "init_gold": "gold_kw_ids",
            "mod_gold": ["kw_kept_ids"],
            "prefix": "kw",
        },
        "Judicial Fields": {
            "pred": "orig_field_ids",
            "init_gold": "gold_field_ids",
            "mod_gold": ["field_kept_ids", "field_miss_agreed_ids"],
            "prefix": "field",
        },
        "Index Terms": {
            "pred": "orig_index_terms",
            "init_gold": "gold_index_terms",
            "mod_gold": ["index_kept_terms", "index_miss_agreed_terms"],
            "prefix": "index",
        },
    }[vector]

    rows = []
    for _, r in df.iterrows():
        pred = parse_col(r.get(col_map["pred"], ""))
        init_gold = parse_col(r.get(col_map["init_gold"], ""))
        mod_gold = sum([parse_col(r.get(c, "")) for c in col_map["mod_gold"]], [])

        if not init_gold and not pred:
            continue

        op, or_, oj = compute_sample_metrics(init_gold, pred)
        mp, mr, mj = compute_sample_metrics(mod_gold, pred)
        p = col_map["prefix"]
        rows.append({
            "Model": r.get('_sheet', ''),
            "ref_id": r.get('ref_id', ''),
            "group": r.get('group', ''),
            "name": r.get('name', ''),
            f"{p}_orig_p": round(op, 3),
            f"{p}_mod_p": round(mp, 3),
            f"{p}_orig_r": round(or_, 3),
            f"{p}_mod_r": round(mr, 3),
            f"{p}_orig_j": round(oj, 3),
            f"{p}_mod_j": round(mj, 3),
        })

    return pd.DataFrame(rows)


def display_metrics(results_dir=None):
    """Renders the aggregated metrics dashboard page."""
    st.title("📊 Aggregated Metrics Dashboard")

    if st.button("⬅ Back to Annotation"):
        st.session_state.show_metrics = False
        if not st.session_state.get('input_file_basename'):
            st.session_state.show_instructions = True
        st.rerun()

    # --- Section: LLM Performance Summary (Original Predictions) ---
    st.markdown("### LLM Performance Summary (Original Predictions)")

    vector = st.radio("Select Vector for Metrics:", ["Keywords", "Judicial Fields", "Index Terms"], horizontal=True)
    v_prefix = {"Keywords": "kw", "Judicial Fields": "field", "Index Terms": "index"}[vector]
    gold_key = {"Keywords": "KW Ids", "Judicial Fields": "Judicial Topic Ids", "Index Terms": "Index Terms"}[vector]
    pred_key = {"Keywords": "matched_ids", "Judicial Fields": "matched_field_ids", "Index Terms": "index_terms"}[vector]

    if results_dir and os.path.exists(results_dir):
        summary_rows = get_cached_metrics(results_dir, gold_key, pred_key)
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows).style.format(precision=3), hide_index=True)
        else:
            st.info(f"No gold standard data found for {vector} in result files.")

    st.markdown("---")
    st.markdown("### ✍️ Annotation-Based Metrics")

    with st.spinner("Loading annotations from Google Sheets..."):
        combined_df = _load_annotation_sheets(results_dir)

    if combined_df.empty:
        st.info("No annotation data found in Google Sheets yet.")
        return

    metrics_df = _compute_annotation_metrics(combined_df, vector)

    if metrics_df.empty:
        st.info(f"No annotated records with gold standard found for {vector}.")
        return

    p = v_prefix

    def render_metric_row(label, subset_df, n_samples):
        st.markdown(f"**{label}** — {n_samples} records")
        c1, c2, c3 = st.columns(3)
        c1.metric("Jaccard (Orig → Mod)",
                  f"{subset_df[f'{p}_orig_j'].mean():.3f} → {subset_df[f'{p}_mod_j'].mean():.3f}")
        c2.metric("Precision (Orig → Mod)",
                  f"{subset_df[f'{p}_orig_p'].mean():.3f} → {subset_df[f'{p}_mod_p'].mean():.3f}")
        c3.metric("Recall (Orig → Mod)",
                  f"{subset_df[f'{p}_orig_r'].mean():.3f} → {subset_df[f'{p}_mod_r'].mean():.3f}")

    st.markdown(f"#### Dataset Level Averages ({vector})")
    # render_metric_row("All Models", metrics_df, len(metrics_df))

    # st.markdown("##### Per Model")
    for model_name, model_df in metrics_df.groupby("Model"):
        with st.expander(f"**{model_name}** — {len(model_df)} records", expanded=True):
            render_metric_row(model_name, model_df, len(model_df))

    st.markdown("#### Sample Level Details")
    st.dataframe(metrics_df, hide_index=True)


def display_instructions(available_models, available_files):
    """Renders the instructions / help page."""
    st.title("📖 Annotation Task Instructions")

    st.markdown("""
## Task Overview

You are reviewing LLM-generated keyword classifications for ancient legal texts.
Each sample shows a source text alongside the keywords a model assigned to it.
Your goal is to **correct** the classification by:

1. **Reviewing matched keywords** — uncheck any keyword that is **irrelevant**
   to the source text (false positives).
2. **Adding missed keywords** — select existing keywords from the thesaurus that
   the model failed to identify (false negatives).
3. **Evaluating suggested keywords** — the model may propose new keywords not yet
   in the thesaurus. Accept, edit, or reject each suggestion.
4. **Defining new keywords** — you may type in entirely new keywords if needed.

The **Gold Annotated Keywords** column (right) shows the human-annotated ground
truth so you can compare against the model's predictions (left).

---

## Metric Definitions

The metrics below are computed **per sample** by comparing the model's predicted
keyword IDs against the gold-standard annotation IDs.

| Metric | Formula | Meaning |
|---|---|---|
| **Precision** | TP / (TP + FP) | Of the keywords the model predicted, how many are correct? |
| **Recall** | TP / (TP + FN) | Of the gold keywords, how many did the model find? |
| **Jaccard Index** | \\|Gold ∩ Pred\\| / \\|Gold ∪ Pred\\| | Overall overlap between the two sets (1 = perfect match, 0 = no overlap). |

Where **TP** = true positives (correctly predicted), **FP** = false positives
(predicted but not in gold), **FN** = false negatives (in gold but not predicted).
    """)

    st.markdown("### Choose a Model to Begin")
    selected_model = st.selectbox(
        "Select Model",
        options=available_models,
        help="You must select a model before starting annotation."
    )

    if st.button("▶ Begin Annotation", type="primary", disabled=not selected_model):
        selected_file = next((fn for fn in available_files if
                              MODEL_NAMES_ALIASES.get(fn.replace("merged_", "").replace(".json", ""),
                                                      fn) == selected_model), None)

        st.session_state.input_file_basename = selected_file
        st.session_state.show_instructions = False
        st.rerun()


_PUNCT_TABLE = str.maketrans('', '', string.punctuation)


def _strip_display_punct(s):
    return s.translate(_PUNCT_TABLE)


def render_suggestion_list(suggestions, current_id, key_prefix, item_map=None, show_dup=False, group_by_category=False, label_fn=None):
    """
    Renders an expandable suggestion list with checkboxes.
    Returns a list of accepted items.
    """
    st.write("**Suggestions** (Accept/Reject)")
    kept_items = []
    dup_items = []

    def render_item(item, header_cat=None):
        if item_map is not None:
            obj = item_map.get(str(item).strip().lower())
            label = obj.full_path if obj else f"Unknown ID: {item}"
        else:
            label = str(item)

        if label_fn is not None:
            label = label_fn(label)

        if header_cat and label.startswith(header_cat):
            label = label[len(header_cat):].strip()
            if label.startswith(">"):
                label = label[1:].strip()
            if not label:
                label = header_cat

        if show_dup:
            c_dup, c_acc, c_label = st.columns([0.27, 0.1, 0.63], vertical_alignment="center")
        else:
            c_acc, c_label = st.columns([0.1, 0.9], vertical_alignment="center")

        with c_label:
            html_box = f"""
            <div style="
                padding: 10px;
                margin-bottom: 12px;
                border: 1px solid rgba(128, 128, 128, 0.3);
                border-radius: 8px;
                background-color: rgba(128, 128, 128, 0.1);
                color: inherit;
                word-wrap: break-word;
                font-size: 14px;
            ">
                {label}
            </div>
            """
            st.markdown(html_box, unsafe_allow_html=True)

        if show_dup:
            with c_dup:
                dup_key = f"{key_prefix}_dup_{current_id}_{item}"
                if dup_key not in st.session_state: st.session_state[dup_key] = False
                if st.toggle("dup", key=dup_key, help=None):
                    dup_items.append(item)
        with c_acc:
            acc_key = f"{key_prefix}_{current_id}_{item}"
            if acc_key not in st.session_state: st.session_state[acc_key] = False
            if st.checkbox("", key=acc_key):
                kept_items.append(item)

    if suggestions:
        if group_by_category and item_map is not None:
            groups = {}
            for item in suggestions:
                obj = item_map.get(str(item).strip().lower())
                cat = obj.full_path.split(">")[0].strip() if hasattr(obj, 'full_path') else "Uncategorized"
                groups.setdefault(cat, []).append(item)

            for cat in sorted(groups.keys()):
                if show_dup:
                    _, _, c_header = st.columns([0.27, 0.1, 0.63], vertical_alignment="center")
                else:
                    _, c_header = st.columns([0.1, 0.9], vertical_alignment="center")
                with c_header:
                    # st.markdown(f"###### {cat}")
                    st.markdown(f"**{cat}**")

                for item in groups[cat]:
                    render_item(item, header_cat=cat)
        else:
            for item in suggestions:
                render_item(item)
    else:
        st.caption("No suggestions.")

    if show_dup:
        return kept_items, dup_items
    return kept_items


def ensure_worksheet_exists(conn, sheet_url, worksheet_name):
    """
    Checks if a worksheet exists in the Google Sheet.
    If it doesn't, it creates a new one.
    """
    try:
        # Access the underlying gspread client and open the spreadsheet
        client = conn.client
        spreadsheet = client.open_by_url(sheet_url)

        # Try to access the worksheet
        try:
            spreadsheet.worksheet(worksheet_name)
        except Exception:
            # If it fails, the worksheet likely doesn't exist, so we create it.
            # 100 rows and 40 columns is a safe starting size.
            spreadsheet.add_worksheet(title=worksheet_name, rows=100, cols=40)
            st.toast(f"Created new tab: {worksheet_name}")

    except Exception:
        # GSheetsServiceAccountClient may not support open_by_url;
        # conn.update() will handle worksheet creation as needed.
        pass


def get_cached_models_data(results_dir):
    """Reads the directory live, then passes the exact file list to the cached loader."""
    # Find files live so we know exactly what is in the folder right now
    json_files = tuple(sorted([
        f for f in os.listdir(results_dir)
        if f.startswith('merged_') and f.endswith('.json')
    ]))

    # Pass the tuple of filenames to the cached function.
    # If the file list changes, the cache automatically invalidates!
    return _load_models_data_cached(results_dir, json_files)


@st.cache_resource
def _load_models_data_cached(results_dir, json_files):
    """The actual heavy lifting, cached strictly based on the list of filenames.
    Uses cache_resource (no deep copy) since the data is large and read-only."""
    all_models_data = {}
    models_by_ref = {}

    for fn in json_files:
        try:
            with open(os.path.join(results_dir, fn), 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_models_data[fn] = data
            for item in data:
                rid = item.get('ref_id')
                if rid is not None:
                    rid_key = str(rid)
                    models_by_ref.setdefault(rid_key, set()).add(fn)
        except Exception as e:
            st.error(f"Error loading {fn}: {e}")

    return all_models_data, models_by_ref


@st.cache_resource
def load_taxonomies(keywords_file, fields_file):
    """Loads keywords and judicial fields once, caching the results."""
    loader = DataLoader()

    try:
        keywords = loader.load_keywords(keywords_file) if keywords_file else []
    except Exception as e:
        st.error(f"Error loading keywords: {e}")
        keywords = []

    try:
        fields = loader.load_judicial_fields(fields_file) if fields_file else []
    except Exception as e:
        st.error(f"Error loading fields: {e}")
        fields = []

    return keywords, fields


@st.cache_data
def load_local_credentials(config_path):
    import yaml
    from yaml.loader import SafeLoader
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=SafeLoader)


@st.cache_data(show_spinner=False)
def load_auth_config(project_root):
    """
    Reads and caches credentials from Streamlit secrets or local YAML.
    Returns: tuple(credentials_dict, cookie_config_dict)
    """
    try:
        credentials = dict(st.secrets["auth_credentials"])
        # Convert nested AttrDict to plain dicts for streamlit-authenticator
        credentials["usernames"] = {
            user: dict(data) for user, data in st.secrets["auth_credentials"]["usernames"].items()
        }
        cookie_config = dict(st.secrets["auth_cookie"])
        return credentials, cookie_config

    except (KeyError, FileNotFoundError):
        # Fallback to local config.yaml
        config_path = os.path.join(project_root, "config.yaml")
        config = load_local_credentials(config_path)  # Assuming this is defined elsewhere
        return config['credentials'], config['cookie']


def setup_authenticator(project_root):
    """
    Initializes the authenticator and renders the login UI.
    Returns: tuple(authenticator_object, is_authenticated_boolean)
    """
    credentials, cookie_config = load_auth_config(project_root)

    authenticator = stauth.Authenticate(
        credentials,
        cookie_config['name'],
        cookie_config['key'],
        cookie_config['expiry_days']
    )

    try:
        authenticator.login()
    except Exception as e:
        st.error(e)

    return authenticator, st.session_state.get('authentication_status')


def build_taxonomy_maps(keywords, fields):
    """Creates fast-lookup dictionaries for keywords and fields."""
    kw_map = {str(k.id).strip().lower(): k for k in keywords}
    field_map = {str(f.id).strip().lower(): f for f in fields}
    return kw_map, field_map


def get_cached_metrics(results_dir, gold_key, pred_key):
    """Reads directory live, passes file list to cached metrics calculator."""
    if not results_dir or not os.path.exists(results_dir):
        return []
    json_files = tuple(sorted([f for f in os.listdir(results_dir) if f.endswith('.json')]))
    return _compute_metrics_cached(results_dir, json_files, gold_key, pred_key)


@st.cache_data
def _compute_metrics_cached(results_dir, json_files, gold_key, pred_key):
    """Heavy lifting for metrics, cached based on the exact list of files."""
    summary_rows = []
    for jf in json_files:
        try:
            with open(os.path.join(results_dir, jf), 'r', encoding='utf-8') as f:
                data = json.load(f)

            precisions, recalls, jaccards = [], [], []

            for item in data:
                if "error" in item: continue
                gold_raw = item.get('original_row', {}).get(gold_key, '')
                if not gold_raw or str(gold_raw).lower() == 'nan':
                    continue

                gold = [g.strip() for g in str(gold_raw).split(',') if g.strip()]
                pred = item.get(pred_key, [])

                p, r, j = compute_sample_metrics(gold, pred)
                precisions.append(p)
                recalls.append(r)
                jaccards.append(j)

            if precisions:
                df = pd.DataFrame({'p': precisions, 'r': recalls, 'j': jaccards})
                model = MODEL_NAMES_ALIASES[jf.replace("merged_", "").replace(".json", "")]
                summary_rows.append({
                    "Model": model,
                    "Samples": len(data),
                    "Gold Count": len(precisions),
                    "Avg Prec": df['p'].mean(),
                    "Avg Rec": df['r'].mean(),
                    "Avg Jac": df['j'].mean()
                })
        except Exception as e:
            st.error(f"Error processing {jf}: {e}")

    return summary_rows


def load_annotated_data_from_sheet(sheet_name, current_id, result, original_row):
    try:
        if 'conn' not in st.session_state:
            st.session_state.conn = st.connection("gsheets", type=GSheetsConnection)

        try:
            if 'sheet_cache' not in st.session_state:
                st.session_state.sheet_cache = {}

            if sheet_name in st.session_state.sheet_cache:
                df = st.session_state.sheet_cache[sheet_name]
            else:
                df = st.session_state.conn.read(worksheet=sheet_name, ttl=0)
                if df is not None and not df.empty:
                    st.session_state.sheet_cache[sheet_name] = df

            if df is None or df.empty:
                st.error("Sheet is empty or does not exist.")
                return False
        except Exception as e:
            st.error(f"Failed to read sheet: {e}")
            return False

        target_filename = result.get("origin_file", "").split("/")[-1]
        target_name = str(result.get("name", ""))
        target_ref = str(original_row.get("Refference") or original_row.get("ref Code", ""))

        # Standard string matching without clean_id
        match_mask = (
                (df['results_filename'].astype(str) == target_filename) &
                (df['name'].astype(str) == target_name)
            # & (df['ref_id'].astype(str) == target_ref)
        )

        matching_rows = df[match_mask]

        if matching_rows.empty:
            st.warning("No pre-annotated data found for this record.")
            return False

        row_data = matching_rows.iloc[-1]

        def parse_list(val):
            if pd.isna(val) or not str(val).strip(): return []
            return [s.strip() for s in str(val).split(',')]

        # 1. Parse all the saved data
        kw_kept = parse_list(row_data.get('kw_kept_ids', ''))
        kw_acc_new = parse_list(row_data.get('kw_accepted_new', ''))
        field_kept = parse_list(row_data.get('field_kept_ids', ''))
        field_miss = parse_list(row_data.get('field_miss_agreed_ids', ''))
        index_kept = parse_list(row_data.get('index_kept_terms', ''))
        index_miss = parse_list(row_data.get('index_miss_agreed_terms', ''))
        comments = str(row_data.get('annotator_comments', ''))
        if comments == 'nan' or pd.isna(row_data.get('annotator_comments')):
            comments = ''

        # 2. Clean up old session state checkboxes for this specific record
        keys_to_delete = [k for k in st.session_state.keys() if f"_{current_id}_" in k]
        for k in keys_to_delete:
            del st.session_state[k]

        # 3. Compute the same intersection/suggestion/missed splits the UI uses,
        #    so we can set each checkbox to exactly the right state.
        import unicodedata as _ud

        def _norm(s):
            return _ud.normalize('NFC', str(s).strip().lower())

        pred_set = set(_norm(m) for m in result.get('matched_ids', []))
        gold_kw = [g.strip() for g in str(original_row.get('KW Ids', '')).split(',') if g.strip()]
        gold_f_list = [g.strip() for g in str(original_row.get('Judicial Topic Ids', '')).split(',') if g.strip()]
        gold_i_list = [g.strip() for g in str(original_row.get('Index Terms', '')).split(',') if g.strip()]

        gold_kw_set = set(_norm(g) for g in gold_kw)
        gold_f_set = set(_norm(g) for g in gold_f_list)
        gold_i_set = set(_norm(g) for g in gold_i_list)

        kw_kept_set = set(_norm(k) for k in kw_kept)
        field_kept_set = set(_norm(f) for f in field_kept)
        field_miss_set = set(_norm(f) for f in field_miss)
        index_kept_set = set(_norm(t) for t in index_kept)
        index_miss_set = set(_norm(t) for t in index_miss)

        # Keywords — suggestions (pred not in gold)
        for item in result.get('matched_ids', []):
            if _norm(item) not in gold_kw_set:  # it's a suggestion, not intersection
                st.session_state[f"kw_sug_{current_id}_{item}"] = (_norm(item) in kw_kept_set)

        # Keywords — missed gold (gold not in pred)
        for gid in gold_kw:
            if _norm(gid) not in pred_set:
                st.session_state[f"kw_miss_{current_id}_{gid}"] = (_norm(gid) in kw_kept_set)

        # New keyword suggestions (suggestions not in taxonomy)
        for i, skw in enumerate(result.get('suggested_kws', [])):
            st.session_state[f"kw_new_acc_{current_id}_{i}"] = (skw in kw_acc_new)
            st.session_state[f"kw_new_edit_{current_id}_{i}"] = skw

        # Fields — suggestions
        for fid in result.get('matched_field_ids', []):
            if _norm(fid) not in gold_f_set:
                st.session_state[f"f_sug_{current_id}_{fid}"] = (_norm(fid) in field_kept_set)

        # Fields — missed gold
        for gid in gold_f_list:
            if _norm(gid) not in set(_norm(f) for f in result.get('matched_field_ids', [])):
                st.session_state[f"f_miss_{current_id}_{gid}"] = (_norm(gid) in field_miss_set)

        # Index terms — suggestions
        for term in result.get('index_terms', []):
            if _norm(term) not in gold_i_set:
                st.session_state[f"i_sug_{current_id}_{term}"] = (_norm(term) in index_kept_set)

        # Index terms — missed gold
        for term in gold_i_list:
            if _norm(term) not in set(_norm(p) for p in result.get('index_terms', [])):
                st.session_state[f"i_miss_{current_id}_{term}"] = (_norm(term) in index_miss_set)

        st.session_state[f"comments_{current_id}"] = comments

        st.toast("Loaded annotated data successfully!")
        return True

    except Exception as e:
        st.error(f"Error loading annotated data: {e}")
        return False


def render_fields_review(result, original_row, field_map, current_id):
    with st.expander("**Judicial Fields Review**", expanded=False):
        matched_field_ids = result.get('matched_field_ids', [])
        gold_field_ids_raw = original_row.get('Judicial Topic Ids', '')
        has_gold_f = (
                gold_field_ids_raw and str(gold_field_ids_raw).strip() and str(gold_field_ids_raw).lower() != 'nan')
        gold_f_set = set(
            [g.strip().lower() for g in str(gold_field_ids_raw).split(',') if g.strip()]) if has_gold_f else set()
        pred_f_set = set(str(fid).strip().lower() for fid in matched_field_ids)
        pred_f_set = set(unicodedata.normalize('NFC', fid) for fid in pred_f_set)
        gold_f_set = set(unicodedata.normalize('NFC', gid) for gid in gold_f_set)

        f_intersection = sorted(list(pred_f_set & gold_f_set))
        f_suggestions = sorted(list(pred_f_set - gold_f_set))
        f_missed = sorted(list(gold_f_set - pred_f_set))

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.write("**Intersection** (Read-only)")
            if f_intersection:
                for fid in f_intersection:
                    f_obj = field_map.get(str(fid).strip().lower())
                    label = f_obj.full_path if f_obj else f"Unknown ID: {fid}"
                    st.info(f"✅ {label}")
            else:
                st.caption("No intersection.")

        with col_f2:
            field_kept_ids = render_suggestion_list(
                suggestions=f_suggestions,
                current_id=current_id,
                key_prefix="f_sug",
                item_map=field_map
            )

        st.write("**Missed Gold Fields** (Agree/Disagree)")
        field_miss_agreed_ids = []
        if f_missed:
            for fid in f_missed:
                f_obj = field_map.get(str(fid).strip().lower())
                label = f_obj.full_path if f_obj else f"Unknown ID: {fid}"
                c_acc, c_label = st.columns([0.03, 0.97], vertical_alignment="center")
                with c_label:
                    html_box = f"""
                    <div style="
                        padding: 10px;
                        margin-bottom: 12px;
                        border: 1px solid rgba(128, 128, 128, 0.3);
                        border-radius: 8px;
                        background-color: rgba(128, 128, 128, 0.1);
                        color: inherit;
                        word-wrap: break-word;
                        font-size: 14px;
                        width: fit-content;
                    ">
                        {label}
                    </div>
                    """
                    st.markdown(html_box, unsafe_allow_html=True)
                with c_acc:
                    miss_key = f"f_miss_{current_id}_{fid}"
                    if miss_key not in st.session_state: st.session_state[miss_key] = True
                    if st.checkbox("", key=miss_key):
                        field_miss_agreed_ids.append(fid)
        else:
            st.caption("No missed gold fields.")

    return field_kept_ids, field_miss_agreed_ids, f_intersection


def render_keywords_review(result, original_row, kw_map, current_id):
    with st.expander("**Keywords Review**", expanded=False):
        matched_ids = result.get('matched_ids', [])
        suggested_kws = result.get('suggested_kws', [])

        gold_kw_ids_raw = original_row.get('KW Ids', '')
        has_gold_kw = (gold_kw_ids_raw
                       and str(gold_kw_ids_raw).strip()
                       and str(gold_kw_ids_raw).lower() != 'nan')
        gold_ids_list = []
        if has_gold_kw:
            gold_ids_list = [g.strip() for g in str(gold_kw_ids_raw).split(',') if g.strip()]

        pred_set = set(str(mid).strip().lower() for mid in matched_ids)
        pred_set = set(unicodedata.normalize('NFC', mid) for mid in pred_set)
        gold_set = set(str(gid).strip().lower() for gid in gold_ids_list)
        gold_set = set(unicodedata.normalize('NFC', gid) for gid in gold_set)
        intersection_ids = sorted(list(pred_set & gold_set))
        suggestion_ids = sorted(list(pred_set - gold_set))
        missed_ids = sorted(list(gold_set - pred_set))

        dup_keywords = []
        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Intersection** (Read-only)")
            if intersection_ids:
                groups = {}
                for mid in intersection_ids:
                    kw_obj = kw_map.get(str(mid).strip().lower())
                    cat = kw_obj.full_path.split(">")[0].strip() if kw_obj and hasattr(kw_obj,
                                                                                       'full_path') else "Uncategorized"
                    groups.setdefault(cat, []).append(mid)

                for cat in sorted(groups.keys()):
                    _, c_header = st.columns([0.26, 0.74], vertical_alignment="center")
                    with c_header:
                        st.markdown(f"**{cat}**")

                    for mid in groups[cat]:
                        kw_obj = kw_map.get(str(mid).strip().lower())
                        label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"

                        if label.startswith(cat):
                            label = label[len(cat):].strip()
                            if label.startswith(">"): label = label[1:].strip()
                            if not label: label = cat

                        c_dup, c_label = st.columns([0.26, 0.74], vertical_alignment="center")
                        with c_dup:
                            if st.toggle("dup", key=f"kw_int_dup_{current_id}_{mid}", help=None):
                                dup_keywords.append(mid)
                        with c_label:
                            st.info(f"✅ {label}")
            else:
                st.caption("No intersection found.")

        with col2:
            kept_suggestion_ids, dup_sug_ids = render_suggestion_list(
                suggestions=suggestion_ids,
                current_id=current_id,
                key_prefix="kw_sug",
                item_map=kw_map,
                show_dup=True,
                group_by_category=True
            )
            dup_keywords.extend(dup_sug_ids)

        with col3:
            st.write("**New Suggestions** (Accept/Edit)")
            final_new_kws = []
            if suggested_kws:
                for i, skw in enumerate(suggested_kws):
                    c_acc, c_edit, c_reset = st.columns([0.1, 0.8, 0.1], vertical_alignment="center")
                    with c_edit:
                        edit_key = f"kw_new_edit_{current_id}_{i}"
                        if edit_key not in st.session_state: st.session_state[edit_key] = skw
                        edited_kw = st.text_input("Edit", key=edit_key, label_visibility="collapsed", help=skw)
                    with c_reset:
                        if st.button("", icon=":material/refresh:", key=f"kw_new_reset_{current_id}_{i}",
                                     help="Reset to original suggestion", use_container_width=True):
                            st.session_state[edit_key] = skw
                            st.rerun()
                    with c_acc:
                        acc_key = f"kw_new_acc_{current_id}_{i}"
                        if acc_key not in st.session_state: st.session_state[acc_key] = False
                        if st.checkbox("", key=acc_key):
                            final_new_kws.append(edited_kw)
            else:
                st.caption("No new suggestions.")

        st.write("**Missed Gold Keywords** (Agree/Disagree)")
        agreed_missed_ids = []
        if missed_ids:
            groups = {}
            for mid in missed_ids:
                kw_obj = kw_map.get(str(mid).strip().lower())
                cat = kw_obj.full_path.split(">")[0].strip() if kw_obj and hasattr(kw_obj,
                                                                                   'full_path') else "Uncategorized"
                groups.setdefault(cat, []).append(mid)

            for cat in sorted(groups.keys()):
                _, _, c_header = st.columns([0.12, 0.05, 0.83], vertical_alignment="center")
                with c_header:
                    st.markdown(f"**{cat}**")

                for mid in groups[cat]:
                    kw_obj = kw_map.get(str(mid).strip().lower())
                    label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"

                    if label.startswith(cat):
                        label = label[len(cat):].strip()
                        if label.startswith(">"): label = label[1:].strip()
                        if not label: label = cat

                    c_dup, c_acc, c_label = st.columns([0.12, 0.05, 0.83], vertical_alignment="center")
                    with c_label:
                        html_box = f"""
                        <div style="
                            padding: 10px;
                            margin-bottom: 12px;
                            border: 1px solid rgba(128, 128, 128, 0.3);
                            border-radius: 8px;
                            background-color: rgba(128, 128, 128, 0.1);
                            color: inherit;
                            word-wrap: break-word;
                            font-size: 14px;
                            width: fit-content;
                        ">
                            {label}
                        </div>
                        """
                        st.markdown(html_box, unsafe_allow_html=True)
                    with c_dup:
                        dup_key = f"kw_miss_dup_{current_id}_{mid}"
                        if dup_key not in st.session_state: st.session_state[dup_key] = False
                        if st.toggle("dup", key=dup_key, help=None):
                            dup_keywords.append(mid)
                    with c_acc:
                        miss_key = f"kw_miss_{current_id}_{mid}"
                        if miss_key not in st.session_state: st.session_state[miss_key] = True
                        if st.checkbox("", key=miss_key):
                            agreed_missed_ids.append(mid)
        else:
            st.caption("No missed gold keywords.")

    return intersection_ids, kept_suggestion_ids, agreed_missed_ids, final_new_kws, dup_keywords


def render_fields_review_non_analyzed(result, original_row, field_map, all_fields, current_id):
    """Non-analyzed: col1 = model predictions (all pre-checked), col2 = FN selection, below = free text."""
    with st.expander("**Judicial Fields Review**", expanded=False):
        matched_field_ids = result.get('matched_field_ids', [])

        col_f1, col_f2 = st.columns(2)

        with col_f1:
            st.write("**Suggestions** (Accept/Reject)")
            field_kept_ids = []
            if matched_field_ids:
                for fid in matched_field_ids:
                    f_obj = field_map.get(str(fid).strip().lower())
                    label = f_obj.full_path if f_obj else f"Unknown ID: {fid}"
                    c_acc, c_label = st.columns([0.1, 0.9], vertical_alignment="center")
                    with c_label:
                        st.markdown(f"""
                        <div style="padding:10px;margin-bottom:12px;border:1px solid rgba(128,128,128,0.3);
                            border-radius:8px;background-color:rgba(128,128,128,0.1);color:inherit;
                            word-wrap:break-word;font-size:14px;">{label}</div>
                        """, unsafe_allow_html=True)
                    with c_acc:
                        acc_key = f"f_pred_{current_id}_{fid}"
                        if acc_key not in st.session_state:
                            st.session_state[acc_key] = True
                        if st.checkbox("", key=acc_key):
                            field_kept_ids.append(fid)
            else:
                st.caption("No model predictions.")

        with col_f2:
            st.write("**Missed Fields** (False Negatives)")
            pred_f_set = set(str(fid).strip().lower() for fid in matched_field_ids)
            available = [f for f in all_fields if str(f.id).strip().lower() not in pred_f_set]
            option_labels = [f.full_path for f in available]
            label_to_id = {f.full_path: str(f.id) for f in available}
            fn_key = f"f_fn_{current_id}"
            selected = st.multiselect("", option_labels, key=fn_key, label_visibility="collapsed")
            field_fn_ids = [label_to_id[lbl] for lbl in selected if lbl in label_to_id]

        st.write("**Topics Missing from Hierarchy**")
        missing_key = f"f_missing_topics_{current_id}"
        if missing_key not in st.session_state:
            st.session_state[missing_key] = ""
        missing_text = st.text_area(
            "", key=missing_key, label_visibility="collapsed",
            placeholder="Enter topics not in the hierarchy, one per line"
        )
        user_defined_topics = [t.strip() for t in missing_text.splitlines() if t.strip()]

    return field_kept_ids, field_fn_ids, user_defined_topics


def render_keywords_review_non_analyzed(result, original_row, kw_map, all_keywords, current_id):
    """Non-analyzed: col1 = model predictions (all pre-checked), col2 = new suggestions, col3 = FN selection."""
    with st.expander("**Keywords Review**", expanded=False):
        matched_ids = result.get('matched_ids', [])
        suggested_kws = result.get('suggested_kws', [])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**Suggestions** (Accept/Reject)")
            kw_kept_ids = []
            if matched_ids:
                groups = {}
                for mid in matched_ids:
                    kw_obj = kw_map.get(str(mid).strip().lower())
                    cat = kw_obj.full_path.split(">")[0].strip() if kw_obj and hasattr(kw_obj, 'full_path') else "Uncategorized"
                    groups.setdefault(cat, []).append(mid)

                for cat in sorted(groups.keys()):
                    _, c_header = st.columns([0.1, 0.9], vertical_alignment="center")
                    with c_header:
                        st.markdown(f"**{cat}**")
                    for mid in groups[cat]:
                        kw_obj = kw_map.get(str(mid).strip().lower())
                        label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"
                        if label.startswith(cat):
                            label = label[len(cat):].strip()
                            if label.startswith(">"): label = label[1:].strip()
                            if not label: label = cat
                        c_acc, c_label = st.columns([0.1, 0.9], vertical_alignment="center")
                        with c_label:
                            st.markdown(f"""
                            <div style="padding:10px;margin-bottom:12px;border:1px solid rgba(128,128,128,0.3);
                                border-radius:8px;background-color:rgba(128,128,128,0.1);color:inherit;
                                word-wrap:break-word;font-size:14px;">{label}</div>
                            """, unsafe_allow_html=True)
                        with c_acc:
                            acc_key = f"kw_pred_{current_id}_{mid}"
                            if acc_key not in st.session_state:
                                st.session_state[acc_key] = True
                            if st.checkbox("", key=acc_key):
                                kw_kept_ids.append(mid)
            else:
                st.caption("No model predictions.")

        with col2:
            st.write("**New Suggestions** (Accept/Edit)")
            final_new_kws = []
            if suggested_kws:
                for i, skw in enumerate(suggested_kws):
                    c_acc, c_edit, c_reset = st.columns([0.1, 0.8, 0.1], vertical_alignment="center")
                    with c_edit:
                        edit_key = f"kw_new_edit_{current_id}_{i}"
                        if edit_key not in st.session_state: st.session_state[edit_key] = skw
                        edited_kw = st.text_input("Edit", key=edit_key, label_visibility="collapsed", help=skw)
                    with c_reset:
                        if st.button("", icon=":material/refresh:", key=f"kw_new_reset_{current_id}_{i}",
                                     help="Reset to original suggestion", use_container_width=True):
                            st.session_state[edit_key] = skw
                            st.rerun()
                    with c_acc:
                        acc_key = f"kw_new_acc_{current_id}_{i}"
                        if acc_key not in st.session_state: st.session_state[acc_key] = False
                        if st.checkbox("", key=acc_key):
                            final_new_kws.append(edited_kw)
            else:
                st.caption("No new suggestions.")

        with col3:
            st.write("**Missed Keywords** (False Negatives)")
            pred_kw_set = set(str(mid).strip().lower() for mid in matched_ids)
            available = [k for k in all_keywords if str(k.id).strip().lower() not in pred_kw_set]
            option_labels = [f"[{k.id}] {k.full_path}" for k in available]
            label_to_id = {f"[{k.id}] {k.full_path}": str(k.id) for k in available}
            fn_key = f"kw_fn_{current_id}"
            selected = st.multiselect("", option_labels, key=fn_key, label_visibility="collapsed")
            kw_fn_ids = [label_to_id[lbl] for lbl in selected if lbl in label_to_id]

        st.write("**Keywords Missing from Hierarchy**")
        missing_key = f"kw_missing_{current_id}"
        if missing_key not in st.session_state:
            st.session_state[missing_key] = ""
        missing_text = st.text_area(
            "", key=missing_key, label_visibility="collapsed",
            placeholder="Enter keywords not in the hierarchy, one per line"
        )
        user_defined_keywords = [k.strip() for k in missing_text.splitlines() if k.strip()]

    return kw_kept_ids, final_new_kws, kw_fn_ids, user_defined_keywords


def render_index_review_non_analyzed(result, original_row, current_id):
    """Non-analyzed: col1 = model predictions (all pre-checked), col2 = FN free text, below = missing terms."""
    with st.expander("**Index Terms Review**", expanded=False):
        pred_index = result.get('index_terms', [])

        col_i1, col_i2 = st.columns(2)

        with col_i1:
            st.write("**Suggestions** (Accept/Reject)")
            index_kept_terms = []
            if pred_index:
                for i, term in enumerate(pred_index):
                    c_acc, c_label = st.columns([0.1, 0.9], vertical_alignment="center")
                    with c_label:
                        st.markdown(f"""
                        <div style="padding:10px;margin-bottom:12px;border:1px solid rgba(128,128,128,0.3);
                            border-radius:8px;background-color:rgba(128,128,128,0.1);color:inherit;
                            word-wrap:break-word;font-size:14px;">{_strip_display_punct(term)}</div>
                        """, unsafe_allow_html=True)
                    with c_acc:
                        acc_key = f"i_pred_{current_id}_{i}_{term}"
                        if acc_key not in st.session_state:
                            st.session_state[acc_key] = True
                        if st.checkbox("", key=acc_key):
                            index_kept_terms.append(term)
            else:
                st.caption("No model predictions.")

        with col_i2:
            st.write("**Missed Index Terms** (False Negatives)")
            fn_key = f"i_fn_{current_id}"
            if fn_key not in st.session_state:
                st.session_state[fn_key] = ""
            fn_text = st.text_area("", key=fn_key, label_visibility="collapsed",
                                   placeholder="Enter missed index terms, one per line")
            index_fn_terms = [t.strip() for t in fn_text.splitlines() if t.strip()]

        st.write("**Index Terms Missing from Hierarchy**")
        missing_key = f"i_missing_{current_id}"
        if missing_key not in st.session_state:
            st.session_state[missing_key] = ""
        missing_text = st.text_area(
            "", key=missing_key, label_visibility="collapsed",
            placeholder="Enter index terms not in the hierarchy, one per line"
        )
        user_defined_index_terms = [t.strip() for t in missing_text.splitlines() if t.strip()]

    return index_kept_terms, index_fn_terms, user_defined_index_terms


def render_index_review(result, original_row, current_id):
    with st.expander("**Index Terms Review**", expanded=False):
        pred_index = result.get('index_terms', [])
        gold_index_raw = original_row.get('Index Terms', '')
        has_gold_i = (gold_index_raw and str(gold_index_raw).strip() and str(gold_index_raw).lower() != 'nan')
        gold_i_list = [g.strip() for g in str(gold_index_raw).split(',') if g.strip()] if has_gold_i else []

        pred_i_set = set(str(p).strip().lower() for p in pred_index)
        pred_i_set = set(unicodedata.normalize('NFC', p) for p in pred_i_set)
        gold_i_set = set(str(g).strip().lower() for g in gold_i_list)
        gold_i_set = set(unicodedata.normalize('NFC', g) for g in gold_i_set)

        i_intersection = sorted(list(pred_i_set & gold_i_set))
        i_suggestions = sorted(list(pred_i_set - gold_i_set))
        i_missed = sorted(list(gold_i_set - pred_i_set))

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.write("**Intersection** (Read-only)")
            if i_intersection:
                for term in i_intersection:
                    st.info(f"✅ {_strip_display_punct(term)}")
            else:
                st.caption("No intersection.")

        with col_i2:
            index_kept_terms = render_suggestion_list(
                suggestions=i_suggestions,
                current_id=current_id,
                key_prefix="i_sug",
                label_fn=_strip_display_punct
            )

        st.write("**Missed Gold Index Terms** (Agree/Disagree)")
        index_miss_agreed_terms = []
        if i_missed:
            for term in i_missed:
                c_acc, c_label = st.columns([0.03, 0.97], vertical_alignment="center")
                with c_label:
                    html_box = f"""
                    <div style="
                        padding: 10px;
                        margin-bottom: 12px;
                        border: 1px solid rgba(128, 128, 128, 0.3);
                        border-radius: 8px;
                        background-color: rgba(128, 128, 128, 0.1);
                        color: inherit;
                        word-wrap: break-word;
                        font-size: 14px;
                        width: fit-content;
                    ">
                        {_strip_display_punct(term)}
                    </div>
                    """
                    st.markdown(html_box, unsafe_allow_html=True)
                with c_acc:
                    miss_key = f"i_miss_{current_id}_{term}"
                    if miss_key not in st.session_state: st.session_state[miss_key] = True
                    if st.checkbox("", key=miss_key):
                        index_miss_agreed_terms.append(term)
        else:
            st.caption("No missed gold index terms.")

    return index_kept_terms, index_miss_agreed_terms, i_intersection


def main():
    # Fix: Safely resolve results_dir whether script is in root/ or src/
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Check if the results directory is sitting right next to the script (server/flat structure)
    if os.path.exists(os.path.join(script_dir, "results", "prioritized/tosefta")):
        results_dir = os.path.join(script_dir, "results", "prioritized/tosefta")
    else:
        # Assume the script is inside a subfolder (like src/), so go up one level (local structure)
        project_root = os.path.dirname(script_dir)
        results_dir = os.path.join(project_root, "results", "prioritized/tosefta")

    st.set_page_config(layout="centered",
                       page_title="RomanJewish Legal Classifier - Review")

    # Widen the centered content area for better annotation columns
    st.markdown("""
        <style>
            .block-container {
                max-width: 1000px;
                padding-left: 2rem;
                padding-right: 2rem;
            }
            /* Make disabled text inputs look normal (black text) */
            input:disabled {
                color: black !important;
                -webkit-text-fill-color: black !important;
            }
        </style>
    """, unsafe_allow_html=True)

    # --- AUTHENTICATION ---
    authenticator, auth_status = setup_authenticator(project_root)

    # 2. Gatekeeper Logic
    if auth_status is False:
        st.error('Username/password is incorrect')
        st.stop()  # st.stop() is slightly safer than return here
    elif auth_status is None:
        st.warning('Please enter your username and password')
        st.stop()

    # --- User is authenticated from here ---

    # --- INITIALIZE SESSION STATE KEYS ---
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'skipped_indices' not in st.session_state:
        st.session_state.skipped_indices = set()
    if 'keyword_manager' not in st.session_state:
        st.session_state.keyword_manager = KeywordManager()
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
    if 'fields' not in st.session_state:
        st.session_state.fields = []
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = True
    if 'show_metrics' not in st.session_state:
        st.session_state.show_metrics = False
    # -------------------------------------

    # Sidebar
    st.sidebar.title("Review Config")
    st.sidebar.write(f"Logged in as: **{st.session_state.get('name')}**")
    # authenticator.logout('Logout', 'sidebar')

    if st.sidebar.button("📖 Instructions", use_container_width=True):
        st.session_state.show_instructions = True
        st.session_state.show_metrics = False
        st.rerun()

    # Metrics and Sheet in two columns
    col1, col2 = st.sidebar.columns([1, 2])
    with col1:
        if st.button("📊 Metrics", use_container_width=True):
            st.session_state.show_metrics = True
            st.session_state.show_instructions = False
            st.rerun()
    with col2:
        st.link_button("📈 Go to Annotations", DEFAULT_SHEET_URL, use_container_width=True)
    st.sidebar.markdown("---")

    cli_input_file, cli_keywords_file, cli_fields_file, available_files = get_config(results_dir, project_root)
    # rename the available_files to more user-friendly model names using MODEL_NAMES_ALIASES
    available_models = [MODEL_NAMES_ALIASES.get(fn.replace("merged_", "").replace(".json", ""), fn) for fn in
                        available_files]
    st.session_state.keywords_file = cli_keywords_file
    st.session_state.fields_file = cli_fields_file

    # if not st.session_state.get('keywords'):
    #     try:
    #         loader = DataLoader()
    #         st.session_state.keywords = loader.load_keywords(st.session_state.keywords_file)
    #     except Exception as e:
    #         st.error(f"Error loading keywords: {e}")
    #         st.session_state.keywords = []
    #
    # if not st.session_state.get('fields'):
    #     try:
    #         if hasattr(st.session_state, 'fields_file') and st.session_state.fields_file:
    #             loader = DataLoader()
    #             st.session_state.fields = loader.load_judicial_fields(st.session_state.fields_file)
    #         else:
    #             st.session_state.fields = []
    #     except Exception as e:
    #         st.error(f"Error loading fields: {e}")
    #         st.session_state.fields = []

    if not st.session_state.get('keywords'):
        st.session_state.keywords, st.session_state.fields = load_taxonomies(
            st.session_state.keywords_file,
            st.session_state.fields_file
        )

    # Load all models data once (cached in session state)
    # load_all_models(results_dir)
    all_models_data, models_by_ref = get_cached_models_data(results_dir)
    st.session_state.models_by_ref = models_by_ref

    # Initialize with first available model if none selected yet
    if not st.session_state.get('input_file_basename') and available_files:
        st.session_state.input_file_basename = available_files[0]
        st.session_state.data_loaded = True

    # Get the current results directly from our cached data
    current_basename = st.session_state.get('input_file_basename')
    current_results = all_models_data.get(current_basename, [])

    # Determine which models have results for the current ref_id
    current_ref_id = None
    if current_results and st.session_state.get('current_index', 0) < len(current_results):
        current_ref_id = str(current_results[st.session_state.current_index].get('ref_id', ''))

    if current_ref_id and st.session_state.get('models_by_ref'):
        models_for_source = sorted(st.session_state.models_by_ref.get(current_ref_id, set()))
    else:
        models_for_source = available_files

    if models_for_source and not st.session_state.show_instructions and st.session_state.get('input_file_basename'):
        current_basename = st.session_state.get('input_file_basename')
        try:
            default_index = available_files.index(current_basename) if current_basename in available_files else 0
        except ValueError:
            default_index = 0

        st.sidebar.subheader("Switch Model")
        selected_model = st.sidebar.selectbox(
            "Switch Model",
            options=available_models,
            index=default_index,
            label_visibility="collapsed"
        )

        # revert the selected_model back to the actual filename using MODEL_NAMES_ALIASES
        selected_file = None
        for fn in available_files:
            alias = MODEL_NAMES_ALIASES.get(fn.replace("merged_", "").replace(".json", ""), fn)
            if alias == selected_model:
                selected_file = fn
                break

        if selected_file and selected_file != st.session_state.get('input_file_basename'):
            switch_model(selected_file, all_models_data)
            st.rerun()

    # st.sidebar.subheader("Output CSV File")
    # output_file = st.sidebar.text_input("Output CSV File", value="annotated_results.csv", label_visibility="collapsed")

    # if os.path.exists(output_file):
    #     with open(output_file, "rb") as file:
    #         st.sidebar.download_button(
    #             label="📥 Download Annotated CSV",
    #             data=file,
    #             file_name=output_file,
    #             mime="text/csv"
    #         )

    if current_results:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Navigation")
        options = [f"{i + 1} | {res.get('name', 'Unknown')}" for i, res in enumerate(current_results)]

        selected_source = st.sidebar.selectbox(
            "Skip to source according to name, number:",
            options=options,
            index=st.session_state.current_index if st.session_state.current_index < len(options) else 0
        )

        if selected_source:
            selected_idx = int(selected_source.split(" | ")[0]) - 1
            if selected_idx != st.session_state.current_index:
                st.session_state.current_index = selected_idx
                st.rerun()

    # st.sidebar.markdown("---")
    # st.sidebar.subheader("Keyword Taxonomy")

    # Acts as a clickable button that reveals the search and list below it
    # with st.sidebar.expander("Search & View All Keywords", expanded=False):
    #     if st.session_state.keywords:
    #         # 1. Add the search text input
    #         search_query = st.text_input("Search keywords...", key="kw_search").lower()

    #         # 2. Handle the search logic
    #         if search_query:
    #             # Filter keywords where the name contains the search string (case-insensitive)
    #             filtered_kws = [
    #                 kw for kw in st.session_state.keywords
    #                 if search_query in getattr(kw, 'name', '').lower()
    #             ]

    #             if filtered_kws:
    #                 st.markdown(f"**Found {len(filtered_kws)} matches:**")
    #                 # Display matches as a flat list for easy reading
    #                 for kw in filtered_kws:
    #                     name = getattr(kw, 'name', 'Unknown')
    #                     kw_id = getattr(kw, 'id', 'N/A')
    #                     st.markdown(f"- {name} (ID: {kw_id})")
    #             else:
    #                 st.write("No keywords found matching your search.")
    #         else:
    #             # 3. If the search box is empty, show the full formatted tree
    #             if 'formatted_kws_html' not in st.session_state:
    #                   st.session_state.formatted_kws_html = format_keywords(st.session_state.keywords)
    # 
    #             # Using a container with a set height gives it a nice scrollbar
    #             with st.container(height=400):
    #                 st.markdown(formatted_kws)
    #     else:
    #         st.warning("No keywords loaded yet.")

    # Initialize toggle state if it doesn't exist
    if 'show_keywords' not in st.session_state:
        st.session_state.show_keywords = False

    # --- Show pages if flagged ---
    if st.session_state.show_instructions:
        display_instructions(available_models, available_files)
        return

    if st.session_state.show_metrics:
        display_metrics(results_dir=results_dir)
        return

    # Main UI
    st.title("Local Law Under Rome")

    if not current_results:
        st.info("No active model data found. Please select a model to begin.")
        if st.button("Go to Instructions"):
            st.session_state.show_instructions = True
            st.rerun()
        return

    if st.session_state.current_index >= len(current_results):
        st.success("All samples reviewed!")
        return

    # result = st.session_state.results[st.session_state.current_index]
    active_file = st.session_state.input_file_basename
    result = all_models_data[active_file][st.session_state.current_index]

    # Extract text content and language
    language_val = result.get('original_row', {}).get('Language', '')
    source_text_content = result.get('text', '')

    st.sidebar.markdown("---")
    with st.sidebar.expander("**📖 Source Text**", expanded=False):
        display_source_text(source_text_content, language_val)

    # Handle error or missing data fields
    if "error" in result:
        st.error(f"Sample {result.get('source_id')} had error: {result['error']}")
        if st.button("Skip"):
            st.session_state.current_index += 1
            st.rerun()
        return

    # --- Prepare common data ---
    active_file_clean = active_file.replace(".json", "")
    current_id = f"{active_file_clean}_idx_{st.session_state.current_index}"
    original_row = result.get('original_row', {})

    if 'kw_map' not in st.session_state or 'field_map' not in st.session_state:
        st.session_state.kw_map, st.session_state.field_map = build_taxonomy_maps(
            st.session_state.keywords, st.session_state.fields
        )
    kw_map = st.session_state.kw_map
    field_map = st.session_state.field_map

    st.write("#### Source Text")
    # Create two columns: The first is 1 part wide, the second is 2 parts wide
    col1, col2 = st.columns([1.5, 1.2])
    with col1:
        st.info(f"Group: {result.get('group')} | Name: {result.get('name')}")
    with col2:
        c1, c2 = st.columns([1.15, 1])
        with c2:
            st.markdown("<div style='margin-top: 14px;'></div>", unsafe_allow_html=True)
            if st.button(":material/refresh: load annotated data"):
                active = st.session_state.get('input_file_basename', '')
                sheet_name = active.replace('.json', '')
                if sheet_name.startswith('merged_'):
                    sheet_name = sheet_name[len('merged_'):]
                # determine whether to load from sheet_name or from f'w_en_{sheet_name}'
                # if the English translation appears in LUR_annotations
                origin_filename = result.get("origin_file", "")
                is_w_en = "w_en" in origin_filename

                if is_w_en and not sheet_name.startswith("w_en"):
                    sheet_name = f"w_en_{sheet_name}"
                success = load_annotated_data_from_sheet(sheet_name,
                                                         current_id,
                                                         result,
                                                         original_row)
                if success:
                    st.rerun()

    language_val = result.get('original_row', {}).get('Language', '')
    display_source_text(result.get('text', ''), language_val)

    st.markdown("<br>", unsafe_allow_html=True)

    is_non_analyzed = str(original_row.get('Analyzed [y/n]', 'y')).strip().lower() == 'n'

    if is_non_analyzed:
        field_kept_ids, field_fn_ids, user_defined_topics = render_fields_review_non_analyzed(
            result, original_row, field_map, st.session_state.fields, current_id)

        kw_kept_ids, final_new_kws, kw_fn_ids, user_defined_keywords = render_keywords_review_non_analyzed(
            result, original_row, kw_map, st.session_state.keywords, current_id)

        index_kept_terms, index_fn_terms, user_defined_index_terms = render_index_review_non_analyzed(
            result, original_row, current_id)

        dup_keywords = []
        kw_final_kept = kw_kept_ids
        f_final_kept = field_kept_ids
        f_miss = field_fn_ids
        i_final_kept = index_kept_terms
        i_miss = index_fn_terms
    else:
        field_kept_ids, field_miss_agreed_ids, f_intersection = render_fields_review(
            result, original_row, field_map, current_id)

        intersection_ids, kept_suggestion_ids, agreed_missed_ids, final_new_kws, dup_keywords = render_keywords_review(
            result, original_row, kw_map, current_id)

        index_kept_terms, index_miss_agreed_terms, i_intersection = render_index_review(
            result, original_row, current_id)

        kw_fn_ids = []
        user_defined_topics = []
        user_defined_keywords = []
        user_defined_index_terms = []
        kw_final_kept = intersection_ids + kept_suggestion_ids + agreed_missed_ids
        f_final_kept = field_kept_ids + f_intersection
        f_miss = field_miss_agreed_ids
        i_final_kept = index_kept_terms + i_intersection
        i_miss = index_miss_agreed_terms

    st.markdown("---")
    st.markdown("<p style='font-size: 16px; margin-bottom: 0px;'>Comments</p>", unsafe_allow_html=True)
    com_key = f"comments_{current_id}"
    if com_key not in st.session_state: st.session_state[com_key] = ""
    annotator_comments = st.text_area("Comments:", key=com_key, label_visibility="collapsed")

    # Display progress
    st.write(f"Progress: {st.session_state.current_index + 1} / {len(current_results)}")

    filename = st.session_state.get('input_file_basename', 'unknown.json')

    col_prev, col_skip, col_next, col_save = st.columns([0.15, 0.12, 0.15, 0.58])
    with col_prev:
        if st.button("⬅ Previous", disabled=(st.session_state.current_index == 0)):
            if (st.session_state.annotations
                    and st.session_state.current_index - 1 not in st.session_state.skipped_indices):
                st.session_state.annotations.pop()
            st.session_state.current_index -= 1
            st.rerun()

    with col_skip:
        if st.button("⏭ Skip"):
            st.session_state.skipped_indices.add(st.session_state.current_index)
            st.session_state.current_index += 1
            st.rerun()

    with col_next:
        if st.button("Next Sample"):
            add_anno(result, filename,
                     kw_final_kept, final_new_kws,
                     f_final_kept, f_miss,
                     i_final_kept, i_miss,
                     annotator_comments, dup_keywords, next_i=True,
                     kw_fn_ids=kw_fn_ids,
                     user_defined_topics=user_defined_topics,
                     user_defined_keywords=user_defined_keywords,
                     user_defined_index_terms=user_defined_index_terms,
                     is_non_analyzed=is_non_analyzed)
            st.rerun()

    with col_save:
        if st.button("Save", type="primary"):
            add_anno(result, filename,
                     kw_final_kept, final_new_kws,
                     f_final_kept, f_miss,
                     i_final_kept, i_miss,
                     annotator_comments, dup_keywords, next_i=False,
                     kw_fn_ids=kw_fn_ids,
                     user_defined_topics=user_defined_topics,
                     user_defined_keywords=user_defined_keywords,
                     user_defined_index_terms=user_defined_index_terms,
                     is_non_analyzed=is_non_analyzed)
            save_results()
            st.rerun()


def save_results():
    if not st.session_state.annotations:
        st.warning("No new annotations to save.")
        return

    # --- Prepare Data ---
    export_data = []
    kw_map = {str(k.id).strip().lower(): k.full_path for k in st.session_state.keywords}

    for ann in st.session_state.annotations:
        row = ann.copy()

        def add_vector_metrics(row, initial_gold, pred, updated_gold, prefix, ignore_list=None):
            if ignore_list is None: ignore_list = []
            initial_gold = [g for g in initial_gold if str(g).strip().lower() not in ignore_list]
            pred = [g for g in pred if str(g).strip().lower() not in ignore_list]
            updated_gold = [g for g in updated_gold if str(g).strip().lower() not in ignore_list]
            op, or_, oj = compute_sample_metrics(initial_gold, pred)
            mp, mr, mj = compute_sample_metrics(updated_gold, pred)
            row[f'{prefix}_orig_p'] = round(op, 4)
            row[f'{prefix}_orig_r'] = round(or_, 4)
            row[f'{prefix}_orig_j'] = round(oj, 4)
            row[f'{prefix}_mod_p'] = round(mp, 4)
            row[f'{prefix}_mod_r'] = round(mr, 4)
            row[f'{prefix}_mod_j'] = round(mj, 4)

        ignore_kws = [str(k).strip().lower() for k in ann.get('dup_keywords', [])]

        if ann.get('is_non_analyzed'):
            # No gold standard — annotator's choices become the gold.
            # kw_fn_ids / field_miss_agreed_ids / index_miss_agreed_terms hold the FN
            # items the annotator manually added; kept + FN = the new annotation gold.
            kw_gold = ann['kw_kept_ids'] + ann.get('kw_fn_ids', [])
            f_gold = ann['field_kept_ids'] + ann['field_miss_agreed_ids']
            i_gold = ann['index_kept_terms'] + ann['index_miss_agreed_terms']
            add_vector_metrics(row, kw_gold, ann['orig_kw_ids'], kw_gold, 'kw', ignore_list=ignore_kws)
            add_vector_metrics(row, f_gold, ann['orig_field_ids'], f_gold, 'field')
            add_vector_metrics(row, i_gold, ann['orig_index_terms'], i_gold, 'index')
        else:
            # Analyzed: orig_* = model vs initial gold; mod_* = model vs annotator-updated gold
            add_vector_metrics(row, ann['gold_kw_ids'], ann['orig_kw_ids'], ann['kw_kept_ids'], 'kw',
                               ignore_list=ignore_kws)
            f_updated_gold = ann['field_kept_ids'] + ann['field_miss_agreed_ids']
            add_vector_metrics(row, ann['gold_field_ids'], ann['orig_field_ids'], f_updated_gold, 'field')
            i_updated_gold = ann['index_kept_terms'] + ann['index_miss_agreed_terms']
            add_vector_metrics(row, ann['gold_index_terms'], ann['orig_index_terms'], i_updated_gold, 'index')

        # Convert lists to strings
        for key, val in row.items():
            if isinstance(val, list):
                if key == 'dup_keywords':
                    vals = [kw_map.get(str(v).strip().lower(), str(v)) for v in val]
                    row[key] = ", ".join(vals)
                else:
                    row[key] = ", ".join([str(v) for v in val])

        export_data.append(row)

    # --- Split analyzed vs non-analyzed ---
    analyzed_rows = [r for r in export_data if not r.get('is_non_analyzed')]
    non_analyzed_rows = [r for r in export_data if r.get('is_non_analyzed')]

    def _derive_sheet_name(results_filename):
        active = st.session_state.get('input_file_basename', '')
        sheet_name = active.removeprefix("merged_").removesuffix(".json")
        if sheet_name.startswith('merged_'):
            sheet_name = sheet_name[len('merged_'):]
        if "w_en" in results_filename and not sheet_name.startswith("w_en"):
            sheet_name = "w_en_" + sheet_name
        return sheet_name

    def _write_rows_to_sheet(rows, sheet_name):
        new_df = pd.DataFrame(rows)
        with st.spinner(f'Syncing with Google Sheets ({sheet_name})...'):
            try:
                if 'conn' not in st.session_state:
                    st.session_state.conn = st.connection("gsheets", type=GSheetsConnection)

                ensure_worksheet_exists(st.session_state.conn, DEFAULT_SHEET_URL, sheet_name)

                try:
                    existing_df = st.session_state.conn.read(worksheet=sheet_name, ttl=0)
                    if existing_df is None:
                        existing_df = pd.DataFrame()
                except Exception:
                    existing_df = pd.DataFrame()

                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                st.session_state.conn.update(worksheet=sheet_name, data=combined_df)

                if 'sheet_cache' in st.session_state and sheet_name in st.session_state.sheet_cache:
                    del st.session_state.sheet_cache[sheet_name]

                st.success(f"Saved to sheet '{sheet_name}' successfully!")
            except Exception as e:
                st.error(f"Google Sheet Error ({sheet_name}): {e}")

    active = st.session_state.get('input_file_basename', '')

    if analyzed_rows:
        sheet_name = _derive_sheet_name(analyzed_rows[-1].get('results_filename', active))
        _write_rows_to_sheet(analyzed_rows, sheet_name)

    if non_analyzed_rows:
        base_sheet = _derive_sheet_name(non_analyzed_rows[-1].get('results_filename', active))
        na_sheet = base_sheet if base_sheet.startswith("non_analyzed_") else f"non_analyzed_{base_sheet}"
        _write_rows_to_sheet(non_analyzed_rows, na_sheet)

    st.session_state.annotations = []


if __name__ == "__main__":
    main()
