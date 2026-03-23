import os
import sys

# Add the project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

import streamlit as st
import pandas as pd
import json
import yaml
from datetime import date
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from classifier import format_keywords
from streamlit_gsheets import GSheetsConnection

from data_loader import DataLoader
from keyword_manager import KeywordManager

DEFAULT_SHEET_URL = "https://docs.google.com/spreadsheets/d/1cb4Pmc7SFCZ3C5kJD8kkDFQsuJXdk16a1afoRElJ3L0/edit?gid=0#gid=0"


# Function to parse arguments
def get_config(results_dir):
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
    project_root = os.path.dirname(results_dir)
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
                      kw_kept_ids, kw_man_ids, kw_new_accepted,
                      field_kept_ids, field_miss_ids, 
                      index_kept_terms, index_miss_terms):
    """Creates the annotation dictionary for 3-Vector Review."""
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
        "results_filename": filename,
        "annotator": st.session_state.get('name', ''),
        "date": date.today().isoformat(),
        "ref_id": original_row.get("Refference") or original_row.get("ref Code"),
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
        "gold_index_terms": gold_index_terms
    }


def add_anno(result, filename, 
             kw_kept_ids,
            #  kw_man_ids,
             kw_new_accepted,
             field_kept_ids, field_miss_ids, 
             index_kept_terms, index_miss_terms):
    """Adds the 3-vector annotation to the session state."""
    annotation = create_annotation(
        result, filename, 
        kw_kept_ids,
        # kw_man_ids,
        kw_new_accepted,
        field_kept_ids, field_miss_ids, 
        index_kept_terms, index_miss_terms
    )

    # Add to the session buffer
    st.session_state.annotations.append(annotation)

    # Update keyword manager (for new suggested keywords that were accepted)
    if kw_new_accepted:
        st.session_state.keyword_manager.update_keywords(kw_new_accepted)

    # Increment index
    st.session_state.current_index += 1


def load_data(input_file):
    # Initialize keys if they don't exist yet
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0

    # Only attempt to load if we have a valid path and it's a new file
    if input_file and os.path.exists(input_file):
        if st.session_state.get('input_file') != input_file or not st.session_state.get('keywords') or not st.session_state.get('fields'):
            try:
                loader = DataLoader()
                # Always ensure keywords are loaded
                st.session_state.keywords = loader.load_keywords(st.session_state.keywords_file)
                
                # Ensure fields are loaded if path exists
                if hasattr(st.session_state, 'fields_file') and st.session_state.fields_file:
                    st.session_state.fields = loader.load_judicial_fields(st.session_state.fields_file)
                else:
                    st.session_state.fields = []

                with open(input_file, 'r') as f:
                    st.session_state.results = json.load(f)

                st.session_state.keyword_manager = KeywordManager()
                st.session_state.annotations = []
                st.session_state.current_index = 0
                st.session_state.input_file = input_file
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(st.session_state.results)} samples.")
                st.rerun()  # Refresh to update the UI with new data
            except Exception as e:
                st.error(f"Error loading files: {e}")


def display_rtl_text(text_content):
    st.markdown(
        f"""
        <div style="
            direction: rtl; 
            text-align: right; 
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
    gold = set(str(g).strip() for g in gold_ids if str(g).strip())
    pred = set(str(p).strip() for p in pred_ids if str(p).strip())
    tp = len(gold & pred)
    precision = tp / len(pred) if pred else 0.0
    recall = tp / len(gold) if gold else 0.0
    union = gold | pred
    jaccard = len(gold & pred) / len(union) if union else 0.0
    return precision, recall, jaccard


def display_metrics(output_file, results_dir=None):
    """Renders the aggregated metrics dashboard page."""
    st.title("📊 Aggregated Metrics Dashboard")
    
    if st.button("⬅ Back to Annotation"):
        st.session_state.show_metrics = False
        st.rerun()

    # --- Section: LLM Performance Summary (Original Predictions) ---
    st.markdown("### LLM Performance Summary (Original Predictions)")
    
    vector = st.radio("Select Vector for Metrics:", ["Keywords", "Judicial Fields", "Index Terms"], horizontal=True)
    v_prefix = {"Keywords": "kw", "Judicial Fields": "field", "Index Terms": "index"}[vector]
    gold_key = {"Keywords": "KW Ids", "Judicial Fields": "Judicial Topic Ids", "Index Terms": "Index Terms"}[vector]
    pred_key = {"Keywords": "matched_ids", "Judicial Fields": "matched_field_ids", "Index Terms": "index_terms"}[vector]

    if results_dir and os.path.exists(results_dir):
        json_files = sorted([f for f in os.listdir(results_dir) if f.endswith('.json')])
        summary_rows = []
        
        for jf in json_files:
            try:
                with open(os.path.join(results_dir, jf), 'r') as f:
                    data = json.load(f)
                
                precisions, recalls, jaccards = [], [], []
                
                for item in data:
                    if "error" in item: continue
                    gold_raw = item.get('original_row', {}).get(gold_key, '')
                    if not gold_raw or str(gold_raw).lower() == 'nan': continue
                    
                    gold = [g.strip() for g in str(gold_raw).split(',') if g.strip()]
                    pred = item.get(pred_key, [])
                    
                    p, r, j = compute_sample_metrics(gold, pred)
                    precisions.append(p)
                    recalls.append(r)
                    jaccards.append(j)
                
                if precisions:
                    df = pd.DataFrame({'p': precisions, 'r': recalls, 'j': jaccards})
                    summary_rows.append({
                        "File": jf,
                        "Samples": len(data),
                        "Gold Count": len(precisions),
                        "Avg Prec": df['p'].mean(),
                        "Avg Rec": df['r'].mean(),
                        "Avg Jac": df['j'].mean()
                    })
            except Exception as e:
                st.error(f"Error processing {jf}: {e}")

        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows).style.format(precision=3), hide_index=True)
        else:
            st.info(f"No gold standard data found for {vector} in result files.")
    
    st.markdown("---")
    st.markdown("### ✍️ Annotation-Based Metrics")

    if not os.path.exists(output_file):
        st.warning("No annotated results file found yet.")
        return
        
    try:
        results_df = pd.read_csv(output_file)
        if not results_df.empty:
            st.markdown(f"#### Dataset Level Averages ({vector})")
            m_cols = [f'{v_prefix}_orig_j', f'{v_prefix}_mod_j', f'{v_prefix}_orig_p', f'{v_prefix}_mod_p', f'{v_prefix}_orig_r', f'{v_prefix}_mod_r']
            existing_m = [c for c in m_cols if c in results_df.columns]
            
            if existing_m:
                dataset_level = results_df[existing_m].mean()
                c1, c2, c3 = st.columns(3)
                c1.metric("Jaccard (Orig → Mod)", f"{dataset_level.get(f'{v_prefix}_orig_j', 0):.3f} → {dataset_level.get(f'{v_prefix}_mod_j', 0):.3f}")
                c2.metric("Precision (Orig → Mod)", f"{dataset_level.get(f'{v_prefix}_orig_p', 0):.3f} → {dataset_level.get(f'{v_prefix}_mod_p', 0):.3f}")
                c3.metric("Recall (Orig → Mod)", f"{dataset_level.get(f'{v_prefix}_orig_r', 0):.3f} → {dataset_level.get(f'{v_prefix}_mod_r', 0):.3f}")

            st.markdown("#### Sample Level Details")
            display_cols = ['ref_id', 'group', 'name'] + existing_m
            avail_cols = [c for c in display_cols if c in results_df.columns]
            st.dataframe(results_df[avail_cols].tail(20), hide_index=True)
    except Exception as e:
        st.error(f"Error loading metrics from {output_file}: {e}")



def display_instructions():
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

    if st.button("▶ Begin Annotation", type="primary"):
        st.session_state.show_instructions = False
        st.rerun()


def main():
    # Fix: Resolve results_dir relative to the script location
    # This ensures it works whether running from root or src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")

    st.set_page_config(layout="centered",
                       page_title="RomanJewish Legal Classifier - Review")


    # --- AUTHENTICATION ---
    # Read credentials from Streamlit secrets (cloud) or config.yaml (local dev)
    try:
        credentials = dict(st.secrets["auth_credentials"])
        # Convert nested AttrDict to plain dicts for streamlit-authenticator
        credentials["usernames"] = {
            user: dict(data) for user, data in st.secrets["auth_credentials"]["usernames"].items()
        }
        cookie_config = dict(st.secrets["auth_cookie"])
    except (KeyError, FileNotFoundError):
        # Fallback to local config.yaml
        config_path = os.path.join(project_root, "config.yaml")
        with open(config_path) as f:
            config = yaml.load(f, Loader=SafeLoader)
        credentials = config['credentials']
        cookie_config = config['cookie']

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

    if st.session_state.get('authentication_status') is False:
        st.error('Username/password is incorrect')
        return
    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')
        return

    # --- User is authenticated from here ---

    # --- INITIALIZE SESSION STATE KEYS ---
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
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

    cli_input_file, cli_keywords_file, cli_fields_file, available_files = get_config(results_dir)
    st.session_state.keywords_file = cli_keywords_file
    st.session_state.fields_file = cli_fields_file

    # Select menu for results file
    if available_files:
        selected_file = st.sidebar.selectbox(
            "Select Results JSON File",
            options=available_files,
            index=None
        )

        if selected_file:
            input_file = os.path.join(results_dir, selected_file)
        else:
            input_file = None
    else:
        # Fallback text input if no files found or custom path needed
        input_file = st.sidebar.text_input("Results JSON File Path", value=cli_input_file if cli_input_file else "")

    # st.session_state.keywords_file = st.sidebar.text_input("Keywords CSV File", value=cli_keywords_file)

    output_file = st.sidebar.text_input("Output CSV File", value="annotated_results.csv")

    if os.path.exists(output_file):
        with open(output_file, "rb") as file:
            st.sidebar.download_button(
                label="📥 Download Annotated CSV",
                data=file,
                file_name=output_file,
                mime="text/csv"
            )
    # Load Data
    load_data(input_file)

    # ... inside main(), after load_data(input_file) ...

    st.sidebar.markdown("---")
    st.sidebar.subheader("Keyword Taxonomy")

    # Acts as a clickable button that reveals the search and list below it
    with st.sidebar.expander("Search & View All Keywords", expanded=False):
        if st.session_state.keywords:
            # 1. Add the search text input
            search_query = st.text_input("Search keywords...", key="kw_search").lower()

            # 2. Handle the search logic
            if search_query:
                # Filter keywords where the name contains the search string (case-insensitive)
                filtered_kws = [
                    kw for kw in st.session_state.keywords
                    if search_query in getattr(kw, 'name', '').lower()
                ]

                if filtered_kws:
                    st.markdown(f"**Found {len(filtered_kws)} matches:**")
                    # Display matches as a flat list for easy reading
                    for kw in filtered_kws:
                        name = getattr(kw, 'name', 'Unknown')
                        kw_id = getattr(kw, 'id', 'N/A')
                        st.markdown(f"- {name} (ID: {kw_id})")
                else:
                    st.write("No keywords found matching your search.")
            else:
                # 3. If the search box is empty, show the full formatted tree
                formatted_kws = format_keywords(st.session_state.keywords)
                # Using a container with a set height gives it a nice scrollbar
                with st.container(height=400):
                    st.markdown(formatted_kws)
        else:
            st.warning("No keywords loaded yet.")

    # Initialize toggle state if it doesn't exist
    if 'show_keywords' not in st.session_state:
        st.session_state.show_keywords = False

    # --- Show pages if flagged ---
    if st.session_state.show_instructions:
        display_instructions()
        return

    if st.session_state.show_metrics:
        display_metrics(output_file, results_dir=results_dir)
        return

    # Main UI
    st.title("Local Law Under Rome")

    if not st.session_state.results:
        st.info("Please select a results JSON file from the sidebar to begin.")
        return

    if st.session_state.current_index >= len(st.session_state.results):
        st.success("All samples reviewed!")
        return

    result = st.session_state.results[st.session_state.current_index]

    # Handle error or missing data fields
    if "error" in result:
        st.error(f"Sample {result.get('source_id')} had error: {result['error']}")
        if st.button("Skip"):
            st.session_state.current_index += 1
            st.rerun()
        return

    st.write("#### Source Text")
    # Create two columns: The first is 1 part wide, the second is 2 parts wide
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.info(f"Group: {result.get('group')} | Name: {result.get('name')}")

    display_rtl_text(result.get('text', ''))

    # --- Prepare keyword data ---
    matched_ids = result.get('matched_ids', [])
    matched_names = result.get('matched_keywords', [])
    suggested_kws = result.get('suggested_kws', [])
    current_id = f"sample_{st.session_state.current_index}"
    kw_map = {str(k.id).strip().lower(): k for k in st.session_state.keywords}

    original_row = result.get('original_row', {})
    gold_kw_ids_raw = original_row.get('KW Ids', '')
    gold_kw_names_raw = original_row.get('Keywords', '')
    has_gold_kw = (gold_kw_ids_raw
                   and str(gold_kw_ids_raw).strip()
                   and str(gold_kw_ids_raw).lower() != 'nan')
    gold_ids_list = []
    if has_gold_kw:
        gold_ids_list = [g.strip() for g in str(gold_kw_ids_raw).split(',') if g.strip()]

    # Split matched into Intersection and Suggestions
    pred_set = set(str(mid).strip().lower() for mid in matched_ids)
    gold_set = set(str(gid).strip().lower() for gid in gold_ids_list)
    intersection_ids = sorted(list(pred_set & gold_set))
    suggestion_ids = sorted(list(pred_set - gold_set))
    missed_ids = sorted(list(gold_set - pred_set))

    
    # Widen the centered content area for better annotation columns
    st.markdown("""
        <style>
            .block-container {
                max-width: 1200px;
                padding-left: 2rem;
                padding-right: 2rem;
            }
        </style>
    """, unsafe_allow_html=True)

    
    # --- KEYWORDS SECTION ---
    st.subheader("Keywords Review")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Intersection** (Read-only)")
        if intersection_ids:
            for mid in intersection_ids:
                kw_obj = kw_map.get(str(mid).strip().lower())
                label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"
                st.info(f"✅ {label}")
        else:
            st.caption("No intersection found.")

    with col2:
        st.write("**Suggestions** (Accept/Reject)")
        kept_suggestion_ids = []
        if suggestion_ids:
            for mid in suggestion_ids:
                kw_obj = kw_map.get(str(mid).strip().lower())
                label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"
                c_label, c_acc = st.columns([0.9, 0.1])
                with c_label:
                    st.text_input("Label", value=label, key=f"kw_sug_lbl_{current_id}_{mid}", label_visibility="collapsed", read_only=True, help=label)
                with c_acc:
                    if st.checkbox("", value=False, key=f"kw_sug_{current_id}_{mid}"):
                        kept_suggestion_ids.append(mid)
        else:
            st.caption("No suggestions.")

    with col3:
        st.write("**New Suggestions** (Accept/Edit)")
        final_new_kws = []
        if suggested_kws:
            for i, skw in enumerate(suggested_kws):
                c_edit, c_acc = st.columns([0.9, 0.1])
                with c_edit:
                    edited_kw = st.text_input("Edit", value=skw, key=f"kw_new_edit_{current_id}_{i}", label_visibility="collapsed", help=skw)
                with c_acc:
                    if st.checkbox("", value=False, key=f"kw_new_acc_{current_id}_{i}"):
                        final_new_kws.append(edited_kw)
        else:
            st.caption("No new suggestions.")

    # Missed Keywords section
    st.write("**Missed Gold Keywords** (Agree/Disagree)")
    agreed_missed_ids = []
    if missed_ids:
        for mid in missed_ids:
            kw_obj = kw_map.get(str(mid).strip().lower())
            label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"
            if st.checkbox(label, value=True, key=f"kw_miss_{current_id}_{mid}"):
                agreed_missed_ids.append(mid)
    else:
        st.caption("No missed gold keywords.")

    # Manual Add (Thesaurus)
    # st.write("**Add other keywords from thesaurus?**")
    # all_kw_names = [f"{k.name} (ID: {k.id})" for k in st.session_state.keywords]
    # manually_added_kws = st.multiselect("Select keywords:", all_kw_names, key=f"kw_man_{current_id}", label_visibility="collapsed")

    # st.markdown("---")

    # Add manual new keyword?
    # st.write("**Define any missing keywords, separated by commas (optional)**")
    # manual_new = st.text_input("-", key=f"manual_{current_id}", label_visibility="collapsed")
    # manual_new_list = [m.strip() for m in manual_new.split(",") if m.strip()]
    # final_new_kws += manual_new_list

    # st.markdown("---")

    # --- JUDICIAL FIELDS SECTION ---
    st.subheader("Judicial Fields Review")

    field_map = {str(f.id).strip().lower(): f for f in st.session_state.get('fields', [])}
    matched_field_ids = result.get('matched_field_ids', [])
    gold_field_ids_raw = original_row.get('Judicial Topic Ids', '')
    has_gold_f = (gold_field_ids_raw and str(gold_field_ids_raw).strip() and str(gold_field_ids_raw).lower() != 'nan')
    gold_f_set = set([g.strip().lower() for g in str(gold_field_ids_raw).split(',') if g.strip()]) if has_gold_f else set()
    pred_f_set = set(str(fid).strip().lower() for fid in matched_field_ids)

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
        st.write("**Suggestions** (Accept/Reject)")
        field_kept_ids = []
        if f_suggestions:
            for fid in f_suggestions:
                f_obj = field_map.get(str(fid).strip().lower())
                label = f_obj.full_path if f_obj else f"Unknown ID: {fid}"
                c_label, c_acc = st.columns([0.9, 0.1])
                with c_label:
                    st.text_input("Label", value=label, key=f"f_sug_lbl_{current_id}_{fid}", label_visibility="collapsed", read_only=True, help=label)
                with c_acc:
                    if st.checkbox("", value=False, key=f"f_sug_{current_id}_{fid}"):
                        field_kept_ids.append(fid)
        else:
            field_kept_ids = []
            st.caption("No suggestions.")

    st.write("**Missed Gold Fields** (Agree/Disagree)")
    field_miss_agreed_ids = []
    if f_missed:
        for fid in f_missed:
            f_obj = field_map.get(str(fid).strip().lower())
            label = f_obj.full_path if f_obj else f"Unknown ID: {fid}"
            if st.checkbox(label, value=True, key=f"f_miss_{current_id}_{fid}"):
                field_miss_agreed_ids.append(fid)
    else:
        st.caption("No missed gold fields.")

    st.markdown("---")

    # --- INDEX TERMS SECTION ---
    st.subheader("Index Terms Review")

    pred_index = result.get('index_terms', [])
    gold_index_raw = original_row.get('Index Terms', '')
    has_gold_i = (gold_index_raw and str(gold_index_raw).strip() and str(gold_index_raw).lower() != 'nan')
    gold_i_list = [g.strip() for g in str(gold_index_raw).split(',') if g.strip()] if has_gold_i else []

    pred_i_set = set(str(p).strip().lower() for p in pred_index)
    gold_i_set = set(str(g).strip().lower() for g in gold_i_list)

    i_intersection = sorted(list(pred_i_set & gold_i_set))
    i_suggestions = sorted(list(pred_i_set - gold_i_set))
    i_missed = sorted(list(gold_i_set - pred_i_set))

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.write("**Intersection** (Read-only)")
        if i_intersection:
            for term in i_intersection:
                st.info(f"✅ {term}")
        else:
            st.caption("No intersection.")

    with col_i2:
        st.write("**Suggestions** (Accept/Reject)")
        index_kept_terms = []
        if i_suggestions:
            for term in i_suggestions:
                c_label, c_acc = st.columns([0.9, 0.1])
                with c_label:
                    st.text_input("Label", value=term, key=f"i_sug_lbl_{current_id}_{term}", label_visibility="collapsed", read_only=True, help=term)
                with c_acc:
                    if st.checkbox("", value=False, key=f"i_sug_{current_id}_{term}"):
                        index_kept_terms.append(term)
        else:
            index_kept_terms = []
            st.caption("No suggestions.")

    st.write("**Missed Gold Index Terms** (Agree/Disagree)")
    index_miss_agreed_terms = []
    if i_missed:
        for term in i_missed:
            if st.checkbox(term, value=True, key=f"i_miss_{current_id}_{term}"):
                index_miss_agreed_terms.append(term)
    else:
        st.caption("No missed gold index terms.")

    st.markdown("---")

    # Display progress
    st.write(f"Progress: {st.session_state.current_index + 1} / {len(st.session_state.results)}")

    # Combine all annotation vectors
    # kw_manually_added_ids = [k.split("(ID: ")[1].strip(")") for k in manually_added_kws]
    kw_final_kept = intersection_ids + kept_suggestion_ids + agreed_missed_ids

    filename = os.path.basename(st.session_state.input_file) if st.session_state.get('input_file') else "unknown.json"

    col_b1, col_b2 = st.columns([0.25, 1.])
    with col_b1:
        if st.button("Next Sample"):
            add_anno(result, filename,
                    # kw_manually_added_ids,
                     kw_final_kept, final_new_kws,
                     field_kept_ids + f_intersection, field_miss_agreed_ids,
                     index_kept_terms + i_intersection, index_miss_agreed_terms)
            st.rerun()

    with col_b2:
        if st.button("Save Annotated Results", type="primary"):
            add_anno(result, filename,
                    #  kw_manually_added_ids,
                     kw_final_kept, final_new_kws,
                     field_kept_ids + f_intersection, field_miss_agreed_ids,
                     index_kept_terms + i_intersection, index_miss_agreed_terms)
            save_results(output_file)
            st.rerun()


def save_results(filename):
    if not st.session_state.annotations:
        st.warning("No new annotations to save.")
        return

    # --- Prepare Data ---
    export_data = []
    kw_map = {str(k.id): k.full_path for k in st.session_state.keywords}
    field_map = {str(f.id): f.full_path for f in st.session_state.get('fields', [])}

    for ann in st.session_state.annotations:
        row = ann.copy()
        
        # Helper to compute and add metrics
        def add_vector_metrics(row, gold, original, modified, prefix):
            op, or_, oj = compute_sample_metrics(gold, original)
            mp, mr, mj = compute_sample_metrics(gold, modified)
            row[f'{prefix}_orig_p'] = round(op, 4)
            row[f'{prefix}_orig_r'] = round(or_, 4)
            row[f'{prefix}_orig_j'] = round(oj, 4)
            row[f'{prefix}_mod_p'] = round(mp, 4)
            row[f'{prefix}_mod_r'] = round(mr, 4)
            row[f'{prefix}_mod_j'] = round(mj, 4)

        # Keywords Metrics
        # kw_mod = ann['kw_kept_ids'] + ann['kw_manually_added_ids']
        kw_mod = ann['kw_kept_ids']
        add_vector_metrics(row, ann['gold_kw_ids'], ann['orig_kw_ids'], kw_mod, 'kw')
        
        # Fields Metrics
        f_mod = ann['field_kept_ids'] + ann['field_miss_agreed_ids']
        add_vector_metrics(row, ann['gold_field_ids'], ann['orig_field_ids'], f_mod, 'field')
        
        # Index Metrics
        i_mod = ann['index_kept_terms'] + ann['index_miss_agreed_terms']
        add_vector_metrics(row, ann['gold_index_terms'], ann['orig_index_terms'], i_mod, 'index')

        # Convert lists to strings for CSV
        for key, val in row.items():
            if isinstance(val, list):
                row[key] = ", ".join([str(v) for v in val])

        export_data.append(row)

    new_df = pd.DataFrame(export_data)
    
    try:
        if os.path.exists(filename):
            try:
                existing_df = pd.read_csv(filename, on_bad_lines='warn')
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                combined_df = combined_df.fillna('')
                combined_df.to_csv(filename, index=False)
            except Exception as e:
                st.error(f"Error reading existing CSV: {e}. Appending instead.")
                new_df.to_csv(filename, mode='a', header=False, index=False)
        else:
            new_df.to_csv(filename, index=False)
            
        st.toast(f"Saved locally to {filename}")
    except Exception as e:
        st.error(f"Failed to save local CSV: {e}")

    # --- Google Sheets Read -> Append -> Update ---
    # Use the results filename (without .json) as the worksheet name
    sheet_name = st.session_state.annotations[0]['results_filename'].replace('.json', '') \
        if st.session_state.annotations else new_df.iloc[0]['results_filename'].replace('.json', '')

    with st.spinner('Syncing with Google Sheets...'):
        try:
            if 'conn' not in st.session_state:
                st.session_state.conn = st.connection("gsheets", type=GSheetsConnection)

            # 1. Read existing data to prevent overwriting
            try:
                # ttl=0 ensures we don't get a cached version of the sheet
                existing_df = st.session_state.conn.read(worksheet=sheet_name, ttl=0)
                # Ensure we are working with a DataFrame
                if existing_df is None:
                    existing_df = pd.DataFrame()
            except Exception:
                # If sheet is empty or doesn't exist yet
                existing_df = pd.DataFrame()

            # 2. Combine Data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # 3. Write back the FULL dataset
            st.session_state.conn.update(worksheet=sheet_name, data=combined_df)

            st.success("Google Sheet updated successfully!")

            # Clear the buffer so we don't save these duplicates again next time
            st.session_state.annotations = []

        except Exception as e:
            st.error(f"Google Sheet Error: {e}")


if __name__ == "__main__":
    main()
