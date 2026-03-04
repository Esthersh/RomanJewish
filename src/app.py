import os
import streamlit as st
import pandas as pd
import sys
import json
import yaml
from datetime import date
from yaml.loader import SafeLoader
import streamlit_authenticator as stauth
from classifier import format_keywords
from streamlit_gsheets import GSheetsConnection

from data_loader import DataLoader
from keyword_manager import KeywordManager


# Function to parse arguments
def get_config(results_dir):
    # Ensure results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # List JSON files in results directory
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    # Default to first file if available, otherwise None
    default_input = json_files[0] if json_files else None

    # Default keywords file
    keywords_file = None
    project_root = os.path.dirname(results_dir)
    kw_path = os.path.join(project_root, "data", "Keywords.csv")
    if os.path.exists(kw_path):
        keywords_file = kw_path

    for i, arg in enumerate(sys.argv):
        if arg == "--input_file" and i + 1 < len(sys.argv):
            default_input = sys.argv[i + 1]
        if arg == "--keywords_file" and i + 1 < len(sys.argv):
            keywords_file = sys.argv[i + 1]

    return default_input, keywords_file, json_files


def create_annotation(result, matched_names, kept_ids, added_kws,
                      suggested_kws, final_new_kws, filename):
    """Creates the annotation dictionary."""
    # Extract gold IDs for metric computation
    original_row = result.get('original_row', {})
    gold_kw_ids_raw = original_row.get('KW Ids', '')
    gold_ids = []
    if gold_kw_ids_raw and str(gold_kw_ids_raw).strip() and str(gold_kw_ids_raw).lower() != 'nan':
        gold_ids = [g.strip() for g in str(gold_kw_ids_raw).split(',') if g.strip()]

    return {
        "results_filename": filename,
        "annotator": st.session_state.get('name', ''),
        "date": date.today().isoformat(),
        "ref_id": result.get('original_row').get("Refference"),
        "source_id": result.get('source_id'),
        "text": result.get("text"),
        "group": result.get("group"),
        "name": result.get("name"),
        "original_matched": matched_names,
        "original_matched_ids": result.get('matched_ids', []),
        "kept_ids": kept_ids,
        "added_existing_ids": [k.split("(ID: ")[1].strip(")") for k in added_kws],
        "gold_ids": gold_ids,
        "original_suggested": suggested_kws,
        "accepted_new_keywords": final_new_kws
    }


def add_anno(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws):
    """Adds the annotation to the session state."""
    if 'input_file' in st.session_state and st.session_state.input_file:
        current_file = os.path.basename(st.session_state.input_file)
    annotation = create_annotation(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws,
                                   current_file)

    # Add to the session buffer
    st.session_state.annotations.append(annotation)

    # Update global list (in-memory)
    if len(final_new_kws) > 0:
        st.session_state.keyword_manager.update_keywords(final_new_kws)

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
        if st.session_state.get('input_file') != input_file:
            try:
                loader = DataLoader()
                st.session_state.keywords = loader.load_keywords(st.session_state.keywords_file)

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
    if 'show_instructions' not in st.session_state:
        st.session_state.show_instructions = True
    # -------------------------------------

    # Sidebar
    st.sidebar.title("Review Config")
    st.sidebar.write(f"Logged in as: **{st.session_state.get('name')}**")
    authenticator.logout('Logout', 'sidebar')

    if st.sidebar.button("📖 Instructions"):
        st.session_state.show_instructions = True
        st.rerun()

    st.sidebar.markdown("---")

    cli_input_file, cli_keywords_file, available_files = get_config(results_dir)
    st.session_state.keywords_file = cli_keywords_file

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
    # else:
        # st.sidebar.info("Annotated CSV will be available to download after your first save.")

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

    # --- Show instructions page if flagged ---
    if st.session_state.show_instructions:
        display_instructions()
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
    col1, col2 = st.columns([0.8, 1])
    with col1:
        st.info(f"Group: {result.get('group')} | Name: {result.get('name')}")

    display_rtl_text(result.get('text', ''))

    # --- Prepare keyword data ---
    matched_ids = result.get('matched_ids', [])
    matched_names = result.get('matched_keywords', [])
    suggested_kws = result.get('suggested_kws', [])
    current_id = f"sample_{st.session_state.current_index}"
    kw_map = {str(k.id): k for k in st.session_state.keywords}

    original_row = result.get('original_row', {})
    gold_kw_ids_raw = original_row.get('KW Ids', '')
    gold_kw_names_raw = original_row.get('Keywords', '')
    has_gold = (gold_kw_ids_raw
                and str(gold_kw_ids_raw).strip()
                and str(gold_kw_ids_raw).lower() != 'nan')
    gold_ids_list = []
    gold_names_list = []
    if has_gold:
        gold_ids_list = [g.strip() for g in str(gold_kw_ids_raw).split(',') if g.strip()]
        gold_names_list = [n.strip() for n in str(gold_kw_names_raw).split(',') if n.strip()]

    # --- Two-Column Layout: Matched vs Gold Keywords ---
    st.subheader("Classification Review")
    col_left, col_right = st.columns(2)

    # Sort helper: alphabetical by full_path, unknowns last
    def _sort_key(kid):
        kw_obj = kw_map.get(str(kid))
        return kw_obj.full_path if kw_obj else f"\uffff{kid}"

    sorted_matched = sorted(matched_ids, key=_sort_key)

    with col_left:
        st.write("**Matched Keywords** — Uncheck if irrelevant")
        kept_ids = []
        for mid in sorted_matched:
            kw_obj = kw_map.get(str(mid))
            label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"
            if st.checkbox(label, value=True, key=f"cb_{current_id}_{mid}"):
                kept_ids.append(mid)

    with col_right:
        st.write("**Gold Annotated Keywords**")
        if has_gold:
            sorted_gold = sorted(
                zip(gold_ids_list, gold_names_list), key=lambda g: _sort_key(g[0])
            )
            for gid, gname in sorted_gold:
                kw_obj = kw_map.get(gid)
                label = kw_obj.full_path if kw_obj else gname
                st.markdown(f"- {label}")
        else:
            st.caption("No gold annotations available.")

    # --- Per-sample Metrics ---
    if has_gold:
        precision, recall, jaccard = compute_sample_metrics(gold_ids_list, matched_ids)
        st.info(
            f"**Original Metrics** — "
            f"Precision: {precision:.2f} · "
            f"Recall: {recall:.2f} · "
            f"Jaccard: {jaccard:.2f}"
        )

    # 2. Add Missed (FN Check)
    st.write("**Are there any missed keywords from the thesaurus?**")
    all_kw_names = [f"{k.name} (ID: {k.id})" for k in st.session_state.keywords]
    added_kws = st.multiselect("Select existing keywords:",
                               all_kw_names,
                               key=f"ms_{current_id}",
                               label_visibility="collapsed")

    final_new_kws = []
    if suggested_kws:
        # 3. New Keyword Suggestions
        st.write("**Suggested Keywords (not from the original list), Edit for correction as needed**")

        for i, skw in enumerate(suggested_kws):
            col_a, col_b = st.columns([0.7, 1])
            with col_a:
                edited_kw = st.text_input(
                    "Edit keyword",
                    value=skw,
                    key=f"new_{current_id}_{i}",
                    label_visibility="collapsed"
                )
            with col_b:
                if st.checkbox("Accept", value=True, key=f"accept_{current_id}_{i}"):
                    final_new_kws.append(edited_kw)
    else:
        st.write("No new keywords suggested by model.")

    # Add manual new keyword?
    st.write("**Define any missing keywords, separated by commas (optional)**")
    manual_new = st.text_input("-",
                               key=f"manual_{current_id}",
                               label_visibility="collapsed")
    if manual_new:
        final_new_kws += [e.strip() for e in manual_new.split(",")]

    # Navigation
    st.markdown("---")
    # Display progress
    st.write(f"Progress: {st.session_state.current_index + 1} / {len(st.session_state.results)}")

    col1, col2 = st.columns([0.25, 1.], )
    with col1:
        if st.button("Next Sample"):
            # Update state
            add_anno(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws)
            # Rerun to load next sample
            st.rerun()

    with col2:
        if st.button("Save Annotated Results", type="primary"):
            # 1. Add current work to memory
            add_anno(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws)
            # 2. Save memory to disk/cloud
            save_results(output_file)
            # 3. Rerun to show next sample
            st.rerun()


def save_results(filename):
    if not st.session_state.annotations:
        st.warning("No new annotations to save.")
        return

    # --- Prepare Data ---
    export_data = []
    kw_map = {str(k.id): k.name for k in st.session_state.keywords}

    for ann in st.session_state.annotations:
        kept_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in ann['kept_ids']]
        added_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in ann['added_existing_ids']]

        row = ann.copy()
        # Convert lists to strings for CSV safety
        row['kept_keywords'] = ", ".join(kept_names) if isinstance(kept_names, list) else kept_names
        row['added_keywords'] = ", ".join(added_names) if isinstance(added_names, list) else added_names
        row['accepted_new_keywords'] = ", ".join(ann['accepted_new_keywords'])

        # Compute original and modified metrics vs gold
        gold_ids = ann.get('gold_ids', [])
        original_ids = [str(mid) for mid in ann.get('original_matched_ids', [])]
        modified_ids = [str(mid) for mid in ann['kept_ids']] + [str(mid) for mid in ann['added_existing_ids']]

        orig_p, orig_r, orig_j = compute_sample_metrics(gold_ids, original_ids)
        mod_p, mod_r, mod_j = compute_sample_metrics(gold_ids, modified_ids)

        row['orig_precision'] = round(orig_p, 4)
        row['orig_recall'] = round(orig_r, 4)
        row['orig_jaccard'] = round(orig_j, 4)
        row['mod_precision'] = round(mod_p, 4)
        row['mod_recall'] = round(mod_r, 4)
        row['mod_jaccard'] = round(mod_j, 4)

        export_data.append(row)

    new_df = pd.DataFrame(export_data)
    try:
        new_df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
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
