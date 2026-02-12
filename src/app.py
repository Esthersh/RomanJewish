import os
import streamlit as st
import pandas as pd
import sys
import json

from streamlit_gsheets import GSheetsConnection

from data_loader import DataLoader
from keyword_manager import KeywordManager


# Function to parse arguments
@st.cache_resource
def get_config(results_dir):
    # Ensure results directory exists
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # List JSON files in results directory
    json_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]

    # Default to first file if available, otherwise None
    default_input = json_files[0] if json_files else None

    # Default keywords file
    keywords_file = "keywords.csv" 
    if os.path.exists("/home/esther/antigravity/RomanJewish/Keywords_05022026.csv"):
        keywords_file = "/home/esther/antigravity/RomanJewish/Keywords_05022026.csv"

    for i, arg in enumerate(sys.argv):
        if arg == "--input_file" and i + 1 < len(sys.argv):
            default_input = sys.argv[i + 1]
        if arg == "--keywords_file" and i + 1 < len(sys.argv):
            keywords_file = sys.argv[i + 1]

    return default_input, keywords_file, json_files


def create_annotation(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws):
    """Creates the annotation dictionary."""
    return {
        "source_id": result.get('source_id'),
        "text": result.get("text_en"),
        "group": result.get("group"),
        "name": result.get("name"),
        "original_matched": matched_names,
        "kept_ids": kept_ids,
        "added_existing_ids": [k.split("(ID: ")[1].strip(")") for k in added_kws],
        "original_suggested": suggested_kws,
        "accepted_new_keywords": final_new_kws
    }


def add_anno(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws):
    """Adds the annotation to the session state."""
    annotation = create_annotation(result, matched_names, kept_ids, added_kws, suggested_kws, final_new_kws)

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


def main():
    # Fix: Resolve results_dir relative to the script location
    # This ensures it works whether running from root or src/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")

    st.set_page_config(layout="centered",
                       page_title="RomanJewish Legal Classifier - Review")

    # --- INITIALIZE SESSION STATE KEYS ---
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'results' not in st.session_state:
        st.session_state.results = []
    if 'annotations' not in st.session_state:
        st.session_state.annotations = []
    if 'keywords' not in st.session_state:
        st.session_state.keywords = []
    # -------------------------------------

    # Sidebar
    st.sidebar.title("Review Config")
    cli_input_file, cli_keywords_file, available_files = get_config(results_dir)

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

    st.session_state.keywords_file = st.sidebar.text_input("Keywords CSV File", value=cli_keywords_file)
    output_file = st.sidebar.text_input("Output CSV File", value="annotated_results.csv")

    # Load Data
    load_data(input_file)

    # Main UI
    st.title("Review Classification Results")

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
    
    display_rtl_text(result.get('text_en', ''))

    st.subheader("Classification Review")

    # Original Matches
    matched_ids = result.get('matched_ids', [])
    matched_names = result.get('matched_keywords', [])
    suggested_kws = result.get('suggested_kws', [])
    current_id = f"sample_{st.session_state.current_index}"

    # Keywords Managment
    kw_map = {str(k.id): k for k in st.session_state.keywords}

    # 1. Review Matched (FP Check)
    st.write("**Matched Keywords**: Uncheck if irrelevant")
    kept_ids = []
    for mid in matched_ids:
        kw_obj = kw_map.get(str(mid))
        # Handle case where keyword ID might not be in the loaded keywords file
        label = kw_obj.full_path if kw_obj else f"Unknown ID: {mid}"
        if st.checkbox(label, value=True, key=f"cb_{current_id}_{mid}"):
            kept_ids.append(mid)

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

        export_data.append(row)

    new_df = pd.DataFrame(export_data)
    try:
        new_df.to_csv(filename, mode='a', header=not os.path.exists(filename), index=False)
        st.toast(f"Saved locally to {filename}")
    except Exception as e:
        st.error(f"Failed to save local CSV: {e}")

    # --- Google Sheets Read -> Append -> Update ---
    with st.spinner('Syncing with Google Sheets...'):
        try:
            if 'conn' not in st.session_state:
                st.session_state.conn = st.connection("gsheets", type=GSheetsConnection)

            # 1. Read existing data to prevent overwriting
            try:
                # ttl=0 ensures we don't get a cached version of the sheet
                existing_df = st.session_state.conn.read(worksheet="Sheet1", ttl=0)
                # Ensure we are working with a DataFrame
                if existing_df is None:
                    existing_df = pd.DataFrame()
            except Exception:
                # If sheet is empty or doesn't exist yet
                existing_df = pd.DataFrame()

            # 2. Combine Data
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)

            # 3. Write back the FULL dataset
            st.session_state.conn.update(worksheet="Sheet1", data=combined_df)

            st.success("Google Sheet updated successfully!")

            # Clear the buffer so we don't save these duplicates again next time
            st.session_state.annotations = []

        except Exception as e:
            st.error(f"Google Sheet Error: {e}")


if __name__ == "__main__":
    main()
