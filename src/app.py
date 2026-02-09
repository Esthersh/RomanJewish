import os

import streamlit as st
import sys
import pandas as pd
import json
from src.data_loader import DataLoader
from src.keyword_manager import KeywordManager


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

    keywords_file = "/home/esther/antigravity/RomanJewish/Keywords_05022026.csv"

    for i, arg in enumerate(sys.argv):
        if arg == "--input_file" and i + 1 < len(sys.argv):
            # If passed via CLI, it might be a full path or just filename
            # We'll assume user knows what they are doing if they pass CLI arg
            default_input = sys.argv[i + 1]
        if arg == "--keywords_file" and i + 1 < len(sys.argv):
            keywords_file = sys.argv[i + 1]

    return default_input, keywords_file, json_files


def main():
    results_dir = "../results"
    st.set_page_config(layout="centered",
                       page_title="RomanJewish Legal Classifier - Review")

    # Sidebar
    st.sidebar.title("Review Config")
    cli_input_file, cli_keywords_file, available_files = get_config(results_dir)

    # Select menu for results file
    # Select menu for results file
    if available_files:
        selected_file = st.sidebar.selectbox(
            "Select Results JSON File",
            options=available_files,
            index=None
        )

        # FIX: Check if a file is actually selected before joining paths
        if selected_file:
            input_file = os.path.join(results_dir, selected_file)
        else:
            input_file = None
    else:
        # Fallback text input if no files found or custom path needed
        input_file = st.sidebar.text_input("Results JSON File Path", value=cli_input_file if cli_input_file else "")

    keywords_file = st.sidebar.text_input("Keywords CSV File", value=cli_keywords_file)
    output_file = st.sidebar.text_input("Output CSV File", value="annotated_results.csv")

    # Load Data
    if 'data_loaded' not in st.session_state or st.session_state.get('input_file') != input_file:
        if input_file and os.path.exists(input_file):
            try:
                loader = DataLoader()
                st.session_state.keywords = loader.load_keywords(keywords_file)

                with open(input_file, 'r') as f:
                    st.session_state.results = json.load(f)

                st.session_state.keyword_manager = KeywordManager()
                st.session_state.annotations = []
                st.session_state.current_index = 0
                st.session_state.input_file = input_file
                st.session_state.data_loaded = True
                st.success(f"Loaded {len(st.session_state.results)} samples from {input_file}.")
            except Exception as e:
                st.error(f"Error loading files: {e}")
                return
        else:
            if input_file:
                st.warning(f"File not found: {input_file}")
            else:
                st.info("Please select or provide a results file.")
            return

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
    # st.text_area("Text", result.get('text_en', ''), height=400, disabled=True)
    text_content = result.get('text_en', '')
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
            width: fit-content; /* Optional: Makes the width match the text length too */
            max-width: 100%;    /* Ensures it doesn't overflow the screen */
        ">
            {text_content}
        </div>
        """,
        unsafe_allow_html=True
    )

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
        if st.checkbox(kw_obj.full_path, value=True, key=f"cb_{current_id}_{mid}"):
            kept_ids.append(mid)

    # 2. Add Missed (FN Check)
    st.write("**Are there any missed keywords from the thesaurus?**")
    all_kw_names = [f"{k.name} (ID: {k.id})" for k in st.session_state.keywords]
    # Pre-select? No, user adds.
    added_kws = st.multiselect("Select existing keywords:", all_kw_names, key=f"ms_{current_id}")

    # 3. New Keyword Suggestions
    st.write("**Suggested Keywords (not from the original list)**")
    st.write("Edit for correction as needed")
    final_new_kws = []

    if suggested_kws:
        for i, skw in enumerate(suggested_kws):
            col_a, col_b = st.columns([0.7, 1])  # Adjusted ratio for better fit
            with col_a:
                # Added label_visibility="collapsed" to remove top padding
                edited_kw = st.text_input(
                    "Edit keyword",
                    value=skw,
                    key=f"new_{current_id}_{i}",
                    label_visibility="collapsed"
                )
            with col_b:
                # Checkbox often sits lower than text_input,
                # so we sometimes add a blank line or spacer to align them,
                # but simpler is better here.
                if st.checkbox("Accept", value=True, key=f"accept_{current_id}_{i}"):
                    final_new_kws.append(edited_kw)
    else:
        st.write("No new keywords suggested by model.")

    # Add manual new keyword?
    st.write("**Do the keywords cover all the legal concepts in the text?**")
    manual_new = st.text_input("Define any missing keywords, separated by commas (optional)",
                               key=f"manual_{current_id}")
    if manual_new:
        # final_new_kws.append(manual_new)
        final_new_kws += [e.strip() for e in manual_new.split(",")]

    # Navigation
    st.markdown("---")
    # Display progress
    st.write(f"Progress: {st.session_state.current_index + 1} / {len(st.session_state.results)}")

    col1, col2 = st.columns([0.25, 1.], )
    with col1:
        if st.button("Next Sample"):
            # Save annotation
            annotation = {
                "source_id": result.get('source_id'),
                "original_matched": matched_names,
                "kept_ids": kept_ids,
                "added_existing_ids": [k.split("(ID: ")[1].strip(")") for k in added_kws],
                "original_suggested": suggested_kws,
                "accepted_new_keywords": final_new_kws
            }
            st.session_state.annotations.append(annotation)

            # Update global list (in-memory)
            st.session_state.keyword_manager.update_keywords(final_new_kws)

            st.session_state.current_index += 1
            st.rerun()

    with col2:
        if st.button("Save Annotated Results", type="primary"):
            save_results(output_file)


def save_results(filename):
    # Process annotations to include names
    export_data = []
    kw_map = {str(k.id): k.name for k in st.session_state.keywords}

    for ann in st.session_state.annotations:
        # Resolve names
        kept_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in ann['kept_ids']]
        added_names = [kw_map.get(str(mid), f"Unknown ID {mid}") for mid in ann['added_existing_ids']]

        row = ann.copy()
        row['kept_keywords'] = kept_names
        row['added_keywords'] = added_names
        export_data.append(row)

    df = pd.DataFrame(export_data)
    df.to_csv(filename, mode='a', header=not os.path.exists(filename))
    st.success(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
