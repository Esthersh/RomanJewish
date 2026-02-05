import streamlit as st
import argparse
import sys
import pandas as pd
import json
import os
from src.data_loader import DataLoader
from src.keyword_manager import KeywordManager

# Function to parse arguments
@st.cache_resource
def get_config():
    input_file = "batch_results.json"
    keywords_file = "Keywords_05022026.csv"
    
    for i, arg in enumerate(sys.argv):
        if arg == "--input_file" and i + 1 < len(sys.argv):
            input_file = sys.argv[i+1]
        if arg == "--keywords_file" and i + 1 < len(sys.argv):
            keywords_file = sys.argv[i+1]
            
    return input_file, keywords_file

def main():
    st.set_page_config(layout="wide", page_title="RomanJewish Legal Classifier - Review")
    
    # Sidebar
    st.sidebar.title("Review Config")
    cli_input_file, cli_keywords_file = get_config()
    
    input_file = st.sidebar.text_input("Results JSON File", value=cli_input_file)
    keywords_file = st.sidebar.text_input("Keywords CSV File", value=cli_keywords_file)
    output_file = st.sidebar.text_input("Output Excel File", value="annotated_results.xlsx")

    # Load Data
    if 'data_loaded' not in st.session_state or st.session_state.get('input_file') != input_file:
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
            st.success(f"Loaded {len(st.session_state.results)} samples.")
        except Exception as e:
            st.error(f"Error loading files: {e}")
            return

    # Main UI
    st.title("Review Classification Results")
    
    if st.session_state.current_index >= len(st.session_state.results):
        st.success("All samples reviewed!")
        if st.button("Save Annotated Results"):
            save_results(output_file)
        return

    result = st.session_state.results[st.session_state.current_index]
    
    # Handle error or missing data fields
    if "error" in result:
        st.error(f"Sample {result.get('source_id')} had error: {result['error']}")
        if st.button("Skip"):
            st.session_state.current_index += 1
            st.rerun()
        return

    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Source Text")
        st.info(f"Group: {result.get('group')} | Name: {result.get('name')}")
        st.text_area("Text", result.get('text_en', ''), height=400, disabled=True)

    with col2:
        st.subheader("Classification Review")
        
        # Original Matches
        matched_ids = result.get('matched_ids', [])
        suggested_kws = result.get('suggested_kws', [])
        current_id = f"sample_{st.session_state.current_index}"
        
        # Keywords Managment
        kw_map = {str(k.id): k for k in st.session_state.keywords}
        
        # 1. Review Matched (FP Check)
        st.write("### Matched Keywords (Uncheck if irrelevant/FP)")
        kept_ids = []
        for mid in matched_ids:
            kw_obj = kw_map.get(str(mid))
            label = f"{kw_obj.name} ({kw_obj.full_path})" if kw_obj else f"Unknown ID: {mid}"
            if st.checkbox(label, value=True, key=f"cb_{current_id}_{mid}"):
                kept_ids.append(mid)

        # 2. Add Missed (FN Check)
        st.write("### Add Missed Keywords (FN)")
        all_kw_names = [f"{k.name} (ID: {k.id})" for k in st.session_state.keywords]
        # Pre-select? No, user adds.
        added_kws = st.multiselect("Select existing keywords:", all_kw_names, key=f"ms_{current_id}")
        
        # 3. New Keyword Suggestions
        st.write("### New Keyword Suggestions")
        final_new_kws = []
        
        if suggested_kws:
            for i, skw in enumerate(suggested_kws):
                col_a, col_b = st.columns([3, 1])
                with col_a:
                    edited_kw = st.text_input(f"Suggestion {i+1}", value=skw, key=f"new_{current_id}_{i}")
                with col_b:
                    if st.checkbox("Accept", value=True, key=f"accept_{current_id}_{i}"):
                        final_new_kws.append(edited_kw)
        else:
            st.write("No new keywords suggested by model.")
            
        # Add manual new keyword?
        manual_new = st.text_input("Add manual new keyword (optional)", key=f"manual_{current_id}")
        if manual_new:
             final_new_kws.append(manual_new)

    # Navigation
    st.markdown("---")
    # Display progress
    st.write(f"Progress: {st.session_state.current_index + 1} / {len(st.session_state.results)}")
    
    if st.button("Next Sample"):
        # Save annotation
        annotation = {
            "source_id": result.get('source_id'),
            "original_matched": matched_ids,
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
    df.to_excel(filename, index=False)
    st.success(f"Results saved to {filename}")

if __name__ == "__main__":
    main()
