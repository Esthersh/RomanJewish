import os
import sys

# Mocking streamlit to verify path logic without running the full app
def verify_paths():
    script_path = os.path.abspath("src/app.py") # Simulating the path relative to CWD if running from root
    
    # Logic copied from app.py
    script_dir = os.path.dirname(script_path)
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")
    
    print(f"Script Dir: {script_dir}")
    print(f"Project Root: {project_root}")
    print(f"Results Dir: {results_dir}")
    
    if os.path.exists(results_dir):
        print("SUCCESS: Results directory found.")
        # List content
        print("Contents:", os.listdir(results_dir))
    else:
        print("FAILURE: Results directory not found.")
        sys.exit(1)

if __name__ == "__main__":
    verify_paths()
