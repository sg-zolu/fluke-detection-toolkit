import os
from .fluke_11section import run_fluke_extraction_for_deployment

def run_fluke_extraction_all(base_dir):
    for folder in os.listdir(base_dir):
        dep_path = os.path.join(base_dir, folder)
        if not os.path.isdir(dep_path):
            continue

        summary_csv = os.path.join(dep_path, f"{folder}_summary.csv")
        corrected_frames_path = os.path.join(dep_path, "corrected_frames")

        if os.path.isfile(summary_csv) and os.path.isdir(corrected_frames_path):
            print(f"\n🔍 Processing deployment: {folder}")
            try:
                run_fluke_extraction_for_deployment(base_dir, folder)
            except Exception as e:
                print(f"❌ Error in {folder}: {e}")
        else:
            print(f"⚠️ Skipping {folder}: summary or corrected_frames not found.")
