import argparse
import os
import shutil


def organize_traces(root_path):
    """
    usage:  python save_trace_plots.py .../bnn_pref/results_sweep/20250217_170202

    """
    # Create output directory
    ts = root_path.split("/")[-1]
    output_dir = os.path.join(root_path, "trace")
    output_dir = f"{output_dir}_{ts}"
    os.makedirs(output_dir, exist_ok=True)

    # Track used names for collision avoidance
    used_names = set()

    for root, dirs, files in os.walk(root_path):
        if "trace.png" in files:
            parent_folder = os.path.basename(root)
            base_name = f"{parent_folder}_trace.png"
            dest_path = os.path.join(output_dir, base_name)

            # Handle duplicate names
            counter = 1
            while os.path.exists(dest_path) or base_name in used_names:
                base_name = f"{parent_folder}_{counter}_trace.png"
                dest_path = os.path.join(output_dir, base_name)
                counter += 1

            used_names.add(base_name)
            shutil.copy2(os.path.join(root, "trace.png"), dest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize trace files")
    parser.add_argument("root_path", help="Root directory to scan")
    args = parser.parse_args()
    organize_traces(args.root_path)
