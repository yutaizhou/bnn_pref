import os
import shutil
from pathlib import Path


def clean_empty_dirs(dirp):
    """
    Delete directories that don't contain any PNG files.

    Args:
        root_dir (str): Path to the root directory to check
    """
    root_path = Path(dirp)

    # Ensure the root directory exists
    if not root_path.exists():
        print(f"Directory {dirp} does not exist!")
        return

    # Get all directories in the root path
    dirs = [d for d in root_path.iterdir() if d.is_dir()]

    for dir_path in dirs:
        # Check if directory contains any PNG files
        has_png = any(file.suffix.lower() == ".png" for file in dir_path.rglob("*"))
        has_npz = any(file.suffix.lower() == ".npz" for file in dir_path.rglob("*"))
        has_ckpt = any(
            subdir.name == "ckpts" and subdir.is_dir() for subdir in dir_path.rglob("*")
        )
        if not (has_png or has_npz or has_ckpt):
            print(f"Deleting directory {dir_path} (no PNG or NPZ or CKPT files found)")
            try:
                shutil.rmtree(dir_path)
            except Exception as e:
                print(f"Error deleting {dir_path}: {e}")
        else:
            print(f"Keeping directory {dir_path} ({has_png=}, {has_npz=}, {has_ckpt=})")


if __name__ == "__main__":
    # Get the directory path from user input
    root_dir = "PATH/TO/YOUR/bnn_pref"
    directories = [
        f"{root_dir}/{d}"
        for d in [
            "results",
            "results/pref",
            "results/offline_rl",
            "results_sweep",
            "results_sweep/pref",
            "results_sweep/offline_rl",
        ]
    ]

    # Confirm before proceeding
    confirm = input("Any hydra jobs running? Are you sure you want to proceed? (y/n): ")
    if confirm == "y":
        for directory in directories:
            clean_empty_dirs(directory)
        print("Done!")
    else:
        print("Aborting...")
