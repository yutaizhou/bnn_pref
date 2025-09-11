import glob
import os
import sys

import numpy as np

"""
hydra slurm sweeper sometimes doesn't save stats.npz for jobs. This script checks out which ones are missing.
"""

tasks = [
    # # * D4RL
    "cheetahRandom",
    "cheetahMediumReplay",
    "cheetahMediumExpert",
    "hopperRandom",
    "hopperMediumReplay",
    "hopperMediumExpert",
    "walkerRandom",
    "walkerMediumReplay",
    "walkerMediumExpert",
    "penHuman",
    "penExpert",
    "penCloned",
    "kitchenComplete",
    "kitchenPartial",
    "kitchenMixed",
    "mazeUDense",
    "mazeMediumDense",
    "mazeLargeDense",
]

dirp = sys.argv[1]
# dirp = "/scr/yutaizho/projects/bnn_pref/results_sweep/pref/20250911_204747_updateAll=True_nitersUpdate=-1_lr=0.0003_ekfM=150_ekfAcq=infogain_acq=disagreement"
# stats["cheetahRandom"]["ekf"][False]["test_logpdf_all"] = (n_seeds, n_steps)
stats = {}
print(f"Checking {dirp}")
for task in tasks:
    for folder in os.listdir(dirp):
        if f"task={task}" in folder:
            if not os.path.exists(os.path.join(dirp, folder, "stats.npz")):
                print(f"stats.npz not found for {task}")
                continue
