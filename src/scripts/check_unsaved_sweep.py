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
task_exists = []
stats_exists = []
print(f"Checking {dirp}")
for task in tasks:
    found_task = False
    found_stats = False
    for folder in os.listdir(dirp):
        if f"task={task}" in folder:
            found_task = True
            if os.path.exists(os.path.join(dirp, folder, "stats.npz")):
                found_stats = True
                continue

    task_exists.append(found_task)
    stats_exists.append(found_stats)


for task, task_exists, stats_exists in zip(tasks, task_exists, stats_exists):
    if not task_exists:
        print(f"{task}: task not found")
    elif task_exists and not stats_exists:
        print(f"{task}: stats not found")
