"""
Reorganize SOAR numpy trajectories (proprio only) for single-task preference learning.

Output layout:
    {out_dir}/{scene_slug}/{task_slug}/{success|failure}/proprio.npy  # (N, T, 7)
    {out_dir}/{scene_slug}/{task_slug}/{success|failure}/actions.npy  # (N, T, 7)
    {out_dir}/{scene_slug}/{task_slug}/{success|failure}/returns.npy  # (N,)
    {out_dir}/{scene_slug}/{task_slug}/meta.json
    {out_dir}/index.json
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, Dict, List, Literal, Optional, Tuple

import numpy as np
import tyro
from tqdm import tqdm

Outcome = Literal["success", "failure"]
TRAJ_LEN = 100
PROPRIO_DIM = 7
ACTION_DIM = 7


@dataclass(frozen=True)
class SceneDef:
    scene_id: int
    name: str
    folder_names: Tuple[str, ...]
    tasks: Tuple[str, ...]


# * 10-scene eval subset (Table 5) + folder names in numpy release
SCENES: Tuple[SceneDef, ...] = (
    SceneDef(
        scene_id=1,
        name="green_block_wooden_bowl",
        folder_names=(
            "mix_of_everything",
            "green-block-brown-bowl",
        ),
        tasks=(
            "put the green block in the wooden bowl",
            "remove the green block from inside the wooden bowl and put it on the table",
            "put the red fruit in the wooden bowl",
            "remove the red fruit from inside the wooden bowl and put it on the table",
        ),
    ),
    SceneDef(
        scene_id=2,
        name="eggplant_wooden_bowl",
        folder_names=(
            "eggplant-wooden-bowl",
            "eggplant-green-block-lemon-wooden-bowl",
        ),
        tasks=(
            "put the purple eggplant in the brown bowl",
            "remove the purple eggplant from inside the brown bowl and put it on the table",
        ),
    ),
    SceneDef(
        scene_id=3,
        name="green_marker_blue_block_wooden_bowl",
        folder_names=(
            "green-marker-blue-block-brown-bowl",
            "green-marker-blue-fish-brown-bowl",
        ),
        tasks=(
            "move the green marker to the left side",
            "move the green marker to the right side",
            "put the blue block in the wooden bowl",
            "remove the blue block from inside the wooden bowl and put it on the table",
        ),
    ),
    SceneDef(
        scene_id=4,
        name="carrot_red_object_green_plate",
        folder_names=("green-plate-carrot-red-object",),
        tasks=(
            "put the red object on the green plate",
            "take the red object out of the green plate and put it on the table",
            "put the carrot on the green plate",
            "take the carrot out of the green plate and put it on the table",
        ),
    ),
    SceneDef(
        scene_id=5,
        name="drawer",
        folder_names=("drawer-open-close",),
        tasks=(
            "open the drawer",
            "close the drawer",
        ),
    ),
    SceneDef(
        scene_id=6,
        name="mushroom_blue_bowl",
        folder_names=("blue_bowl_mushroom",),
        tasks=(
            "put the mushroom in the blue bowl",
            "remove the mushroom from inside the blue bowl and put it on the table",
        ),
    ),
    SceneDef(
        scene_id=7,
        name="mushroom_green_spoon_metal_pot",
        folder_names=(
            "mushroom-green-spoon-silver-pot",
            "green-spoon-silver-pot",
        ),
        tasks=(
            "put the mushroom in the metal pot",
            "remove the mushroom from the metal pot and put it on the table",
            "move the green spoon to the left",
            "move the green spoon to the right",
        ),
    ),
    SceneDef(
        scene_id=8,
        name="carrot_eggplant_lemon_blue_plate",
        folder_names=("blue-tray-eggplant-lemon-carrot",),
        tasks=(
            "put the carrot on the blue plate",
            "remove the carrot from the blue plate and put it on the table",
            "put the purple eggplant on the blue plate",
            "remove the purple eggplant from the blue plate and put it on the table",
            "put the lemon on the blue plate",
            "remove the lemon from the blue plate and put it on the table",
        ),
    ),
    SceneDef(
        scene_id=9,
        name="green_veggie_pink_spoon_blue_plate",
        folder_names=("pink-spoon-green-veggie-silver-pot",),
        tasks=(
            "put the green veggie on the blue plate",
            "remove the green veggie from the blue plate and put it on the table",
            "put the pink spoon on the blue plate",
            "remove the pink spoon from the blue plate and put it on the table",
        ),
    ),
    SceneDef(
        scene_id=10,
        name="cloth",
        folder_names=(
            "cloth-blue-object-red-object",
            "blue_plate_cloth_toothpaste_red_object_yellow_ball",
            "light_green_plate_green_block_cloth_carrot_pot",
        ),
        tasks=(
            "fold the cloth from right to left",
            "unfold the cloth from left to right",
        ),
    ),
)

FOLDER_TO_SCENE: Dict[str, SceneDef] = {
    folder: scene
    for scene in SCENES
    for folder in scene.folder_names
}

# * raw language_task.txt (normalized key) -> canonical task
TASK_ALIASES: Dict[str, str] = {
    # scene 2
    "put the eggplant in the wooden bowl": "put the purple eggplant in the brown bowl",
    "remove the eggplant from inside the wooden bowl and put it on the table": (
        "remove the purple eggplant from inside the brown bowl and put it on the table"
    ),
    # scene 3
    "put the blue block in the brown bowl": "put the blue block in the wooden bowl",
    "remove the blue block from inside the brown bowl and put it on the table": (
        "remove the blue block from inside the wooden bowl and put it on the table"
    ),
    "move green marker to the left side of the table": "move the green marker to the left side",
    "move green marker to the right side of the table": "move the green marker to the right side",
    # scene 4
    "move the red object from the green plate to the table": (
        "take the red object out of the green plate and put it on the table"
    ),
    "move the carrot from the green plate to the table": (
        "take the carrot out of the green plate and put it on the table"
    ),
    # scene 7
    "put mushroom into the silver pot": "put the mushroom in the metal pot",
    "put the mushroom into the silver pot": "put the mushroom in the metal pot",
    "remove mushroom from inside the silver pot and place it on the table": (
        "remove the mushroom from the metal pot and put it on the table"
    ),
    "remove the mushroom from inside the silver pot and place it on the table": (
        "remove the mushroom from the metal pot and put it on the table"
    ),
    "move green spoon to the left side of the table": "move the green spoon to the left",
    "move green spoon to the right side of the table": "move the green spoon to the right",
    "move the green spoon to the left side of the table": "move the green spoon to the left",
    "move the green spoon to the right side of the table": "move the green spoon to the right",
    # scene 8 (blue tray <-> blue plate)
    "put the carrot on the blue tray": "put the carrot on the blue plate",
    "remove the carrot from the blue tray and put it on the table": (
        "remove the carrot from the blue plate and put it on the table"
    ),
    "put the purple eggplant on the blue tray": "put the purple eggplant on the blue plate",
    "remove the purple eggplant from the blue tray and put it on the table": (
        "remove the purple eggplant from the blue plate and put it on the table"
    ),
    "put the lemon on the blue tray": "put the lemon on the blue plate",
    "remove the lemon from the blue tray and put it on the table": (
        "remove the lemon from the blue plate and put it on the table"
    ),
    "move the carrot to the left side": "put the carrot on the blue plate",
    "move the carrot to the right side": "put the carrot on the blue plate",
    # scene 9
    "put the green veggie on the blue tray": "put the green veggie on the blue plate",
    "remove the green veggie from the blue tray and put it on the table": (
        "remove the green veggie from the blue plate and put it on the table"
    ),
    "put the pink spoon on the blue tray": "put the pink spoon on the blue plate",
    "remove the pink spoon from the blue tray and put it on the table": (
        "remove the pink spoon from the blue plate and put it on the table"
    ),
}

def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


# precompute normalized alias lookup
_ALIASES_NORM: Dict[str, str] = {
    normalize_text(k): v for k, v in TASK_ALIASES.items()
}


def task_slug(canonical_task: str) -> str:
    slug = normalize_text(canonical_task)
    slug = re.sub(r"[^a-z0-9]+", "_", slug)
    return slug.strip("_")


def scene_slug(scene: SceneDef) -> str:
    return f"scene_{scene.scene_id:02d}_{scene.name}"


def parse_task_list(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    if text.startswith("["):
        return list(json.loads(text))
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text]


def _build_canonical_lookup(tasks: Tuple[str, ...]) -> Dict[str, str]:
    return {normalize_text(task): task for task in tasks}


def resolve_canonical_task(
    language_task: str,
    scene: SceneDef,
    task_list_text: Optional[str] = None,
) -> Optional[str]:
    """
    Map trajectory language_task.txt onto one canonical task for this scene.
    """
    lang_norm = normalize_text(language_task)
    canon_by_norm = _build_canonical_lookup(scene.tasks)

    #* exact canonical match
    if lang_norm in canon_by_norm:
        return canon_by_norm[lang_norm]

    #* alias table (raw or normalized key)
    if language_task in TASK_ALIASES:
        return TASK_ALIASES[language_task]
    if lang_norm in _ALIASES_NORM:
        return _ALIASES_NORM[lang_norm]

    #* match against task_list.txt entries, then alias/canonicalize
    if task_list_text is not None:
        for raw_task in parse_task_list(text=task_list_text):
            if normalize_text(raw_task) != lang_norm:
                continue
            if raw_task in TASK_ALIASES:
                return TASK_ALIASES[raw_task]
            if lang_norm in _ALIASES_NORM:
                return _ALIASES_NORM[lang_norm]
            if normalize_text(raw_task) in canon_by_norm:
                return canon_by_norm[normalize_text(raw_task)]

    return None


@dataclass
class TrajRecord:
    proprio: np.ndarray  # (T, 7)
    actions: np.ndarray  # (T, 7)
    outcome: Outcome
    raw_language_task: str
    canonical_task: str
    source_path: str


@dataclass
class MakeSoarDatasetConfig:
    soar_numpy_root: Path = Path("data/soar/soar-dataset-local")
    """Path to soar-dataset-local/ (contains berkeley_robot_* folders)."""
    out_dir: Path = Path("data/soar/soar_pref_proprio")
    save_actions: bool = True
    verbose: bool = True


def _read_traj(traj_dir: Path, outcome: Outcome) -> Optional[TrajRecord]:
    proprio_fp = traj_dir / "eef_poses.npy"
    actions_fp = traj_dir / "actions.npy"
    language_fp = traj_dir / "language_task.txt"
    task_list_fp = traj_dir / "task_list.txt"

    if not (proprio_fp.exists() and language_fp.exists()):
        return None

    proprio = np.load(proprio_fp)  # (T, 7)
    if proprio.shape != (TRAJ_LEN, PROPRIO_DIM):
        return None

    actions = (
        np.load(actions_fp)
        if actions_fp.exists()
        else np.zeros((TRAJ_LEN, ACTION_DIM), dtype=np.float32)
    )
    if actions.shape != (TRAJ_LEN, ACTION_DIM):
        return None

    raw_language_task = language_fp.read_text().strip()
    task_list_text = (
        task_list_fp.read_text() if task_list_fp.exists() else None
    )

    scene_folder = _scene_folder_from_path(traj_dir=traj_dir)
    if scene_folder is None:
        return None
    scene = FOLDER_TO_SCENE.get(scene_folder)
    if scene is None:
        return None

    canonical_task = resolve_canonical_task(
        language_task=raw_language_task,
        scene=scene,
        task_list_text=task_list_text,
    )
    if canonical_task is None:
        return None

    return TrajRecord(
        proprio=proprio.astype(np.float32),
        actions=actions.astype(np.float32),
        outcome=outcome,
        raw_language_task=raw_language_task,
        canonical_task=canonical_task,
        source_path=str(traj_dir),
    )


def _scene_folder_from_path(traj_dir: Path) -> Optional[str]:
    prev_name: Optional[str] = None
    for parent in traj_dir.parents:
        if parent.name.startswith("berkeley_robot_"):
            return prev_name
        prev_name = parent.name
    return None


def _iter_scene_traj_dirs(
    soar_root: Path,
    scene_folder: str,
) -> List[Tuple[Path, Outcome]]:
    traj_dirs: List[Tuple[Path, Outcome]] = []
    for robot_dir in sorted(soar_root.glob("berkeley_robot_*")):
        scene_dir = robot_dir / scene_folder
        if not scene_dir.exists():
            continue
        for outcome in ("success", "failure"):
            for traj_dir in sorted(scene_dir.rglob(f"{outcome}/traj*")):
                if traj_dir.is_dir() and traj_dir.name.startswith("traj"):
                    traj_dirs.append((traj_dir, outcome))
    return traj_dirs


def collect_trajectories(
    soar_numpy_root: Path,
    verbose: bool = True,
) -> Tuple[
    DefaultDict[Tuple[int, str, Outcome], List[TrajRecord]],
    Dict[str, int],
]:
    """
    Crawl raw SOAR numpy tree and bucket by (scene_id, canonical_task, outcome).
    """
    buckets: DefaultDict[Tuple[int, str, Outcome], List[TrajRecord]] = defaultdict(
        list
    )
    stats: Dict[str, int] = defaultdict(int)

    scene_iter = SCENES
    if verbose:
        scene_iter = tqdm(SCENES, desc="scenes")

    for scene in scene_iter:
        for scene_folder in scene.folder_names:
            traj_dirs = _iter_scene_traj_dirs(
                soar_root=soar_numpy_root,
                scene_folder=scene_folder,
            )
            inner_iter = traj_dirs
            if verbose:
                inner_iter = tqdm(
                    traj_dirs,
                    desc=f"scene {scene.scene_id:02d}/{scene_folder}",
                    leave=False,
                )
            for traj_dir, outcome in inner_iter:
                stats["seen"] += 1
                record = _read_traj(traj_dir=traj_dir, outcome=outcome)
                if record is None:
                    stats["skipped"] += 1
                    continue
                key = (scene.scene_id, record.canonical_task, outcome)
                buckets[key].append(record)
                stats["kept"] += 1

    return buckets, dict(stats)


def _stack_records(records: List[TrajRecord]) -> Tuple[np.ndarray, np.ndarray]:
    proprio_NTD = np.stack([r.proprio for r in records], axis=0)  # (N, T, 7)
    actions_NTD = np.stack([r.actions for r in records], axis=0)  # (N, T, 7)
    return proprio_NTD, actions_NTD


def save_reorganized_dataset(
    buckets: DefaultDict[Tuple[int, str, Outcome], List[TrajRecord]],
    out_dir: Path,
    save_actions: bool = True,
) -> Dict:
    """
    Write scene/task/{success,failure}/ arrays to disk.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index: Dict = {"scenes": {}, "totals": {"success": 0, "failure": 0}}

    scene_by_id = {scene.scene_id: scene for scene in SCENES}
    tasks_by_scene: DefaultDict[int, set[str]] = defaultdict(set)
    for scene_id, canonical_task, _ in buckets:
        tasks_by_scene[scene_id].add(canonical_task)

    for scene_id in sorted(tasks_by_scene):
        scene = scene_by_id[scene_id]
        scene_key = scene_slug(scene=scene)
        index["scenes"][scene_key] = {"tasks": {}}

        for canonical_task in sorted(tasks_by_scene[scene_id]):
            slug = task_slug(canonical_task=canonical_task)
            task_dir = out_dir / scene_key / slug
            task_meta = {
                "scene_id": scene.scene_id,
                "scene_name": scene.name,
                "canonical_task": canonical_task,
                "task_slug": slug,
                "counts": {"success": 0, "failure": 0},
            }

            for outcome in ("success", "failure"):
                records = buckets.get((scene_id, canonical_task, outcome), [])
                if not records:
                    continue

                proprio_NTD, actions_NTD = _stack_records(records=records)
                outcome_dir = task_dir / outcome
                outcome_dir.mkdir(parents=True, exist_ok=True)

                np.save(outcome_dir / "proprio.npy", proprio_NTD)
                if save_actions:
                    np.save(outcome_dir / "actions.npy", actions_NTD)

                returns_N = np.full((len(records),), float(outcome == "success"))
                np.save(outcome_dir / "returns.npy", returns_N)

                masks_NT = np.ones((len(records), TRAJ_LEN), dtype=np.bool_)
                np.save(outcome_dir / "masks.npy", masks_NT)

                raw_language_tasks = [r.raw_language_task for r in records]
                source_paths = [r.source_path for r in records]
                with open(outcome_dir / "sources.json", "w") as f:
                    json.dump(
                        {
                            "raw_language_tasks": raw_language_tasks,
                            "source_paths": source_paths,
                        },
                        f,
                        indent=2,
                    )

                task_meta["counts"][outcome] = len(records)
                index["totals"][outcome] += len(records)

            with open(task_dir / "meta.json", "w") as f:
                json.dump(task_meta, f, indent=2)
            index["scenes"][scene_key]["tasks"][slug] = task_meta

    with open(out_dir / "index.json", "w") as f:
        json.dump(index, f, indent=2)

    return index


def make_soar_dataset(cfg: MakeSoarDatasetConfig) -> Dict:
    soar_numpy_root = Path(cfg.soar_numpy_root)
    if not soar_numpy_root.exists():
        raise FileNotFoundError(f"SOAR root not found: {soar_numpy_root}")

    buckets, crawl_stats = collect_trajectories(
        soar_numpy_root=soar_numpy_root,
        verbose=cfg.verbose,
    )
    index = save_reorganized_dataset(
        buckets=buckets,
        out_dir=Path(cfg.out_dir),
        save_actions=cfg.save_actions,
    )
    index["crawl_stats"] = crawl_stats
    return index


def main(cfg: MakeSoarDatasetConfig) -> None:
    index = make_soar_dataset(cfg=cfg)
    print(json.dumps(index["crawl_stats"], indent=2))
    print(f"Wrote reorganized dataset to {cfg.out_dir}")


if __name__ == "__main__":
    main(tyro.cli(MakeSoarDatasetConfig))
