"""Stage 3: label each clip singles or doubles from the scoreboard text (a "/" in a name means doubles)."""

import os
import pickle

import tyro

from rtt_utils.parallel import ParallelConfig, run_parallel


def detail_from_clip(clip: str):
    """
    Parse a clip folder name like:
      CYcOic8UHpU_429075_429730_0_6_2_7
      bMWLSDeKxp0_17105_17530_0_0_0_1
      _a3Q98UDsog_23360_23755_0_1_0_0
      C0vaE_eDHPY_335307_336007_0_8_1_9

    Strategy:
      - Split on underscores
      - Everything before the first integer = hash
      - First integer = start
      - Remainder = score string
    """
    parts = [p for p in clip.split("_") if p]

    start_idx = None
    for i, p in enumerate(parts):
        if p.isdigit():
            start_idx = i
            break

    if start_idx is None:
        raise ValueError(f"No integer part found in clip name: {clip}")

    vid_hash = "_".join(parts[:start_idx]) or clip
    try:
        start = int(parts[start_idx])
    except Exception as e:
        raise ValueError(f"Failed to parse start int from {clip!r}, part={parts[start_idx]!r}") from e

    score = "_".join(parts[start_idx + 1 :]) if len(parts) > start_idx + 1 else ""
    return vid_hash, start, score


def iter_pickle(sb_path: str):
    """Yield the OCR records of a multi-object pickle one at a time."""
    with open(sb_path, "rb") as f:
        while True:
            try:
                obj = pickle.load(f)
            except EOFError:
                return
            if isinstance(obj, (list, tuple)):
                yield from obj
            else:
                yield obj


def run_task(gameplay_dir):
    """Write metadata/singles.txt or metadata/doubles.txt for one clip."""
    clip_name = gameplay_dir.split("/")[-1]
    sb_save_dir = os.path.join(gameplay_dir, "sb")
    sb_subset_file = os.path.join(sb_save_dir, "sb_subset.pkl")
    metadata_dir = os.path.join(gameplay_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    _, start, _ = detail_from_clip(clip_name)

    is_doubles = False

    FRAME_WIN = 300  # frames to scan after start frame
    low, high = (start, start + FRAME_WIN)

    for rec in iter_pickle(sb_subset_file):
        if not (isinstance(rec, (list, tuple)) and len(rec) == 3):
            continue
        ok, idx, result = rec
        if not ok or result is None:
            continue
        if not isinstance(idx, (int, float)) or idx < low or idx > high:
            continue

        dets = result[0] if isinstance(result, (list, tuple)) and result else None
        if not dets:
            continue

        for r in dets:
            try:
                token = r[1][0]
            except Exception:
                continue
            if isinstance(token, str) and "/" in token:
                is_doubles = True

                break
        if is_doubles:
            break

    label = "doubles" if is_doubles else "singles"
    with open(os.path.join(metadata_dir, f"{label}.txt"), 'w') as _:
        pass


def discover_tasks(root_dir, batch_indices):
    """Every clip that has sb/sb_subset.pkl."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Looking through {batch_dir}.")
        for clip_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, clip_name)
            sb_save_dir = os.path.join(gameplay_dir, "sb")
            sb_subset_file = os.path.join(sb_save_dir, "sb_subset.pkl")
            if os.path.exists(sb_subset_file):
                tasks.append(gameplay_dir)
            else:
                print(gameplay_dir)
    tasks = sorted(tasks)

    return tasks


if __name__ == "__main__":
    # python game_type_distinguish.py --num-resources 40 --workers-per-resource 1
    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    run_parallel(tasks, run_task, config)
