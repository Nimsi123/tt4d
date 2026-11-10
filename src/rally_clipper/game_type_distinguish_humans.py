"""Stage 9: label each clip singles or doubles again, now from how many 4DHumans bodies stand on each side."""

import os
from collections import defaultdict

import numpy as np
import tyro

from rtt_utils.constants import TABLE_LENGTH, TABLE_WIDTH
from rtt_utils.humans import load_task
from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import load_file_paths

# Play-area filter. Tighter than get_left_right_players (which uses
# 3*TABLE_LENGTH) because those helpers then sort-and-pick the closest, which
# rejects umpires/coaches standing well behind a player. We don't do that
# pick-closest step, so we must filter umpires directly here.
PLAY_X_MAX = 2.0 * TABLE_LENGTH
PLAY_Y_MAX = TABLE_WIDTH

# Fraction of the only_tracknet frames that must produce a non-abstain vote
# before we'll trust the majority. Below this we mark humans_failed.
MIN_VOTE_FRACTION = 0.10


def vote_singles_or_doubles(metadata, smpl):
    """
    Per-frame vote: count valid (in-play-area) human detections on the left
    half (x < 0) and right half (x > 0). Real doubles play has two players on
    *each* side simultaneously, so:
      - doubles if both sides have >= 2 distinct players
      - singles if both sides have exactly 1
      - abstain otherwise (asymmetric like 1 vs 2 -> a single side with an
        umpire/coach is not enough evidence either way)

    Returns: (singles_votes, doubles_votes, abstain_votes, n_total_frames)
    """
    # Group detections by frame
    by_frame = defaultdict(list)  # frame -> list of (pid, x, y)
    for m, s in zip(metadata, smpl):
        frame = int(m["frame"])
        pid = m["object_id"]
        transl = dict(s)["transl"]
        x, y = float(transl[0]), float(transl[1])
        by_frame[frame].append((pid, x, y))

    n_total_frames = len(by_frame)

    singles_votes = 0
    doubles_votes = 0
    abstain_votes = 0

    for dets in by_frame.values():
        left_pids = set()
        right_pids = set()
        for pid, x, y in dets:
            if abs(x) > PLAY_X_MAX or abs(y) > PLAY_Y_MAX:
                continue
            if x < 0:
                left_pids.add(pid)
            elif x > 0:
                right_pids.add(pid)
            # x == 0 is the table center, ignore.

        if len(left_pids) >= 2 and len(right_pids) >= 2:
            doubles_votes += 1
        elif len(left_pids) == 1 and len(right_pids) == 1:
            singles_votes += 1
        else:
            abstain_votes += 1

    return singles_votes, doubles_votes, abstain_votes, n_total_frames


def run_task(gameplay_dir):
    if not os.path.isdir(gameplay_dir):
        return
    paths = load_file_paths(gameplay_dir)
    metadata_dir = paths["metadata_dir"]
    os.makedirs(metadata_dir, exist_ok=True)

    singles_humans_flag = os.path.join(metadata_dir, "singles_humans.txt")
    doubles_humans_flag = os.path.join(metadata_dir, "doubles_humans.txt")
    failed_humans_flag = os.path.join(metadata_dir, "humans_failed.txt")

    # Clear any stale flags so a re-run reflects current data.
    for p in (singles_humans_flag, doubles_humans_flag, failed_humans_flag):
        if os.path.exists(p):
            os.remove(p)

    try:
        task = load_task(gameplay_dir, ret_frames=False, thin=False)
    except Exception as e:
        print(f"[{gameplay_dir}] load_task failed: {e}")
        with open(failed_humans_flag, "w"):
            pass
        return

    metadata = task["metadata"]
    smpl = task["smpl"]

    if metadata is None or smpl is None or len(metadata) == 0:
        print(f"[{gameplay_dir}] no 4DHumans data available")
        with open(failed_humans_flag, "w"):
            pass
        return

    singles_votes, doubles_votes, abstain_votes, n_total_frames = vote_singles_or_doubles(metadata, smpl)
    decided_votes = singles_votes + doubles_votes
    min_required = max(1, int(np.ceil(MIN_VOTE_FRACTION * n_total_frames)))

    if decided_votes < min_required:
        print(
            f"[{gameplay_dir}] insufficient votes: "
            f"singles={singles_votes} doubles={doubles_votes} abstain={abstain_votes} "
            f"frames={n_total_frames} min_required={min_required}"
        )
        with open(failed_humans_flag, "w"):
            pass
        return

    # Tie -> singles (more common, conservative).
    label = "doubles_humans" if doubles_votes > singles_votes else "singles_humans"
    flag_path = doubles_humans_flag if label == "doubles_humans" else singles_humans_flag
    with open(flag_path, "w"):
        pass

    print(
        f"[{gameplay_dir}] {label} "
        f"(singles={singles_votes} doubles={doubles_votes} abstain={abstain_votes} frames={n_total_frames})"
    )


def discover_tasks(root_dir, batch_indices):
    """Cheap discovery: just enumerate clip dirs. We let run_task decide
    whether 4DHumans data is actually available; doing per-clip stat()s here
    serially would dominate the runtime over many batches."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Looking through {batch_dir}.", flush=True)
        for clip_name in sorted(os.listdir(batch_dir)):
            gameplay_dir = os.path.join(batch_dir, clip_name)
            tasks.append(gameplay_dir)
    return tasks


if __name__ == "__main__":
    # python game_type_distinguish_humans.py --num-resources 40 --workers-per-resource 1
    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    run_parallel(tasks, run_task, config)
