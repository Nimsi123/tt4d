"""Stage 10: build the partial TT4D (calibration, players, 2D ball) of every clip and save nima_recons/<clip>_tt4d.pkl."""

import multiprocessing as mp
import os
import shutil
from concurrent.futures import ThreadPoolExecutor

import tyro

from rtt_utils.parallel import ParallelConfig, run_parallel
from ttg.types.tt4d import TT4D


def load_smpl(gpu_id):
    """Load the SMPL model once per worker."""
    return TT4D._get_smpl()


def run_task(gameplay_dir, smpl_model):
    """Process a gameplay directory and save the TT4D object."""
    clip_name = os.path.basename(gameplay_dir)
    save_dir = os.path.join(gameplay_dir, "nima_recons")
    pkl_path = os.path.join(save_dir, f"{clip_name}_tt4d.pkl")
    # Clear any stale prior output before regenerating. Moved here from
    # discover_tasks so discovery stays pure I/O-inspection.
    if os.path.exists(pkl_path):
        shutil.rmtree(save_dir)

    tt4d = TT4D.from_gameplay_dir_partial(gameplay_dir, smpl_model)
    tt4d.save(pkl_path)


def _read_first_csv_line(path):
    """Read the first line of a CSV-ish file as a list of floats.

    Replaces np.loadtxt(..., max_rows=1) which pays full numpy parser overhead
    per file — prohibitive at discovery scale.
    """
    with open(path) as f:
        line = f.readline()
    return [float(x) for x in line.strip().split(",")]


def _check_clip(gameplay_dir):
    """Validate a single clip directory. Returns gameplay_dir on success, else None.

    Checks are ordered by selectivity so we early-out on the most common
    failures before issuing the remaining stats.
    """
    clip_name = os.path.basename(gameplay_dir)

    # (1) Must be second-clipped on the only_tracknet side.
    clip_indices_only_tracknet_path = os.path.join(gameplay_dir, "metadata_only_tracknet", "second_clipped_indices.txt")
    if not os.path.exists(clip_indices_only_tracknet_path):
        return None

    # (2) 4DHumans result. Prefer the only_tracknet location (no subset check
    # needed); fall back to the generic location which requires the tracknet
    # slice to be a subset of the generic slice.
    path_4dh_only_tracknet = os.path.join(
        gameplay_dir,
        "human_pose_tracker_only_tracknet",
        "4DHumans",
        clip_name,
        f"{clip_name}_keypoints_2d.npy",
    )
    if not os.path.exists(path_4dh_only_tracknet):
        path_4dh = os.path.join(
            gameplay_dir,
            "human_pose_tracker",
            "4DHumans",
            clip_name,
            f"{clip_name}_keypoints_2d.npy",
        )
        if not os.path.exists(path_4dh):
            return None

        clip_indices_path = os.path.join(gameplay_dir, "metadata", "second_clipped_indices.txt")
        if not os.path.exists(clip_indices_path):
            return None

        try:
            clip_indices = _read_first_csv_line(clip_indices_path)
            clip_indices_only_tracknet = _read_first_csv_line(clip_indices_only_tracknet_path)
        except (OSError, ValueError):
            return None

        is_subset = (
            clip_indices[0] <= clip_indices_only_tracknet[0] and clip_indices_only_tracknet[1] <= clip_indices[1]
        )
        if not is_subset:
            return None

    # (3) Calibration: one scandir replaces three exists() calls.
    calib_dir = os.path.join(gameplay_dir, "thomas_calibration")
    try:
        calib_entries = {e.name for e in os.scandir(calib_dir)}
    except OSError:
        return None
    if "calib.yaml" not in calib_entries or "moves.txt" in calib_entries or "failed.txt" in calib_entries:
        return None

    # (4) Ball tracker output.
    ball_file = os.path.join(gameplay_dir, "ball_tracker_only_tracknet", f"{clip_name}_ball.csv")
    if not os.path.exists(ball_file):
        return None

    return gameplay_dir


def discover_tasks(root_dir: str, batch_indices: list[int], stat_workers: int = 64):
    """Discover valid gameplay directories across the given batch range.

    Per-clip validation is stat-heavy and IO-bound, so we fan it out across a
    thread pool. Threads (not processes) because the workload is stat/open
    syscalls that release the GIL.
    """
    # Enumerate candidate clip directories using scandir (cheaper than listdir
    # + os.path.join + later exists() calls since DirEntry.path is cached).
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Going through {batch_dir}.")

        candidates = []
        with os.scandir(batch_dir) as it:
            for entry in it:
                if entry.is_dir():
                    candidates.append(entry.path)

        # Parallel per-clip validation.
        with ThreadPoolExecutor(max_workers=stat_workers) as ex:
            for result in ex.map(_check_clip, candidates):
                if result is not None:
                    tasks.append(result)

    tasks = sorted(tasks, key=lambda x: x[0])
    print(f"Identified {len(tasks)} tasks.")

    return tasks


if __name__ == "__main__":
    """python partial_recons.py --resource_ids 0 1 2 3 4 5 6 7 8 --workers_per_resource 2"""
    mp.set_start_method("spawn", force=True)

    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    run_parallel(tasks, run_task, config, setup=load_smpl)
