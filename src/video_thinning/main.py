"""Stage 7: find the duplicated frames of each clip (e.g. 25 fps footage stored at 30 fps) and save their indices."""

import json
import os

import tyro

from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import load_file_paths, remove_incomplete_process_directory
from video_thinning.preprocess import duplicate_frame_indices


def run_task(task):
    """Write video_thinning/frames_to_drop.json for one clip."""
    gameplay_dir, video_file, save_dir, save_file = task
    os.makedirs(save_dir, exist_ok=True)
    frames_to_drop = duplicate_frame_indices(video_file)
    with open(save_file, "w") as f:
        json.dump(frames_to_drop, f)


def discover_tasks(root_dir, batch_indices):
    """Every calibrated, second-clipped clip without video_thinning/frames_to_drop.json."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Going through {batch_dir}.")
        for clip_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, clip_name)
            paths = load_file_paths(gameplay_dir)

            video_file = paths["video_file"]
            video_only_tracknet_file = paths["video_second_clipped_only_tracknet_file"]
            save_dir = paths["video_thinning_dir"]
            done_path = paths["video_thinning_frames_to_drop_file"]
            calib_file = paths["calib_file"]
            calib_moves_file = paths["calib_moves_file"]
            calib_failed_file = paths["calib_failed_file"]

            if (
                os.path.isfile(video_file)
                and os.path.isfile(video_only_tracknet_file)
                and not os.path.exists(done_path)
                and os.path.exists(calib_file)
                and not os.path.exists(calib_moves_file)
                and not os.path.exists(calib_failed_file)
            ):
                tasks.append((gameplay_dir, video_file, save_dir, done_path))
    tasks = sorted(tasks)
    print(f"Total tasks: {len(tasks)}")
    return tasks


if __name__ == "__main__":
    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    if config.worker_split == 0:
        for _, _, save_dir, _ in tasks:
            remove_incomplete_process_directory(save_dir)
    tasks = config.slice_tasks(tasks)
    run_parallel(tasks, run_task, config)
