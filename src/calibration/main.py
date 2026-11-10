"""Stage 4: calibrate the camera of every clip from the table with the vendored tt3d calibrator."""

import os
import subprocess

import tyro

from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import remove_incomplete_process_directory


def process_video(gameplay_dir: str, gpu_id: int):
    """Run camera calibration using the table tennis table"""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # this proc sees only one GPU
    env["EGL_DEVICE_ID"] = str(gpu_id)
    env["PYOPENGL_PLATFORM"] = "egl"

    cmd = [
        "python",
        "tt3d/tt3d/calibration/calibrate_exhaustive.py",
        f"{gameplay_dir}",
        "--device=cuda:0",  # index 0 inside this process
    ]

    subprocess.run(
        cmd,
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
    )


def discover_tasks(root_dir, batch_indices):
    """Every clip with a video and no thomas_calibration/done.txt (a half-written output directory is removed)."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.isdir(batch_dir):
            continue
        for hash_id in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, hash_id)
            video_dir = os.path.join(gameplay_dir, "video")
            video_file = os.path.join(video_dir, hash_id + ".mp4")

            output_dir = os.path.join(gameplay_dir, "thomas_calibration")
            done_flag = os.path.join(output_dir, "done.txt")

            if os.path.isfile(video_file) and not os.path.exists(done_flag):
                tasks.append(gameplay_dir)

    tasks = sorted(tasks)
    print(f"Total tasks: {len(tasks)}")
    return tasks


if __name__ == "__main__":
    """
    python main.py --resource-ids 1 2 3 4 5 6 7 8 9 --workers-per-resource 1 --num-workers 1 --worker-split 0
    python main.py --resource-ids 1 --workers-per-resource 1 --num-workers 1 --worker-split 0"""
    config = tyro.cli(ParallelConfig)

    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    if config.worker_split == 0:
        for gameplay_dir in tasks:
            remove_incomplete_process_directory(os.path.join(gameplay_dir, "thomas_calibration"))
    tasks = config.slice_tasks(tasks)
    run_parallel(tasks, process_video, config, setup=lambda gpu_id: gpu_id)
