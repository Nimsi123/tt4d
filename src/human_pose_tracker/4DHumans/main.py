"""Stage 8: run 4DHumans/PHALP (track.py) on every second-clipped clip to get per-frame SMPL bodies."""

import os
import subprocess

import tyro

from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import load_file_paths, remove_extraneous_process_items


def process_video(task, gpu_id: int):
    """Run 4DHumans/PHALP tracking on a single video."""
    video_dir, output_dir = task
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)  # this proc sees only one GPU
    env["EGL_DEVICE_ID"] = str(gpu_id)
    env["PYOPENGL_PLATFORM"] = "egl"

    cmd = [
        "python",
        "track.py",
        f"video.source={video_dir}",
        "device=cuda:0",  # index 0 inside this process
        "video.start_frame=-1",
        "video.end_frame=-1",
        f"video.output_dir={output_dir}",
        "base_tracker=pose",
        "phalp.low_th_c=0.8",
        "phalp.small_w=25",
        "phalp.small_h=50",
    ]

    subprocess.run(
        cmd,
        check=True,
        env=env,
        stdout=subprocess.DEVNULL,
    )


def discover_tasks(root_dir, batch_indices, clean):
    """Return (video_dir, output_dir) for every second-clipped clip without 4DHumans done.txt; `clean` removes partial output."""
    tasks = []

    for i in batch_indices:
        batch_dir = f"{root_dir}/batch_clipped_long_{i}"
        if not os.path.isdir(batch_dir):
            continue
        for clip_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, clip_name)
            paths = load_file_paths(gameplay_dir)

            video_dir = paths["video_second_clipped_only_tracknet_dir"]
            video_file = paths["video_second_clipped_only_tracknet_file"]
            output_dir = paths["human_pose_tracker_only_tracknet_dir"]
            done_flag = paths["human_pose_tracker_only_tracknet_done_path"]

            if os.path.isfile(video_file) and not os.path.exists(done_flag):
                if clean:
                    remove_extraneous_process_items(
                        [
                            os.path.join(output_dir, x)
                            for x in ["_DEMO", "_TMP", ".hydra", "track.log", "results", "results_tracks", clip_name]
                        ]
                    )
                tasks.append((video_dir, output_dir))
    tasks = sorted(tasks, key=lambda x: x[0])
    return tasks


if __name__ == "__main__":
    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices(), clean=config.worker_split == 0)
    tasks = config.slice_tasks(tasks)
    run_parallel(tasks, process_video, config, setup=lambda gpu_id: gpu_id)
