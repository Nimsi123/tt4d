"""Optional after stage 5: draw the tracked ball on each clip's video (ball_tracker_only_tracknet/<clip>_only_tracknet_ball.mp4)."""

import os

import tyro

from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.video import overlay_ball_on_video


def run_task(gameplay_dir):
    """Draw the tracked ball on the clip's video."""
    hash = os.path.basename(gameplay_dir)
    video_file = os.path.join(gameplay_dir, "video", f"{hash}.mp4")
    save_dir = os.path.join(gameplay_dir, "ball_tracker_only_tracknet")
    save_file = os.path.join(save_dir, f"{hash}_only_tracknet_ball.mp4")
    frame_list_path = os.path.join(save_dir, f"{hash}_ball.csv")

    overlay_ball_on_video(video_file, save_file, frame_list_path)


def discover_tasks(root_dir, batch_indices, video_data_folder, ball_data_folder):
    """Every clip with a video and finished ball tracking."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.isdir(batch_dir):
            continue
        for clip_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, clip_name)
            video_dir = os.path.join(gameplay_dir, video_data_folder)
            video_file = os.path.join(video_dir, clip_name + ".mp4")
            save_dir = os.path.join(gameplay_dir, ball_data_folder)
            done_path = os.path.join(save_dir, "done.txt")
            if os.path.isfile(video_file) and os.path.exists(done_path):
                tasks.append(gameplay_dir)

    tasks = sorted(tasks)
    print(f"Identified {len(tasks)} tasks.")

    return tasks


if __name__ == "__main__":
    """python overlay_ball_on_videos.py --num-resources 5 --workers-per-resource 1"""
    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices(), "video", "ball_tracker_only_tracknet")
    run_parallel(tasks, run_task, config)
