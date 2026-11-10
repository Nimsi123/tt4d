"""Stage 5: track the ball in 2D with TrackNet and write ball_tracker_only_tracknet/<clip>_ball.csv per clip."""

import os

import torch
import torch.multiprocessing as mp
import tyro
from predict import only_tracknet_one_video
from utils.general import get_model

from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import get_home_path, remove_incomplete_process_directory

TRACKNET_WEIGHTS = f"{get_home_path()}/src/ball_tracker/finetune_10/TrackNet_best.pt"


def load_tracknet(gpu_id):
    """Load the fine-tuned TrackNet once per worker; returns (model, sequence length, background mode)."""
    device = torch.device("cuda:0")  # the worker sees only its GPU

    print(f"[Worker for GPU {gpu_id}] Loading models onto GPU {gpu_id} ...")
    tracknet_ckpt = torch.load(TRACKNET_WEIGHTS, map_location=device)
    raw_state = tracknet_ckpt['model']
    # remove the "module." prefix because we used torch ddp
    new_state = {k.replace("module.", ""): v for k, v in raw_state.items()}

    tracknet_seq_len = tracknet_ckpt['param_dict']['seq_len']
    bg_mode = tracknet_ckpt['param_dict']['bg_mode']

    tracknet = get_model('TrackNet', tracknet_seq_len, bg_mode).to(device)
    tracknet.load_state_dict(new_state)
    tracknet.eval()

    print(f"[Worker for GPU {gpu_id}] Models loaded. Waiting for tasks...")
    return tracknet, tracknet_seq_len, bg_mode


def run_task(gameplay_dir, state):
    """Run TrackNet over one clip's video and mark the output directory done."""
    tracknet, tracknet_seq_len, bg_mode = state
    hash_id = os.path.basename(gameplay_dir)
    video_file = os.path.join(gameplay_dir, "video", f"{hash_id}.mp4")
    save_dir = os.path.join(gameplay_dir, "ball_tracker_only_tracknet")
    output_video = "test" in gameplay_dir  # example check

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    eval_mode = "weight"
    batch_size = 16
    traj_len = 8
    num_workers = 2

    only_tracknet_one_video(
        video_file,
        tracknet_seq_len,
        bg_mode,
        tracknet,
        save_dir,
        eval_mode,
        batch_size,
        output_video,
        traj_len,
        num_workers,
    )

    # Mark completion
    with open(os.path.join(save_dir, "done.txt"), "w") as _:
        pass


def discover_tasks(root_dir, batch_indices):
    """Every calibrated clip (calib.yaml, no moves.txt or failed.txt) without ball_tracker_only_tracknet/done.txt."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Going through {batch_dir}.")
        for folder_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, folder_name)
            video_dir = os.path.join(gameplay_dir, "video")
            video_file = os.path.join(video_dir, folder_name + ".mp4")
            save_dir = os.path.join(gameplay_dir, "ball_tracker_only_tracknet")
            done_path = os.path.join(save_dir, "done.txt")

            calib_dir = os.path.join(gameplay_dir, "thomas_calibration")
            calib_file = os.path.join(calib_dir, "calib.yaml")
            calib_moves_file = os.path.join(calib_dir, "moves.txt")
            calib_failed_file = os.path.join(calib_dir, "failed.txt")

            if (
                os.path.isfile(video_file)
                and not os.path.exists(done_path)
                and os.path.exists(calib_file)
                and not os.path.exists(calib_moves_file)
                and not os.path.exists(calib_failed_file)
            ):
                tasks.append(gameplay_dir)
    tasks = sorted(tasks)
    print(f"Total tasks: {len(tasks)}")
    return tasks


if __name__ == "__main__":
    """python only_tracknet_main.py --resource-ids 1 2 3 4 5 6 7 8 9 --workers-per-resource 1 --num-workers 1 --worker-split 0"""
    mp.set_start_method("spawn", force=True)  # Required for PyTorch + multiprocessing on some platforms

    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    if config.worker_split == 0:
        for gameplay_dir in tasks:
            remove_incomplete_process_directory(os.path.join(gameplay_dir, "ball_tracker_only_tracknet"))
    tasks = config.slice_tasks(tasks)
    run_parallel(tasks, run_task, config, setup=load_tracknet)
