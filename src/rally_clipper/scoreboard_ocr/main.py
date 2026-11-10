"""Stage 1: OCR the scoreboard of every full match video and save the per-frame readings to sb/sb.pkl."""

import logging
import os

import torch.multiprocessing as mp
import tyro
import util

from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import remove_incomplete_process_directory

# drop all DEBUG-level messages (from any logger)
logging.disable(logging.DEBUG)


def discover_tasks(root_dir, batch_indices):
    """Every match video without sb/done.txt; a half-written sb/ directory is removed first."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_unclipped_long_{i}")
        if not os.path.isdir(batch_dir):
            continue
        for hash_id in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, hash_id)
            video_file = os.path.join(gameplay_dir, "video", f"{hash_id}.mp4")
            sb_save_dir = os.path.join(gameplay_dir, "sb")
            done_path = os.path.join(sb_save_dir, "done.txt")
            if os.path.exists(video_file) and not os.path.exists(done_path):
                if os.path.isdir(sb_save_dir):
                    remove_incomplete_process_directory(sb_save_dir)
                tasks.append(gameplay_dir)
    tasks = sorted(tasks)
    return tasks


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # Ensure PyTorch compatibility

    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    run_parallel(tasks, util.process_one, config)
