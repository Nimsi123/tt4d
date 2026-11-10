"""Stage 2: cut each full match into rally clips at the score changes found by the scoreboard OCR."""

import os
import pickle

import tyro

import rally_clipper.scoreboard_ocr.util as util
from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.video import create_video_subset, get_video_fps


def _load_scores(gameplay_dir):
    """Read every OCR record of sb/sb.pkl and return (all records, the score sequence derived from them)."""
    sb_save_file = os.path.join(gameplay_dir, "sb", "sb.pkl")
    with open(sb_save_file, 'rb') as file:
        all_data = []
        while True:
            try:
                obj = pickle.load(file)
                all_data.extend(obj)
            except EOFError:
                break
    return all_data, util.scores_from_raw_data(all_data)


def _clip_ranges(scores, video_fps):
    """Yield (clip start, clip end, starting score) for every score change that lasts 100 to 750 frames."""
    for c in util.changes_from_scores(scores):
        start, end = c[0]
        frame_difference = end - start
        if 100 <= frame_difference <= 750:
            # add some buffer
            start -= video_fps * 1
            end += video_fps * 1
            yield start, end, "_".join(map(str, c[1]))


def create_scoreboard_based_clips(task):
    """Write one video clip per score change into the clipped batch directory."""
    gameplay_dir, save_batch_dir = task
    id = gameplay_dir.split("/")[-1]
    _, scores = _load_scores(gameplay_dir)

    if len(scores) <= 10:
        print("******", id)
        print(len(scores))
        return

    video_path = os.path.join(gameplay_dir, "video", f"{id}.mp4")
    video_fps = int(get_video_fps(video_path)[0])

    for start, end, starting_score in _clip_ranges(scores, video_fps):
        clip = f"{id}_{start}_{end}_{starting_score}"
        if os.path.isdir(f"{save_batch_dir}/{clip}/video"):
            continue
        os.makedirs(f"{save_batch_dir}/{clip}/video")
        create_video_subset(video_path, f"{save_batch_dir}/{clip}/video/{clip}.mp4", start, end)


def add_sb_info(task):
    """Save the OCR records that fall inside each clip as sb/sb_subset.pkl next to the clip."""
    gameplay_dir, save_batch_dir = task
    id = gameplay_dir.split("/")[-1]
    all_data, scores = _load_scores(gameplay_dir)

    if len(scores) <= 10:
        print("******", id)
        print(len(scores))
        return

    video_path = os.path.join(gameplay_dir, "video", f"{id}.mp4")
    video_fps = int(get_video_fps(video_path)[0])

    for start, end, starting_score in _clip_ranges(scores, video_fps):
        clip = f"{id}_{start}_{end}_{starting_score}"
        if os.path.isdir(f"{save_batch_dir}/{clip}/video"):
            # save sb subset info
            all_data_subset = [
                (ok, idx, result) for ok, idx, result in all_data if start <= idx <= end
            ]  # find the subset of all_data corresponding to start, end
            sb_save_dir = os.path.join(save_batch_dir, clip, "sb")
            os.makedirs(sb_save_dir, exist_ok=True)
            with open(os.path.join(sb_save_dir, "sb_subset.pkl"), 'wb') as file:
                pickle.dump(all_data_subset, file)


def run_task(task):
    """Clip one match and save the OCR records of each clip."""
    create_scoreboard_based_clips(task)
    add_sb_info(task)


def discover_tasks(root_dir, batch_indices):
    """Every match with finished OCR, paired with the clipped batch directory it writes into."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_unclipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Looking through {batch_dir}.")
        save_batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        os.makedirs(save_batch_dir, exist_ok=True)
        for hash_id in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, hash_id)
            sb_save_dir = os.path.join(gameplay_dir, "sb")
            done_path = os.path.join(sb_save_dir, "done.txt")
            if os.path.exists(done_path):
                tasks.append((gameplay_dir, save_batch_dir))
    tasks = sorted(tasks, key=lambda x: x[0])

    return tasks


if __name__ == "__main__":
    # python first_clipper.py --num-resources 40 --workers-per-resource 1
    config = tyro.cli(ParallelConfig)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    run_parallel(tasks, run_task, config)
