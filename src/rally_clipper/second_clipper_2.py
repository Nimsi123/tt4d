"""Stage 6: cut each clip down to the rally itself, found from the 2D ball track, and clip the ball CSV with it."""

import os

import numpy as np
import pandas as pd
import tyro

import rally_clipper.scoreboard_ocr.util as util
from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.video import create_video_subset


def second_clip_only_tracknet(df):
    """Return the (start, end) frames of the rally in a clip from its 2D ball track, or (-1, -1) if none is found."""
    # Parameters
    N1 = 15
    N2 = 6
    horizontal_kernel = np.hstack((np.ones(N1), np.zeros(N2), -np.ones(N1)))
    N1 = 20
    vertical_kernel = -np.hstack((np.ones(N1), -np.ones(N1)))
    minimum_point_length_threshold = 60
    corr_x_peak = 50

    x = df["X"].to_numpy()
    y = df["Y"].to_numpy()

    # intervals where both signals hold zero for at least 20 frames
    overlapping_zero_intervals = util.find_overlapping_intervals(
        util.compute_const_intervals(x, strength_threshold=1, time_threshold=20),
        util.compute_const_intervals(y, strength_threshold=1, time_threshold=20),
    )

    # possible gameplay signals
    # intervals complementary to overlapping_zero_intervals where there is at least 2 seconds (40fps) of gameplay
    possible_signal_intervals = util.find_complementary_intervals(x, overlapping_zero_intervals, min_length=2 * 40)

    delta_x = np.diff(x)
    delta_y = np.diff(y)

    correlation_x = np.convolve(delta_x, horizontal_kernel[::-1], mode='same')
    correlation_y = np.convolve(delta_y, vertical_kernel[::-1], mode='same')

    # bug catch. due to short length of df["X"] and df["Y"]
    if len(correlation_x) != len(correlation_y):
        return -1, -1

    mask = np.zeros_like(correlation_x, dtype=bool)
    for start, end in possible_signal_intervals:
        mask[start : end + 1] = True

    # zero out all positions not in any interval
    correlation_x[~mask] = 0
    correlation_y[~mask] = 0

    signal_intervals = util.find_complementary_intervals(correlation_x, overlapping_zero_intervals)
    for _i, (start, end) in enumerate(signal_intervals):
        if np.abs((end - start) - 50) < 20:
            dot = np.dot(correlation_x[start:end], correlation_y[start:end])
            if abs(dot) >= 30000:
                # likely a ball tracking bug
                correlation_x[start:end] = 0
                correlation_y[start:end] = 0

    best_guess = util.best_point_estimate(correlation_x, signal_intervals, minimum_point_length_threshold, corr_x_peak)
    if best_guess == -1:
        return -1, -1

    start, end = signal_intervals[best_guess]

    x_subset = x[start + 20 : end - 20]
    y_subset = y[start + 20 : end - 20]
    x_diff_subset = np.diff(x_subset)
    y_diff_subset = np.diff(y_subset)
    if np.count_nonzero((x_diff_subset == 0) & (x_subset[:-1] != 0) & (y_diff_subset == 0) & (y_subset[:-1] != 0)) > 30:
        return -1, -1

    return start, end


def process_clip(task):
    """Write video_second_clipped_only_tracknet/, metadata_only_tracknet/ and the clipped ball CSV for one clip."""
    clipped_root_dir, clip_name, start, end = task
    clipped_gameplay_dir = os.path.join(clipped_root_dir, clip_name)

    video_file = os.path.join(clipped_gameplay_dir, "video", f"{clip_name}.mp4")
    video_second_clipped_dir = os.path.join(clipped_gameplay_dir, "video_second_clipped_only_tracknet")

    if os.path.exists(video_second_clipped_dir):
        return  # skip if already processed

    # Create video clip
    os.makedirs(video_second_clipped_dir)
    output_file = os.path.join(video_second_clipped_dir, f"{clip_name}.mp4")
    completed = create_video_subset(video_file, output_file, start, end)
    if not completed:
        return

    # Save metadata
    metadata_dir = os.path.join(clipped_gameplay_dir, "metadata_only_tracknet")
    os.makedirs(metadata_dir, exist_ok=True)
    with open(os.path.join(metadata_dir, "second_clipped_indices.txt"), 'w') as file:
        file.write(f"{start}, {end}\n")
    # Create ball_tracker_only_tracknet clip
    csv_path = os.path.join(clipped_gameplay_dir, "ball_tracker_only_tracknet", f"{clip_name}_ball.csv")
    csv_df = pd.read_csv(csv_path, index_col="Frame")
    clipped_csv_df = csv_df.iloc[start : end + 1, :]  # i'm guessing start, end are inclusive??
    clipped_csv_df.index = range(0, len(clipped_csv_df))
    clipped_csv_path = os.path.join(
        clipped_gameplay_dir, "ball_tracker_only_tracknet", f"{clip_name}_second_clipped_only_tracknet_ball.csv"
    )
    clipped_csv_df.to_csv(clipped_csv_path, index_label="Frame")

    # Create ball_tracker_10 clip on _only_tracknet clip (legacy tracker; absent on clips that only ran only_tracknet_main.py)
    csv_path = os.path.join(clipped_gameplay_dir, "ball_tracker_10", f"{clip_name}_ball.csv")
    if not os.path.exists(csv_path):
        return
    csv_df = pd.read_csv(csv_path, index_col="Frame")
    clipped_csv_df = csv_df.iloc[start : end + 1, :]  # i'm guessing start, end are inclusive??
    clipped_csv_df.index = range(0, len(clipped_csv_df))
    clipped_csv_path = os.path.join(
        clipped_gameplay_dir, "ball_tracker_10", f"{clip_name}_second_clipped_only_tracknet_ball.csv"
    )
    clipped_csv_df.to_csv(clipped_csv_path, index_label="Frame")


def discover_tasks(root_dir, batch_indices):
    """Every clip with finished ball tracking and no video_second_clipped_only_tracknet/ yet."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.isdir(batch_dir):
            continue
        print(f"Going through {batch_dir}.")
        for clip_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, clip_name)
            video_second_clipped_only_tracknet_file = os.path.join(
                gameplay_dir, "video_second_clipped_only_tracknet", f"{clip_name}.mp4"
            )
            ball_tracker_only_tracknet_dir = os.path.join(gameplay_dir, "ball_tracker_only_tracknet")
            ball_tracker_only_tracknet_done_path = os.path.join(ball_tracker_only_tracknet_dir, "done.txt")
            if not os.path.exists(video_second_clipped_only_tracknet_file) and os.path.exists(
                ball_tracker_only_tracknet_done_path
            ):
                tasks.append((batch_dir, clip_name))

    print(f"Identified {len(tasks)} tasks.")
    return tasks


if __name__ == "__main__":
    """cd src/rally_clipper && python second_clipper_2.py --num-resources 20"""
    config = tyro.cli(ParallelConfig)

    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    tasks = config.slice_tasks(tasks)

    clips = []
    for _, (batch_dir, clip_name) in enumerate(tasks):
        gameplay_dir = os.path.join(batch_dir, clip_name)
        ball_tracker_only_tracknet_dir = os.path.join(gameplay_dir, "ball_tracker_only_tracknet")
        csv = os.path.join(ball_tracker_only_tracknet_dir, f"{clip_name}_ball.csv")
        try:
            df = pd.read_csv(csv, index_col="Frame")
        except Exception:
            print("Failed to read from csv file")
            continue
        start, end = second_clip_only_tracknet(df)
        if start == -1 and end == -1:
            continue
        clips.append((batch_dir, clip_name, start, end))

    print(f"New clips added: {len(clips)}")
    if tasks:
        print(f"Clipping efficiency rate: {len(clips) / len(tasks)}")

    run_parallel(clips, process_clip, config)
