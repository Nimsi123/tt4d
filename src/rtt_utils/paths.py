"""Repository root and the layout of one gameplay (clip) directory."""

import os
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def get_home_path():
    """Return the repository root as a string (the directory that holds src/ and pipeline_outputs/)."""
    return str(REPO_ROOT)


def remove_extraneous_process_items(items_to_remove):
    """Delete each file or directory in the list that exists."""
    for item in items_to_remove:
        if os.path.exists(item):
            try:
                if os.path.isfile(item) or os.path.islink(item):
                    os.remove(item)  # Remove file or symbolic link
                    print(f"Removed file: {item}")
                elif os.path.isdir(item):
                    shutil.rmtree(item)
                    print(f"Removed directory: {item}")
            except Exception as e:
                print(f"Error removing {item}: {e}")


def remove_incomplete_process_directory(folder_to_remove):
    """Delete a stage's output directory that has no done.txt, so the stage starts it from scratch."""
    assert not os.path.isfile(os.path.join(folder_to_remove, "done.txt"))
    if os.path.exists(folder_to_remove):
        shutil.rmtree(folder_to_remove)


def detail_from_clip_name(clip_name):
    """Split '<11-char video hash>_<start>_<end>_<score>' into (hash, start, end, score)."""
    hash = clip_name[0:11]
    start = clip_name[12:].split("_")[0]
    end = clip_name[12:].split("_")[1]
    score = "_".join(clip_name[12:].split("_")[2:])
    return hash, int(start), int(end), score


def load_file_paths(gameplay_dir):
    """Return a dict of every file and directory the pipeline stages read or write inside a gameplay directory."""
    clip_name = gameplay_dir.split("/")[-1]

    # Video directories and files
    video_dir = os.path.join(gameplay_dir, "video")
    video_file = os.path.join(video_dir, f"{clip_name}.mp4")

    video_second_clipped_dir = os.path.join(gameplay_dir, "video_second_clipped")
    video_second_clipped_file = os.path.join(video_second_clipped_dir, f"{clip_name}.mp4")

    video_second_clipped_only_tracknet_dir = os.path.join(gameplay_dir, "video_second_clipped_only_tracknet")
    video_second_clipped_only_tracknet_file = os.path.join(video_second_clipped_only_tracknet_dir, f"{clip_name}.mp4")

    # Calibration
    calib_dir = os.path.join(gameplay_dir, "thomas_calibration")
    calib_file = os.path.join(calib_dir, "calib.yaml")
    calib_moves_file = os.path.join(calib_dir, "moves.txt")
    calib_failed_file = os.path.join(calib_dir, "failed.txt")
    calib_done_path = os.path.join(calib_dir, "done.txt")

    # Ball tracker (with inpaintnet)
    ball_tracker_10_dir = os.path.join(gameplay_dir, "ball_tracker_10")
    ball_tracker_10_done_path = os.path.join(ball_tracker_10_dir, "done.txt")
    ball_tracker_10_file = os.path.join(ball_tracker_10_dir, f"{clip_name}_ball.csv")

    # Ball tracker (only tracknet)
    ball_tracker_only_tracknet_dir = os.path.join(gameplay_dir, "ball_tracker_only_tracknet")
    ball_tracker_only_tracknet_done_path = os.path.join(ball_tracker_only_tracknet_dir, "done.txt")
    ball_tracker_only_tracknet_file = os.path.join(ball_tracker_only_tracknet_dir, f"{clip_name}_ball.csv")

    human_pose_tracker_dir = os.path.join(gameplay_dir, "human_pose_tracker", "4DHumans")
    human_pose_tracker_done_path = os.path.join(human_pose_tracker_dir, "done.txt")
    human_pose_tracker_only_tracknet_dir = os.path.join(gameplay_dir, "human_pose_tracker_only_tracknet", "4DHumans")
    human_pose_tracker_only_tracknet_done_path = os.path.join(human_pose_tracker_only_tracknet_dir, "done.txt")

    # Metadata directories
    metadata_dir = os.path.join(gameplay_dir, "metadata")
    metadata_second_clipped_indices_file = os.path.join(metadata_dir, "second_clipped_indices.txt")
    metadata_only_tracknet_dir = os.path.join(gameplay_dir, "metadata_only_tracknet")
    metadata_only_tracknet_second_clipped_indices_file = os.path.join(
        metadata_only_tracknet_dir, "second_clipped_indices.txt"
    )

    # Video thinning
    video_thinning_dir = os.path.join(gameplay_dir, "video_thinning")
    video_thinning_frames_to_drop_file = os.path.join(video_thinning_dir, "frames_to_drop.json")

    # Gameplay type flags (singles vs doubles)
    singles_flag = os.path.join(metadata_dir, "singles.txt")
    doubles_flag = os.path.join(metadata_dir, "doubles.txt")
    # Humans-mesh-based gameplay type flags (written by game_type_distinguish_humans.py)
    singles_humans_flag = os.path.join(metadata_dir, "singles_humans.txt")
    doubles_humans_flag = os.path.join(metadata_dir, "doubles_humans.txt")
    humans_failed_flag = os.path.join(metadata_dir, "humans_failed.txt")

    recon_file = os.path.join(gameplay_dir, "nima_recons", f"{clip_name}_tt4d_processed.pkl")

    return dict(
        clip_name=clip_name,
        gameplay_dir=gameplay_dir,
        # Video
        video_dir=video_dir,
        video_file=video_file,
        video_second_clipped_dir=video_second_clipped_dir,
        video_second_clipped_file=video_second_clipped_file,
        video_second_clipped_only_tracknet_dir=video_second_clipped_only_tracknet_dir,
        video_second_clipped_only_tracknet_file=video_second_clipped_only_tracknet_file,
        # Calibration
        calib_dir=calib_dir,
        calib_file=calib_file,
        calib_moves_file=calib_moves_file,
        calib_failed_file=calib_failed_file,
        calib_done_path=calib_done_path,
        # Ball tracker
        ball_tracker_10_dir=ball_tracker_10_dir,
        ball_tracker_10_file=ball_tracker_10_file,
        ball_tracker_10_done_path=ball_tracker_10_done_path,
        ball_tracker_only_tracknet_dir=ball_tracker_only_tracknet_dir,
        ball_tracker_only_tracknet_file=ball_tracker_only_tracknet_file,
        ball_tracker_only_tracknet_done_path=ball_tracker_only_tracknet_done_path,
        # Human pose tracker
        human_pose_tracker_dir=human_pose_tracker_dir,
        human_pose_tracker_done_path=human_pose_tracker_done_path,
        human_pose_tracker_only_tracknet_dir=human_pose_tracker_only_tracknet_dir,
        human_pose_tracker_only_tracknet_done_path=human_pose_tracker_only_tracknet_done_path,
        # Metadata
        metadata_dir=metadata_dir,
        metadata_second_clipped_indices_file=metadata_second_clipped_indices_file,
        metadata_only_tracknet_dir=metadata_only_tracknet_dir,
        metadata_only_tracknet_second_clipped_indices_file=metadata_only_tracknet_second_clipped_indices_file,
        # Video thinning
        video_thinning_dir=video_thinning_dir,
        video_thinning_frames_to_drop_file=video_thinning_frames_to_drop_file,
        # Gameplay type flags
        singles_flag=singles_flag,
        doubles_flag=doubles_flag,
        singles_humans_flag=singles_humans_flag,
        doubles_humans_flag=doubles_humans_flag,
        humans_failed_flag=humans_failed_flag,
        # Reconstruction
        recon_file=recon_file,
    )


def is_singles(gameplay_dir):
    """True if the scoreboard OCR labeled the clip as singles."""
    return os.path.exists(load_file_paths(gameplay_dir)["singles_flag"])


def is_doubles(gameplay_dir):
    """True if the scoreboard OCR labeled the clip as doubles."""
    return os.path.exists(load_file_paths(gameplay_dir)["doubles_flag"])


def is_singles_humans(gameplay_dir):
    """True if the 4DHumans detections labeled the clip as singles."""
    return os.path.exists(load_file_paths(gameplay_dir)["singles_humans_flag"])


def is_doubles_humans(gameplay_dir):
    """True if the 4DHumans detections labeled the clip as doubles."""
    return os.path.exists(load_file_paths(gameplay_dir)["doubles_humans_flag"])
