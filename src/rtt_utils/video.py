"""Video reading, clipping and overlay helpers (ffmpeg/ffprobe and OpenCV)."""

import subprocess
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw


def get_video_fps(video_path):
    """Return (fps, numerator, denominator) from the stream's r_frame_rate, or (-1, -1, -1) if ffprobe fails."""
    cmd = f'ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of default=noprint_wrappers=1 "{video_path}"'
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

    fps_str = result.stdout.strip().split('=')[-1]
    try:
        numerator, denominator = map(int, fps_str.split('/'))
    except (TypeError, ValueError):
        print("ERROR: ", repr(fps_str))
        return -1, -1, -1
    fps = numerator / denominator

    return fps, numerator, denominator


def num_frames_in_video(video_path):
    """Count the frames by decoding the whole video (the container's frame count is not trusted)."""
    cap = cv2.VideoCapture(video_path)
    counter = 0
    ret, frame = cap.read()
    while ret:
        counter += 1
        ret, frame = cap.read()
    return counter


def get_frames_from_video(video_path):
    """Decode every frame of the video into a list of BGR arrays."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    while ret:
        frames.append(frame)
        ret, frame = cap.read()
    return frames


def create_video_subset(video_path, output_path, start_frame, end_frame):
    """Cut frames [start_frame, end_frame] into output_path with a stream copy; False if the fps is unreadable."""
    fps, _, _ = get_video_fps(video_path)
    if fps == -1:
        return False
    start_time = start_frame / fps
    end_time = end_frame / fps

    start_time_str = f"{int(start_time // 3600):02}:{int((start_time % 3600) // 60):02}:{int(start_time % 60):02}.{int((start_time % 1) * 1000):03}"
    end_time_str = f"{int(end_time // 3600):02}:{int((end_time % 3600) // 60):02}:{int(end_time % 60):02}.{int((end_time % 1) * 1000):03}"

    cmd = ["ffmpeg", "-ss", start_time_str, "-to", end_time_str, "-i", video_path, "-c", "copy", output_path]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return True


def draw_traj(img, traj, radius=3, color='red'):
    """Draw the ball positions in `traj` (a deque of [x, y] or None) as circles on a BGR frame."""
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)

    for i in range(len(traj)):
        if traj[i] is not None:
            draw_x = traj[i][0]
            draw_y = traj[i][1]
            bbox = (draw_x - radius, draw_y - radius, draw_x + radius, draw_y + radius)
            draw = ImageDraw.Draw(img)
            draw.ellipse(bbox, fill='rgb(255,255,255)', outline=color)
            del draw
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    return img


def overlay_ball_on_video(video_file, save_file, frame_list_path, color='yellow'):
    """
    Given a video file and a CSV of (Frame,Visibility,X,Y),
    overlay the ball trajectory (rolling queue) on the video and save the result.
    """

    assert video_file != save_file

    traj_len = 8  # how many past positions to remember in the rolling queue

    if isinstance(frame_list_path, str):
        # --- Read CSV file to get each frame's predicted (x, y) and visibility.
        x_pred = []
        y_pred = []
        vis_pred = []
        with open(frame_list_path) as f:
            # Skip header line (Frame,Visibility,X,Y)
            next(f)
            for line in f:
                # Each line: 0,1,608,289
                frame_idx_str, vis_str, x_str, y_str = line.strip().split(',')
                x_pred.append(int(x_str))
                y_pred.append(int(y_str))
                vis_pred.append(bool(int(vis_str)))
    else:
        vis_pred, x_pred, y_pred = frame_list_path[:, 1], frame_list_path[:, 2], frame_list_path[:, 3]

    # --- Open the video capture
    cap = cv2.VideoCapture(video_file)
    # Retrieve fps and frame shape from the capture
    fps = cap.get(cv2.CAP_PROP_FPS)  # or int(...) if you want an integer
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shape = (width, height)  # for cv2.VideoWriter, shape is (width, height)

    # --- Prepare the output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_file, fourcc, fps, shape)

    # --- Create a queue (deque) for storing up to 'traj_len' points of the trajectory
    pred_queue = deque()

    # --- Read frames from the video, overlay the ball if visible, and draw trajectory
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # no more frames or error reading

        # Check if we're beyond the CSV (e.g., if the video is longer than CSV data)
        if idx >= len(x_pred):
            break

        # Maintain a rolling queue of length <= traj_len
        if len(pred_queue) >= traj_len:
            pred_queue.pop()  # remove oldest from the right

        # Push new ball coordinates at the left if visible, else None
        if vis_pred[idx]:
            pred_queue.appendleft([x_pred[idx], y_pred[idx]])
        else:
            pred_queue.appendleft(None)

        # Draw the rolling trajectory on the current frame
        frame = draw_traj(frame, pred_queue, color=color)

        out.write(frame)
        idx += 1

    out.release()
    cap.release()
