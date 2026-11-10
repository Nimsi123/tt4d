"""Iterate over the frames of a video file."""

import cv2


class VideoFrameIterator:
    """Yields the BGR frames of a video in order and releases the capture at the end."""

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError("Error: Could not open video file.")

    def __iter__(self):
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.release()
            raise StopIteration
        return frame
