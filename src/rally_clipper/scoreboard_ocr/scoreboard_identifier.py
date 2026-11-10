"""Find the scoreboard in a frame with a fine-tuned YOLOv5 and read it with PaddleOCR."""

import ssl

# PaddleOCR downloads its weights at import time; the cluster's certificates fail verification.
ssl._create_default_https_context = ssl._create_unverified_context

import cv2  # noqa: E402
import torch  # noqa: E402
from paddleocr import PaddleOCR  # noqa: E402


class TextIdentifier:
    """Scoreboard detector (YOLOv5, scoreboard_model.pt in this directory) plus OCR of the crop."""

    def __init__(self):

        self.ocr = PaddleOCR(
            use_angle_cls=True, lang='en'
        )  # need to run only once to download and load model into memory

        model = torch.hub.load('ultralytics/yolov5', 'custom', path='scoreboard_model.pt')
        model.to(torch.device("cuda")).eval()
        self.model = model

    ## returns text in image with easyocr api
    def call_ocr(self, frame):
        valid_board = self.identify_and_crop_score(frame)
        if valid_board is None:
            return False, "Board not found, skip frame"

        preprocessed_image = self.ocr_image_preprocessing(valid_board)
        text_result = self.ocr.ocr(preprocessed_image, cls=True)
        # print(text_result)
        return True, text_result

    ## crops image using yolo model
    def identify_and_crop_score(self, frame):
        results = self.model(frame)
        if results.pandas().xyxy[0].empty:
            return None

        xmin, ymin, xmax, ymax = results.pandas().xyxy[0].iloc[0][['xmin', 'ymin', 'xmax', 'ymax']]
        xmin, ymin, xmax, ymax = map(int, [xmin, ymin, xmax, ymax])
        cropped_image = frame[ymin:ymax, xmin : xmax + 2]
        return cropped_image

    def ocr_image_preprocessing(self, frame):

        frame = cv2.resize(frame, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_CUBIC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale_factor = 3  # Magnify by 2x

        # Get new dimensions
        height, width = frame.shape[:2]
        new_dimensions = (int(width * scale_factor), int(height * scale_factor))

        # Resize the image
        magnified_image = cv2.resize(frame, new_dimensions, interpolation=cv2.INTER_LINEAR)

        # crop the noise from top and bottom
        height, width = magnified_image.shape[:2]
        top_crop = height // 10  # 1/8th of the height
        bottom_crop = height - top_crop
        magnified_image = magnified_image[top_crop:bottom_crop, :]

        # invert image since paddleocr trained dataset doesnt include numbers with black background
        inverted_image = cv2.bitwise_not(magnified_image)

        return inverted_image
