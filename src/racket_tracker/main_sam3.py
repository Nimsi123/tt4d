"""Stage 12: detect and segment the rackets in every singles clip with SAM3 (racket_tracker_sam3/)."""

import csv
import os
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
import tyro
from PIL import Image

from rtt_utils.parallel import ParallelConfig, run_parallel
from rtt_utils.paths import load_file_paths, remove_incomplete_process_directory

# Default text prompt for racket detection
DEFAULT_RACKET_PROMPT = "racket"


def run_racket_tracking_one_video(
    video_file: str,
    processor,
    save_dir: str,
    output_video: bool = False,
    conf_threshold: float = 0.3,
    text_prompt: str = DEFAULT_RACKET_PROMPT,
):
    """
    Run SAM3 racket detection on a single video and save results.

    Args:
        video_file: Path to the input video file
        processor: SAM3 Processor instance
        save_dir: Directory to save results
        output_video: Whether to save annotated video
        conf_threshold: Confidence threshold for detection
        text_prompt: Text prompt for SAM3 (default: "table tennis racket")
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    clip_name = os.path.basename(video_file).replace('.mp4', '')

    # Open video
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_file}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video writer if needed
    out_video = None
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_video_path = os.path.join(save_dir, f"{clip_name}_annotated.mp4")
        out_video = cv2.VideoWriter(out_video_path, fourcc, fps, (width, height))

    # Prepare CSV output file
    csv_path = os.path.join(save_dir, f"{clip_name}_racket.csv")

    all_detections = []
    frame_idx = 0

    # Update processor confidence threshold
    processor.confidence_threshold = conf_threshold

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Run SAM3 inference
        inference_state = processor.set_image(pil_image)
        output = processor.set_text_prompt(prompt=text_prompt, state=inference_state)

        masks = output.get("masks", None)
        boxes = output.get("boxes", None)
        scores = output.get("scores", None)

        frame_detections = []
        if boxes is not None and len(boxes) > 0:
            for i in range(len(boxes)):
                box = boxes[i].cpu().numpy()
                score = scores[i].item() if scores is not None else 1.0

                # Extract mask for this detection if available
                mask = masks[i].cpu().numpy() if masks is not None and i < len(masks) else None

                x1, y1, x2, y2 = box
                frame_detections.append(
                    {
                        'class': 0,  # Single class: racket
                        'conf': score,
                        'x1': float(x1),
                        'y1': float(y1),
                        'x2': float(x2),
                        'y2': float(y2),
                        'mask': mask,
                    }
                )

        all_detections.append({'frame': frame_idx, 'detections': frame_detections})

        # Draw detections on frame if outputting video
        if output_video and out_video is not None:
            annotated_frame = frame.copy()
            for det in frame_detections:
                x1, y1, x2, y2 = int(det['x1']), int(det['y1']), int(det['x2']), int(det['y2'])
                conf = det['conf']

                # Draw bounding box
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                label = f"racket {conf:.2f}"
                (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_frame, (x1, y1 - label_h - baseline), (x1 + label_w, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                # Draw mask overlay if available
                if det['mask'] is not None:
                    mask = det['mask'].squeeze()
                    if mask.shape == (height, width):
                        mask_overlay = np.zeros_like(annotated_frame)
                        mask_overlay[:, :, 1] = (mask * 128).astype(np.uint8)  # Green tint
                        annotated_frame = cv2.addWeighted(annotated_frame, 1, mask_overlay, 0.3, 0)

            out_video.write(annotated_frame)

        frame_idx += 1

    cap.release()
    if out_video is not None:
        out_video.release()

    # Save detections to CSV (same format as YOLO-based tracker)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        # Header
        writer.writerow(['Frame', 'NumDetections', 'Class', 'Confidence', 'X1', 'Y1', 'X2', 'Y2'])

        for frame_data in all_detections:
            frame = frame_data['frame']
            detections = frame_data['detections']

            if len(detections) == 0:
                # No detections for this frame
                writer.writerow([frame, 0, '', '', '', '', '', ''])
            else:
                for i, det in enumerate(detections):
                    writer.writerow(
                        [
                            frame,
                            len(detections) if i == 0 else '',  # Only write num_detections for first detection
                            det['class'],
                            f"{det['conf']:.4f}",
                            f"{det['x1']:.2f}",
                            f"{det['y1']:.2f}",
                            f"{det['x2']:.2f}",
                            f"{det['y2']:.2f}",
                        ]
                    )

    # Save masks to numpy file for later use (optional)
    masks_path = os.path.join(save_dir, f"{clip_name}_masks.npz")
    masks_data = {}
    for frame_data in all_detections:
        frame = frame_data['frame']
        detections = frame_data['detections']
        for i, det in enumerate(detections):
            if det['mask'] is not None:
                masks_data[f"frame_{frame}_det_{i}"] = det['mask']

    if masks_data:
        np.savez_compressed(masks_path, **masks_data)

    print(f"Saved racket detections to {csv_path}")


def start_racket_tracking_task(
    video_file: str,
    processor,
    save_dir: str,
    output_video: bool = False,
    text_prompt: str = DEFAULT_RACKET_PROMPT,
):
    """
    Run racket tracking inference for a single video.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    run_racket_tracking_one_video(video_file, processor, save_dir, output_video, text_prompt=text_prompt)

    # Mark completion
    with open(os.path.join(save_dir, "done.txt"), "w") as _:
        pass


@dataclass
class Sam3Config(ParallelConfig):
    text_prompt: str = DEFAULT_RACKET_PROMPT
    """Text prompt for SAM3 detection."""


def load_sam3(gpu_id: int):
    """Load the SAM3 image model once per worker (the worker sees only its GPU as "cuda")."""
    # Importing sam3 switches TF32 on; the dataset was made with that switch disabled, so keep PyTorch's settings.
    tf32_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = tf32_flags

    print(f"[Worker for GPU {gpu_id}] Loading SAM3 model onto GPU {gpu_id} ...")

    model = build_sam3_image_model(
        device="cuda",
        eval_mode=True,
        load_from_HF=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    model = model.to("cuda")  # Ensure model is on the correct device
    processor = Sam3Processor(model, device="cuda")

    print(f"[Worker for GPU {gpu_id}] Model loaded. Waiting for tasks...")
    return processor


def run_task(task, processor):
    """Detect rackets in one clip's second-clipped video and mark the output directory done."""
    gameplay_dir, text_prompt = task
    hash_id = os.path.basename(gameplay_dir)
    video_file = os.path.join(gameplay_dir, "video_second_clipped_only_tracknet", f"{hash_id}.mp4")
    save_dir = os.path.join(gameplay_dir, "racket_tracker_sam3")
    output_video = True

    start_racket_tracking_task(video_file, processor, save_dir, output_video, text_prompt=text_prompt)


def discover_tasks(root_dir: str, batch_indices):
    """Every singles clip with a second-clipped video and no racket_tracker_sam3/done.txt."""
    tasks = []
    for i in batch_indices:
        batch_dir = os.path.join(root_dir, f"batch_clipped_long_{i}")
        if not os.path.exists(batch_dir):
            continue
        print(f"Going through {batch_dir}.")
        for clip_name in os.listdir(batch_dir):
            gameplay_dir = os.path.join(batch_dir, clip_name)
            paths = load_file_paths(gameplay_dir)

            video_file = paths["video_second_clipped_only_tracknet_file"]
            singles_flag = paths["singles_flag"]
            save_dir = os.path.join(gameplay_dir, "racket_tracker_sam3")
            done_path = os.path.join(save_dir, "done.txt")

            if not os.path.exists(singles_flag):
                continue

            if os.path.isfile(video_file) and not os.path.exists(done_path):
                tasks.append(gameplay_dir)

    tasks = sorted(tasks)
    print(f"Total tasks: {len(tasks)}")
    return tasks


if __name__ == "__main__":
    """
    python main_sam3.py --resource-ids 0 1 2 3 --workers-per-resource 1 --num-workers 1 --worker-split 0 --batch-start 1 --batch-end 52
    """
    mp.set_start_method("spawn", force=True)
    config = tyro.cli(Sam3Config)
    tasks = discover_tasks(config.get_root_dir(), config.get_batch_indices())
    if config.worker_split == 0:
        for gameplay_dir in tasks:
            remove_incomplete_process_directory(os.path.join(gameplay_dir, "racket_tracker_sam3"))
    tasks = config.slice_tasks(tasks)
    print(f"Text prompt             : {config.text_prompt}")
    run_parallel([(g, config.text_prompt) for g in tasks], run_task, config, setup=load_sam3)
