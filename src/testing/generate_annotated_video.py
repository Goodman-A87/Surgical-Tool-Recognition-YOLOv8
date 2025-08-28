# In src/testing/generate_annotated_video.py

from ultralytics import YOLO
from pathlib import Path
import cv2
import argparse
from tqdm import tqdm


def process_and_save_video(video_path_str: str, confidence_threshold: float):
    """
    Loads the champion model, processes a video frame-by-frame, and saves
    the annotated result to a new file without displaying a live window.

    Args:
        video_path_str (str): The path to the video file to process.
        confidence_threshold (float): The minimum confidence score for a detection.
    """
    # 1. Configuration and Path Setup
    project_root = Path(__file__).resolve().parent.parent.parent
    model_path = (
        project_root
        / "runs"
        / "tournament"
        / "yolov8l_50epochs"
        / "weights"
        / "best.pt"
    )
    input_video_path = Path(video_path_str)

    output_folder = project_root / "results"
    output_folder.mkdir(exist_ok=True)
    output_video_path = output_folder / f"{input_video_path.stem}_annotated.mp4"

    # --- Safety Checks ---
    if not model_path.exists():
        print(f"❌ ERROR: Champion model not found at {model_path}")
        return
    if not input_video_path.exists():
        print(f"❌ ERROR: Input video not found at {input_video_path}")
        return

    # 2. Load the Champion Model
    print(f"✅ Loading champion model: {model_path.name}")
    model = YOLO(model_path)

    # 3. Setup Video Capture and Writer
    cap = cv2.VideoCapture(str(input_video_path))
    if not cap.isOpened():
        print(f"❌ ERROR: Could not open video file.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(
        str(output_video_path), fourcc, fps, (frame_width, frame_height)
    )

    print(f"✅ Processing video: {input_video_path.name} ({total_frames} frames)")

    # 4. Process Video Frame-by-Frame with a Progress Bar
    for _ in tqdm(range(total_frames), desc="Annotating video"):
        ret, frame = cap.read()
        if not ret:
            break

        # Run prediction
        results = model.predict(frame, conf=confidence_threshold, verbose=False)
        result = results[0]

        # Draw boxes and labels on the frame
        for box in result.boxes:
            coords = [int(x) for x in box.xyxy[0]]
            x1, y1, x2, y2 = coords
            conf = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            cv2.rectangle(frame, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Write the annotated frame to the output video
        out.write(frame)

    # 5. Cleanup
    cap.release()
    out.release()

    print("\n--- Processing Complete ---")
    print(f"✅ Annotated video saved to: {output_video_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an annotated video with tool detections."
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (e.g., 'data/raw/CholecTrack20/Testing/VID01/VID01.mp4').",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold for detection (e.g., 0.5 for 50%).",
    )

    args = parser.parse_args()
    process_and_save_video(video_path_str=args.video, confidence_threshold=args.conf)
