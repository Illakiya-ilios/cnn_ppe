"""
Download a ready-to-use PPE-trained YOLO model for the demo.

Fetches the YOLOv11 PPE model from Hugging Face
(melihuzunoglu/ppe-detection) and copies it to ./ppe_model.pt in the project
so config.py can reference it directly.

Usage:
    python download_model.py

Classes in this model: helmet, human, no-helmet, vest
Configure the app for it (already the defaults in config.py when
MODEL_PATH = "ppe_model.pt"):
    PERSON_CLASS       = human
    EQUIPMENT_CLASSES  = helmet, vest
    REQUIRED_EQUIPMENT = helmet, vest
    NEGATIVE_CLASSES   = no-helmet:helmet
"""

import shutil
import sys

DEST = "ppe_model.pt"
REPO = "melihuzunoglu/ppe-detection"
FILENAME = "best.pt"


def main():
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. Install with:")
        print("    pip install huggingface_hub")
        sys.exit(1)

    print(f"Downloading {REPO}/{FILENAME} ...")
    cached = hf_hub_download(repo_id=REPO, filename=FILENAME)
    shutil.copyfile(cached, DEST)
    print(f"Saved model to ./{DEST}")

    # Report the class names so the operator can confirm the config matches.
    try:
        from ultralytics import YOLO
        names = YOLO(DEST).names
        print("Model classes:", names)
    except Exception as exc:  # noqa: BLE001
        print(f"(Could not introspect classes: {exc})")

    print(
        "\nReady. Run the demo with:\n"
        "    python main.py --source 0            # webcam\n"
        "    python main.py --source clip.mp4     # a video file\n"
    )


if __name__ == "__main__":
    main()
