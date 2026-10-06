"""
Train a PPE detection model on the Ultralytics Construction-PPE dataset.

Produces a model with 11 classes covering helmet, gloves, vest, boots, goggles
plus explicit "no_*" (missing) classes and Person -- a much closer fit to the
full factory PPE requirement than the helmet+vest-only model.

The dataset (~178 MB) downloads automatically on first run.

Usage:
    python train_model.py                     # CPU-friendly defaults
    python train_model.py --epochs 50         # more epochs = better accuracy
    python train_model.py --model yolo11s.pt  # larger (slower, more accurate)
    python train_model.py --resume            # continue an interrupted run

On CPU this is slow. Start with the default short run to get a working demo
model, then train longer when you can (ideally on a GPU machine).

After training, the best weights are copied to ./ppe_model.pt and the class
names are printed so you can confirm config.py matches.
"""

import argparse
import shutil
import sys

DEST = "ppe_model.pt"


def parse_args():
    p = argparse.ArgumentParser(description="Train Construction-PPE model")
    p.add_argument("--model", default="yolo11n.pt",
                   help="Base model to fine-tune (yolo11n.pt is fastest)")
    p.add_argument("--epochs", type=int, default=25,
                   help="Training epochs (CPU: keep modest; more = better)")
    p.add_argument("--imgsz", type=int, default=640, help="Training image size")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--device", default="cpu", help="cpu, 0, cuda:0, ...")
    p.add_argument("--name", default=None,
                   help="Run name (defaults to ppe_<base-model-stem>)")
    p.add_argument("--cache", action="store_true",
                   help="Cache images in RAM to speed up each epoch")
    p.add_argument("--resume", action="store_true", help="Resume last run")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics is required: pip install -r requirements.txt")
        sys.exit(1)

    import os
    run_name = args.name or f"ppe_{os.path.splitext(os.path.basename(args.model))[0]}"

    print(f"Training {args.model} on Construction-PPE "
          f"(epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch}, "
          f"device={args.device}, run={run_name})")
    print("The dataset (~178 MB) will download automatically on first run.\n")

    model = YOLO(args.model)
    results = model.train(
        data="construction-ppe.yaml",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=run_name,
        cache=args.cache,
        resume=args.resume,
        patience=10,          # early-stop if no improvement
        plots=True,
    )

    # Locate and copy the best weights to ./ppe_model.pt
    try:
        best = model.trainer.best  # path to best.pt
        shutil.copyfile(best, DEST)
        print(f"\nBest weights copied to ./{DEST}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nCould not auto-copy best weights ({exc}).")
        print("Find them under runs/detect/<name>/weights/best.pt and copy "
              f"to ./{DEST} manually.")

    try:
        print("Model classes:", YOLO(DEST).names)
    except Exception:
        pass

    print(
        "\nDone. config.py is already set up for these classes when "
        f"{DEST} is present.\n"
        "Run the demo with:  python main.py --source 0\n"
    )


if __name__ == "__main__":
    main()
