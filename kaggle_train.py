"""
Kaggle GPU training for the Construction-PPE model.
=====================================================

Train on Kaggle's free GPU, then download best.pt and drop it into your local
project as ppe_model.pt. The app auto-detects the 11-class model (no code edits).

HOW TO USE ON KAGGLE
--------------------
1. Go to https://www.kaggle.com/code -> "New Notebook".
2. In the right sidebar: Settings -> Accelerator -> "GPU T4 x2" (or any GPU).
   Also set "Internet" -> ON (needed to install packages / download dataset).
3. Copy the CELLS below (each marked "# === CELL n ===") into separate
   notebook cells, or paste this whole file into one cell and run it.
4. Run all. Training ~15-40 min on GPU for 25 epochs.
5. When done, the best weights are zipped to /kaggle/working/ppe_model.pt and
   also shown in the Output tab -> download it.
6. Locally: replace ./ppe_model.pt with the downloaded file, then
   run:  python main.py --source 0

NOTE: The ultralytics version is pinned to match your local install so the
weights load without a version mismatch.
"""

# === CELL 1: install (pinned to your local version) ===
# Pin to the version you run locally so the .pt loads cleanly.
# Your local version is 8.4.170 (change if yours differs).
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.4.170"], check=True)

import ultralytics, torch
print("ultralytics:", ultralytics.__version__)
print("CUDA available:", torch.cuda.is_available(),
      "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")


# === CELL 2: train ===
from ultralytics import YOLO

# Base model: yolo26m is strong at people + PPE. On GPU this is fast.
# Drop to yolo11m.pt if yolo26m isn't available in this version.
BASE_MODEL = "yolo26m.pt"
EPOCHS = 25
IMGSZ = 640
BATCH = 16          # GPU can handle a bigger batch than CPU

model = YOLO(BASE_MODEL)
results = model.train(
    data="construction-ppe.yaml",   # auto-downloads (~178 MB) on Kaggle
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=0,                        # use the GPU
    name="ppe_construction_gpu",
    patience=15,
    plots=True,
    cache=True,                      # cache images in RAM (GPU boxes have plenty)
)


# === CELL 3: validate + show metrics ===
metrics = model.val()
print("mAP50:   ", round(float(metrics.box.map50), 4))
print("mAP50-95:", round(float(metrics.box.map), 4))
print("Classes: ", model.names)


# === CELL 4: export weights for local download ===
import shutil, os
best = model.trainer.best  # path to best.pt
out = "/kaggle/working/ppe_model.pt"
shutil.copyfile(best, out)
print("Saved:", out, f"({os.path.getsize(out)/1e6:.1f} MB)")
print("Download this file from the Kaggle 'Output' tab (right panel),")
print("then place it in your project as ppe_model.pt and run: python main.py --source 0")
