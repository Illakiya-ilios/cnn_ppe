# PPE Safety Violation Detection System

Real-time detection of PPE (Personal Protective Equipment) non-compliance from
live camera footage, with on-screen pop-up alerts, snapshot capture, a CSV
violation log, and a JSON session report. Packaged for an enterprise proof of
concept: configurable, observable, stream-resilient, and container-ready.

Addresses the client problem statement *"PPE/safety Violation detection — every
equipment floor level to be equipped with a high resolution camera to identify
and announce if any person is roaming without PPE's... with a detailed report of
how many times safety norms are violated and by whom."*

Combines two reference projects:
- [vyasdeepti/PPE-Object-Detection-using-YOLO11](https://github.com/vyasdeepti/PPE-Object-Detection-using-YOLO11) — YOLO-based PPE detection
- [kemalkilicaslan/Industrial-Safety-Gear-Detection-System](https://github.com/kemalkilicaslan/Industrial-Safety-Gear-Detection-System) — clean per-person assignment / tracking / compliance pipeline

## Pipeline

```
[Camera / RTSP / Video]
      -> YOLO Detection          (persons + PPE)
      -> Equipment Assignment    (IoU + center-in-box)
      -> Person Tracking         (centroid tracker, persistent IDs)
      -> Compliance Assessment   (Compliant / Partial / Non-compliant)
      -> Alerting + Reporting     (debounced pop-up, snapshot, CSV, JSON)
      -> Annotated frame / output video
```

## Enterprise-POC features

- **Flexible config** — every setting is overridable via `PPE_*` environment
  variables or CLI flags; nothing needs code edits per site.
- **Structured logging** — levelled, timestamped logs to console and a rotating
  file (`output/ppe_system.log`).
- **Stream resilience** — automatic RTSP/CCTV reconnect with backoff; files stop
  cleanly at end-of-stream.
- **Reporting** — per-event CSV plus a JSON session summary (per-person counts,
  status breakdown, most-missing items, duration, frames processed).
- **Evidence capture** — a snapshot image is saved at the moment each alert fires.
- **Performance controls** — frame-stride processing and a live FPS overlay.

## Install

```bash
pip install -r requirements.txt
```

Python 3.8+. Dependencies: `ultralytics`, `opencv-python`, `numpy`.

## Quick start (with a real PPE model)

```bash
pip install -r requirements.txt
python download_model.py        # fetches a PPE-trained YOLOv11 model -> ppe_model.pt
python main.py --source 0       # webcam, real helmet/vest compliance
```

`download_model.py` pulls the YOLOv11 PPE model
[melihuzunoglu/ppe-detection](https://huggingface.co/melihuzunoglu/ppe-detection)
(classes: `helmet`, `human`, `no-helmet`, `vest`) and saves it as `ppe_model.pt`.
When that file exists, `config.py` uses it automatically with the matching class
setup; otherwise it falls back to stock `yolo11n.pt` (person-only).

## Run

```bash
python main.py                              # webcam, defaults from config.py
python main.py --source rtsp://cam/stream1  # IP / CCTV camera
python main.py --source clip.mp4 --no-preview
python main.py --model ppe_model.pt --conf 0.4 --device 0
```

Press `q` in the preview window (or Ctrl+C) to stop. Outputs land in `output/`.

### CLI flags

| Flag | Meaning |
|------|---------|
| `--source` | `0` (webcam), RTSP URL, or video file path |
| `--model` | Path to YOLO `.pt` weights |
| `--device` | `cpu`, `0`, `cuda:0`, ... |
| `--conf` | Detection confidence threshold |
| `--stride` | Process every Nth frame (default 3; perf tuning, see note) |
| `--output` | Annotated video path (`none` to disable) |
| `--no-preview` | Run headless (no window) |
| `--log-level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |

## Performance (CPU)

A heavy model like `yolo26m` runs ~3 fps per frame on CPU. The default
`FRAME_STRIDE=3` processes every 3rd frame, giving ~3x smoother throughput
(measured ~9.6 effective fps vs ~3.4 at stride 1) while still feeding the
tracker and temporal confirmation enough frames. Lower to `1` for maximum
accuracy on a GPU/fast machine; raise it if the preview still lags. The
violation/temporal thresholds are counted in *processed* frames, so they stay
consistent as you change the stride.

## Configuration

Defaults live in `config.py`; override any of them via `PPE_<NAME>` env vars
(e.g. `PPE_VIDEO_SOURCE`, `PPE_CONF_THRESH`, `PPE_REQUIRED_EQUIPMENT`). CLI flags
take highest precedence. Key settings: model + class names, video source,
thresholds, alert debounce/cooldown, stream reconnect behavior, output paths.

## Outputs

| File | Contents |
|------|----------|
| `output/violations.csv` | One row per confirmed violation (timestamp, person ID, status, missing/detected items) |
| `output/session_report.json` | Run summary: per-person counts, status breakdown, duration, frames |
| `output/snapshots/*.jpg` | Frame captured at each alert |
| `output/ppe-detection-output.mp4` | Annotated video |
| `output/ppe_system.log` | Application log |

## Models and classes

- **Default (recommended for the demo):** run `python download_model.py` to get
  the PPE-trained `ppe_model.pt`. `config.py` auto-detects it and configures
  `PERSON_CLASS=human`, `EQUIPMENT_CLASSES={helmet, vest}`, and the negative
  class `no-helmet -> helmet`.
- **Fallback:** if `ppe_model.pt` is missing, the app loads stock `yolo11n.pt`,
  which only detects `person` (every person reads as a violation) and logs a
  warning. Useful only to confirm the plumbing runs.
- **Your own model:** set `MODEL_PATH` (or `PPE_MODEL_PATH` / `--model`) and
  update `EQUIPMENT_CLASSES` / `REQUIRED_EQUIPMENT` / `PERSON_CLASS` to match.

### Negative classes

Some PPE models emit explicit "absence" detections (e.g. a `no-helmet` box over
a bare head), which is a stronger signal than inferring absence. Map them to the
required item they violate via `NEGATIVE_CLASSES`:

```
PPE_NEGATIVE_CLASSES="no-helmet:helmet,no-vest:vest"
```

A person assigned a negative detection is treated as missing that item.

## Training a better model (Construction-PPE, 11 classes)

The strongest model covers `helmet, gloves, vest, boots, goggles` plus explicit
`no_*` (missing) classes and `Person`. `config.py` auto-detects it via the
`construction-ppe` profile, so no code changes are needed once `ppe_model.pt` is
in place.

### Option 1 — train locally (CPU, slow)

```bash
python train_model.py --epochs 25            # from yolo11n (fastest)
python train_model.py --model yolo26m.pt --epochs 25   # stronger, much slower on CPU
```

The dataset (~178 MB) auto-downloads. Best weights are copied to `ppe_model.pt`.
CPU note: `yolo11n` ~8 min/epoch; `yolo26m` can be 40-80+ min/epoch.

### Option 2 — train on a free GPU, run locally (recommended)

Training is far faster on a GPU (`yolo26m`, 25 epochs ≈ 15-40 min vs. 1-3 days on
CPU). Inference then runs fine on your local CPU. A `.pt` trained on GPU runs
identically on CPU.

**Kaggle (free T4 GPU):**
1. Upload `kaggle_train.ipynb` to https://www.kaggle.com/code (or paste
   `kaggle_train.py`).
2. Settings -> Accelerator = **GPU**, Internet = **ON**.
3. Run all. The dataset auto-downloads; training runs on the GPU.
4. Download `ppe_model.pt` from the **Output** tab.
5. Locally: replace `./ppe_model.pt` with the downloaded file, then
   `python main.py --source 0`.

The notebook pins `ultralytics==8.4.170` (match your local version) so the
weights load without a version mismatch. If your local version differs, edit the
pin in cell 1.

## Project structure

```
.
├── main.py            # Entry point: CLI, video loop, HUD, shutdown
├── pipeline.py        # Per-frame compliance pipeline (reusable)
├── detector.py        # YOLO wrapper, equipment assignment, compliance logic
├── tracker.py         # Centroid-based multi-person tracker
├── alerts.py          # AlertManager + ViolationLogger + SessionReporter
├── video_source.py    # Robust capture with RTSP reconnect
├── logging_setup.py   # Centralized structured logging
├── config.py          # Settings with env/CLI override support
├── download_model.py  # Fetches a ready-to-use PPE model -> ppe_model.pt
├── requirements.txt
├── test_logic.py      # Unit tests: geometry + tracker (no model)
├── test_compliance.py # Unit tests: compliance incl. negative classes (no model)
└── test_pipeline.py   # End-to-end check (loads model)
```

## Tests

```bash
python test_logic.py        # geometry + tracker, no model
python test_compliance.py   # compliance + negative-class logic, no model
python test_pipeline.py     # loads model, exercises full wiring
```

## Roadmap (from the broader problem statements)

- **Named violator identification** (face recognition) — the report already
  keys on a per-person track ID as the placeholder to attach identity to.
- **Fall / accident detection** (pose-based).
- Dashboard / aggregated multi-camera reporting and notifications (email/SMS).

## Notes

- Pop-ups use `tkinter`, falling back to a native Windows message box, then to
  logged console output on headless systems.
- For research / authorized workplace safety monitoring. Ensure compliance with
  local privacy regulations for workplace video surveillance.
```
