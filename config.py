"""
Configuration for the PPE Safety Violation Detection System.

Values are resolved with the following precedence (highest first):
    1. CLI arguments (see settings.py / main.py --help)
    2. Environment variables (prefix PPE_, e.g. PPE_VIDEO_SOURCE=rtsp://...)
    3. The defaults declared in this file

Keeping the defaults here means the app runs with zero configuration, while
operators can override any value per-site without editing code.
"""

import os


def _env(name, default, cast=str):
    """Read PPE_<NAME> from the environment, cast it, else return default."""
    raw = os.environ.get(f"PPE_{name}")
    if raw is None:
        return default
    try:
        if cast is bool:
            return raw.strip().lower() in ("1", "true", "yes", "on")
        return cast(raw)
    except (ValueError, TypeError):
        return default


def _env_set(name, default):
    """Read a comma-separated set from PPE_<NAME>, else return default."""
    raw = os.environ.get(f"PPE_{name}")
    if raw is None:
        return set(default)
    return {item.strip() for item in raw.split(",") if item.strip()}


def _coerce_source(value):
    """Webcam indices arrive as strings from env/CLI; turn '0' into int 0."""
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
# Defaults target the PPE-trained YOLOv11 model from Hugging Face
# (melihuzunoglu/ppe-detection), fetched to ./ppe_model.pt via download_model.py.
# Its classes are: helmet, human, no-helmet, vest.
#
# If ppe_model.pt is absent, the app falls back to the stock "yolo11n.pt"
# (person detection only, every person reads as a violation) so it still runs.
def _env_map(name, default):
    """Read a 'k:v,k:v' map from PPE_<NAME>, else return default."""
    raw = os.environ.get(f"PPE_{name}")
    if raw is None:
        return dict(default)
    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if ":" in pair:
            k, v = pair.split(":", 1)
            out[k.strip()] = v.strip()
    return out


# ---------------------------------------------------------------------------
# Model profiles
# ---------------------------------------------------------------------------
# Each profile declares the class vocabulary for a given model so the detector
# can auto-select the right one by inspecting the loaded model's class names.
# This avoids guessing from the filename and makes swapping models painless.
MODEL_PROFILES = {
    # Ultralytics Construction-PPE (11 classes) -- trained via train_model.py.
    "construction-ppe": {
        "person_class": "Person",
        "equipment_classes": {"helmet", "gloves", "vest", "boots", "goggles"},
        "required_equipment": {"helmet", "vest"},   # start strict-but-simple
        "negative_classes": {
            "no_helmet": "helmet",
            "no_gloves": "gloves",
            "no_boots": "boots",
            "no_goggle": "goggles",
        },
        # Signature class names used to recognize this model at load time.
        "signature": {"no_helmet", "goggles", "boots", "Person"},
    },
    # melihuzunoglu/ppe-detection (helmet + vest only).
    "helmet-vest": {
        "person_class": "human",
        "equipment_classes": {"helmet", "vest"},
        "required_equipment": {"helmet", "vest"},
        "negative_classes": {"no-helmet": "helmet"},
        "signature": {"human", "no-helmet"},
    },
    # Stock COCO model fallback (person only, no PPE).
    "coco": {
        "person_class": "person",
        "equipment_classes": set(),
        "required_equipment": set(),
        "negative_classes": {},
        "signature": {"person"},
    },
}

_HAS_PPE_MODEL = os.path.exists("ppe_model.pt")
MODEL_PATH = _env("MODEL_PATH", "ppe_model.pt" if _HAS_PPE_MODEL else "yolo11n.pt")

# Optional dedicated PERSON-detection model (two-model pipeline).
# The PPE model's own "person" class can be weak. Set this to a strong stock
# detector (e.g. "yolo26m.pt" / "yolo11m.pt") to detect PEOPLE reliably, while
# MODEL_PATH is used only for PPE/equipment. Set to None/"" to use a single
# model for both. The person class in this model is assumed to be COCO "person".
# Default OFF (single-model). Enable only if you want a separate person model:
#   set PPE_PERSON_MODEL_PATH=yolo26m.pt
PERSON_MODEL_PATH = _env("PERSON_MODEL_PATH", "")
PERSON_MODEL_CLASS = _env("PERSON_MODEL_CLASS", "person")

# Default class config. These are overridden automatically by the detector when
# it recognizes a known model profile (see PPEDetector), unless the operator
# sets the corresponding PPE_* env vars, which always win.
_default_profile = MODEL_PROFILES["construction-ppe" if _HAS_PPE_MODEL else "coco"]

PERSON_CLASS = _env("PERSON_CLASS", _default_profile["person_class"])
EQUIPMENT_CLASSES = _env_set("EQUIPMENT_CLASSES", _default_profile["equipment_classes"])
REQUIRED_EQUIPMENT = _env_set("REQUIRED_EQUIPMENT", _default_profile["required_equipment"])
NEGATIVE_CLASSES = _env_map("NEGATIVE_CLASSES", _default_profile["negative_classes"])

# True if the operator explicitly pinned class config via env (don't auto-detect).
CLASS_CONFIG_FROM_ENV = any(
    os.environ.get(f"PPE_{k}") is not None
    for k in ("PERSON_CLASS", "EQUIPMENT_CLASSES", "REQUIRED_EQUIPMENT", "NEGATIVE_CLASSES")
)

# Run inference on CPU or a specific device ("cpu", "0", "cuda:0", ...).
DEVICE = _env("DEVICE", "cpu")

# ---------------------------------------------------------------------------
# Input / Output
# ---------------------------------------------------------------------------
# 0 = default webcam, "rtsp://..." = IP/CCTV camera, "path.mp4" = video file.
VIDEO_SOURCE = _coerce_source(_env("VIDEO_SOURCE", 0))

# Annotated recording path, or None to disable.
OUTPUT_VIDEO = _env("OUTPUT_VIDEO", "output/ppe-detection-output.mp4")

# Directory for logs, reports, and output media.
OUTPUT_DIR = _env("OUTPUT_DIR", "output")

# CSV of individual violation events.
VIOLATION_LOG = _env("VIOLATION_LOG", "output/violations.csv")

# JSON session summary written on shutdown (per-person counts, run stats).
SESSION_REPORT = _env("SESSION_REPORT", "output/session_report.json")

# Application log file (structured logging also goes to the console).
LOG_FILE = _env("LOG_FILE", "output/ppe_system.log")
LOG_LEVEL = _env("LOG_LEVEL", "INFO")

# ---------------------------------------------------------------------------
# Detection / tracking thresholds
# ---------------------------------------------------------------------------
# Base YOLO confidence floor for the model call. Kept low so weaker negative
# ("no_X") detections surface; per-category thresholds below do the real
# filtering. Diagnostics on real footage showed worn-PPE at 0.5-0.86 but valid
# "no_helmet" signals down around 0.25-0.48, so the floor must sit below those.
CONF_THRESH = _env("CONF_THRESH", 0.20, float)

# Positive equipment (helmet/vest/...) detections are strong in practice, so
# require solid confidence to count them as "worn".
EQUIPMENT_CONF_THRESH = _env("EQUIPMENT_CONF_THRESH", 0.40, float)

# Negative ("no_X") detections run weaker; use a lower bar than positives, but
# high enough to reject the 0.1-0.2 noise band and avoid false violations.
NEGATIVE_CONF_THRESH = _env("NEGATIVE_CONF_THRESH", 0.35, float)

# Person detections need a higher confidence than equipment, but not too high:
# the model's person class over-triggers on person-shaped objects at low
# confidence, yet real people -- especially close to the camera -- can sit
# around 0.30-0.80. 0.28 sits above the junk band and below genuine people.
PERSON_CONF_THRESH = _env("PERSON_CONF_THRESH", 0.28, float)

# Reject person boxes smaller than this fraction of the frame area (filters
# tiny spurious detections). 0 disables the filter.
MIN_PERSON_AREA_FRAC = _env("MIN_PERSON_AREA_FRAC", 0.005, float)

IOU_THRESH = _env("IOU_THRESH", 0.15, float)     # Min equipment->person score
MAX_LOST = _env("MAX_LOST", 45, int)             # Frames before retiring a track

# Tracker matching knobs. The tracker combines bounding-box IoU with a
# size-adaptive centroid distance plus motion prediction, so IDs stay stable
# even when a person moves a lot between frames (common on slow CPU inference).
DIST_THRESH = _env("DIST_THRESH", 120.0, float)  # Baseline centroid tolerance (px)
TRACK_IOU_WEIGHT = _env("TRACK_IOU_WEIGHT", 0.5, float)  # IoU vs distance blend (0..1)
TRACK_MIN_IOU = _env("TRACK_MIN_IOU", 0.1, float)        # IoU alone accepts a match
TRACK_DIST_SCALE = _env("TRACK_DIST_SCALE", 1.5, float)  # Tolerance x person diagonal

# Process every Nth frame (1 = every frame). Default 3 keeps a heavy model
# (e.g. yolo26m) responsive on CPU: ~3x throughput while still processing
# enough frames/sec for stable tracking and temporal confirmation. Set to 1
# for maximum accuracy on a fast machine/GPU, or higher if still too slow.
FRAME_STRIDE = _env("FRAME_STRIDE", 3, int)

# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
# Measured in PROCESSED frames (i.e. after FRAME_STRIDE). With the default
# stride of 3 at ~24fps, 6 processed frames ~= 0.75s of sustained violation
# before an alert fires -- responsive without being twitchy.
VIOLATION_FRAMES_BEFORE_ALERT = _env("VIOLATION_FRAMES_BEFORE_ALERT", 6, int)
ALERT_COOLDOWN_SECONDS = _env("ALERT_COOLDOWN_SECONDS", 15, int)

# ---------------------------------------------------------------------------
# Temporal confirmation
# ---------------------------------------------------------------------------
# A violation is only confirmed when it dominates a person's recent frames,
# so a single missed helmet/vest detection does not raise a false alarm.
TEMPORAL_WINDOW = _env("TEMPORAL_WINDOW", 10, int)         # processed frames remembered per person
TEMPORAL_CONFIRM_FRAC = _env("TEMPORAL_CONFIRM_FRAC", 0.6, float)  # majority needed
TEMPORAL_MIN_FRAMES = _env("TEMPORAL_MIN_FRAMES", 5, int)  # min non-compliant frames

# Save a snapshot image of the frame at the moment each alert fires.
SAVE_ALERT_SNAPSHOTS = _env("SAVE_ALERT_SNAPSHOTS", True, bool)
SNAPSHOT_DIR = _env("SNAPSHOT_DIR", "output/snapshots")

# Show the live annotated preview window ('q' to quit). Disable for headless.
SHOW_PREVIEW = _env("SHOW_PREVIEW", True, bool)

# ---------------------------------------------------------------------------
# Stream robustness (important for RTSP / CCTV)
# ---------------------------------------------------------------------------
# On read failure, attempt to reopen the source this many times before giving
# up. Set 0 to disable auto-reconnect (sensible for finite video files).
STREAM_MAX_RECONNECTS = _env("STREAM_MAX_RECONNECTS", 5, int)
STREAM_RECONNECT_DELAY = _env("STREAM_RECONNECT_DELAY", 3.0, float)  # seconds

# ---------------------------------------------------------------------------
# Compliance colors (B, G, R) for OpenCV
# ---------------------------------------------------------------------------
COLOR_COMPLIANT = (50, 220, 50)      # green
COLOR_UNCERTAIN = (0, 215, 255)      # amber/yellow - not enough evidence
COLOR_NONCOMPLIANT = (40, 40, 220)   # red
COLOR_NOPERSON = (150, 150, 150)     # grey - no person detected
COLOR_TEXT = (255, 255, 255)


def apply_overrides(overrides: dict):
    """Apply a dict of {SETTING_NAME: value} onto this module (used by CLI)."""
    g = globals()
    for key, value in overrides.items():
        if value is None:
            continue
        if key == "VIDEO_SOURCE":
            value = _coerce_source(value)
        g[key] = value


def as_dict():
    """Return the current public settings as a plain dict (for logging)."""
    return {
        k: v
        for k, v in globals().items()
        if k.isupper() and not k.startswith("_")
    }
