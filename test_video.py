"""
Test the PPE pipeline on a PRE-RECORDED video.
================================================

Runs the full detection -> assignment -> tracking -> compliance -> temporal
confirmation pipeline over a video file, and for every CONFIRMED violation it
saves an annotated snapshot that highlights WHO violated and WHAT PPE is
missing. Also writes a CSV log and prints a summary.

Usage:
    python test_video.py --source path/to/clip.mp4
    python test_video.py --source clip.mp4 --require helmet,vest,gloves
    python test_video.py --source clip.mp4 --no-preview
    python test_video.py --source clip.mp4 --out-dir output/my_test

Diagnostic mode (raw model, no pipeline):
    python test_video.py --source clip.mp4 --diagnose
    python test_video.py --source frame.jpg --diagnose --diag-conf 0.05

Per-person assignment debug (shows each person's per-item PPE state):
    python test_video.py --source clip.mp4 --debug
    python test_video.py --source clip.mp4 --debug --require helmet,vest,gloves

Each confirmed violation produces:
    output/test_video/snapshots/violation_id<ID>_<item>_<time>.jpg   (annotated)
    a row in output/test_video/violations.csv
and a final summary is printed and saved to output/test_video/summary.json.
"""

import argparse
import csv
import json
import os
import sys
from collections import Counter
from datetime import datetime

import cv2

import config


def parse_args():
    p = argparse.ArgumentParser(description="Test PPE pipeline on a recorded video")
    p.add_argument("--source", required=True, help="Path to the video file")
    p.add_argument("--model", help="Override model path (default: config/ppe_model.pt)")
    p.add_argument("--require", help="Comma-separated required PPE (e.g. helmet,vest,gloves)")
    p.add_argument("--conf", type=float, help="Person/equipment confidence override")
    p.add_argument("--out-dir", default="output/test_video", help="Output directory")
    p.add_argument("--no-preview", action="store_true", help="Run headless")
    p.add_argument("--stride", type=int, help="Process every Nth frame")
    p.add_argument("--diagnose", action="store_true",
                   help="Raw-model diagnostic: print every detection + confidence "
                        "(no pipeline/compliance/snapshots), then exit")
    p.add_argument("--diag-conf", type=float, default=0.05,
                   help="Confidence floor for --diagnose (low, to see faint hits)")
    p.add_argument("--every", type=int, default=30,
                   help="For --diagnose on video: sample every Nth frame")
    p.add_argument("--debug", action="store_true",
                   help="Per-person assignment debug: print each tracked person's "
                        "per-item PPE state (PRESENT/ABSENT/UNKNOWN) and draw it")
    p.add_argument("--debug-every", type=int, default=15,
                   help="Print the --debug breakdown every Nth processed frame")
    return p.parse_args()


def apply_overrides(args):
    ov = {}
    if args.model:
        ov["MODEL_PATH"] = args.model
    if args.require:
        ov["REQUIRED_EQUIPMENT"] = {s.strip() for s in args.require.split(",") if s.strip()}
    if args.conf is not None:
        ov["CONF_THRESH"] = args.conf
        ov["PERSON_CONF_THRESH"] = args.conf
    if args.stride is not None:
        ov["FRAME_STRIDE"] = max(1, args.stride)
    if args.no_preview:
        ov["SHOW_PREVIEW"] = False
    # Snapshots are produced by this script directly (annotated), so disable
    # the pipeline's own raw-frame snapshotting to avoid duplicates.
    ov["SAVE_ALERT_SNAPSHOTS"] = False
    config.apply_overrides(ov)


def status_color(status):
    return {
        "Compliant": config.COLOR_COMPLIANT,
        "Uncertain": config.COLOR_UNCERTAIN,
        "Non-compliant": config.COLOR_NONCOMPLIANT,
    }.get(status, config.COLOR_UNCERTAIN)


def draw_person(frame, pr):
    x1, y1, x2, y2 = [int(v) for v in pr.box]
    color = status_color(pr.status)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"ID {pr.person_id}: {pr.status}"
    if pr.missing:
        label += f"  missing: {', '.join(sorted(pr.missing))}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), color, -1)
    cv2.putText(frame, label, (x1 + 4, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, config.COLOR_TEXT, 1, cv2.LINE_AA)


_STATE_MARK = {"PRESENT": "OK ", "ABSENT": "XX ", "UNKNOWN": "?? "}


def print_person_breakdown(result, frame_no, log):
    """Print a per-person, per-item assignment breakdown to the console."""
    if not result.persons:
        print(f"[f{frame_no}] no persons")
        return
    print(f"[f{frame_no}] {result.person_count} person(s):")
    for pr in result.persons:
        print(f"  Person ID {pr.person_id}  status={pr.status} "
              f"(this-frame={pr.raw_status})")
        if pr.states:
            for item in sorted(pr.states):
                state = pr.states[item]
                mark = {"PRESENT": "[v]", "ABSENT": "[x]", "UNKNOWN": "[?]"}.get(state, "[?]")
                print(f"      {mark} {item:10s} {state}")
        if pr.detected:
            print(f"      detected on person: {sorted(pr.detected)}")


def draw_person_debug(frame, pr):
    """Draw the per-item state breakdown next to the person box."""
    x1, y1, x2, y2 = [int(v) for v in pr.box]
    lines = [f"ID {pr.person_id}: {pr.status}"]
    for item in sorted(pr.states):
        st = pr.states[item]
        lines.append(f"  {_STATE_MARK.get(st, '?? ')}{item}={st}")
    # Background panel for readability.
    y = y1 + 2
    for i, line in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        yy = y + i * (th + 6)
        cv2.rectangle(frame, (x2 + 4, yy), (x2 + 10 + tw, yy + th + 4),
                      (0, 0, 0), -1)
        col = (config.COLOR_COMPLIANT if "PRESENT" in line
               else config.COLOR_NONCOMPLIANT if "ABSENT" in line
               else config.COLOR_UNCERTAIN if "UNKNOWN" in line
               else config.COLOR_TEXT)
        cv2.putText(frame, line, (x2 + 7, yy + th),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)


def annotate_violation(frame, violating_pr, all_persons, frame_no):
    """Return a copy of the frame with every person drawn and a banner."""
    img = frame.copy()
    for pr in all_persons:
        draw_person(img, pr)
    # Red evidence banner describing the specific violation.
    h, w = img.shape[:2]
    missing = ", ".join(sorted(violating_pr.missing)) or "PPE"
    banner = (f"VIOLATION  ID {violating_pr.person_id}  "
              f"missing: {missing}  frame#{frame_no}")
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), config.COLOR_NONCOMPLIANT, -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.putText(img, banner, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                config.COLOR_TEXT, 2, cv2.LINE_AA)
    cv2.putText(img, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                config.COLOR_TEXT, 1, cv2.LINE_AA)
    return img


def run_diagnose(args):
    """
    Run the raw model directly and print every detection with confidence.
    No pipeline, no compliance logic -- shows WHY an item is / isn't detected.
    Works on an image or a video (sampled every --every frames).
    """
    from ultralytics import YOLO
    model = YOLO(args.model or config.MODEL_PATH)
    print("Model classes:", model.names)
    print(f"Diagnose: {args.source}  (conf floor={args.diag_conf})\n")

    def show(result, tag=""):
        counts = Counter()
        if result.boxes is None or len(result.boxes) == 0:
            print(f"{tag}  (no detections)")
            return counts
        rows = []
        for box in result.boxes:
            name = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            xyxy = [round(x, 1) for x in box.xyxy[0].tolist()]
            rows.append((name, conf, xyxy))
            counts[name] += 1
        for name, conf, xyxy in sorted(rows, key=lambda r: -r[1]):
            print(f"{tag}  {name:12s} conf={conf:.3f} box={xyxy}")
        return counts

    total = Counter()
    ext = os.path.splitext(args.source)[1].lower()
    if ext in (".mp4", ".avi", ".mov", ".mkv"):
        cap = cv2.VideoCapture(args.source)
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.every == 0:
                total.update(show(model(frame, conf=args.diag_conf, verbose=False)[0],
                                  tag=f"[f{idx}]"))
            idx += 1
        cap.release()
    else:
        total.update(show(model(args.source, conf=args.diag_conf, verbose=False)[0]))

    print("\n==== DETECTION TOTALS ====")
    for name, n in total.most_common():
        print(f"  {name:12s} {n}")

    hp, hn = total.get("helmet", 0), total.get("no_helmet", 0)
    print("\n==== HELMET READ ====")
    print(f"  helmet (worn): {hp}   no_helmet (bare): {hn}")
    if hp == 0 and hn == 0:
        print("  -> No helmet evidence at all: false negative / domain mismatch. "
              "UNKNOWN is correct; model needs work.")
    elif hp > 0 and hn == 0:
        print("  -> Model sees worn helmets. If the app still flagged violations, "
              "lower CONF_THRESH / raise sensitivity.")
    elif hn > 0 and hp == 0:
        print("  -> Only 'no_helmet' seen. If a helmet was worn, that's a model "
              "error (retrain); if truly bare, it's correct.")
    else:
        print("  -> Mixed; inspect per-frame rows above.")


def main():
    args = parse_args()
    apply_overrides(args)

    if not os.path.exists(args.source):
        print(f"[ERROR] Video not found: {args.source}")
        sys.exit(1)

    if args.diagnose:
        run_diagnose(args)
        return

    out_dir = args.out_dir
    snap_dir = os.path.join(out_dir, "snapshots")
    os.makedirs(snap_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "violations.csv")
    summary_path = os.path.join(out_dir, "summary.json")
    annotated_video = os.path.join(out_dir, "annotated.mp4")

    # Build pipeline components (reuse the production pipeline).
    from logging_setup import get_logger
    from detector import PPEDetector
    from tracker import CentroidTracker
    from alerts import AlertManager, ViolationLogger, SessionReporter
    from pipeline import CompliancePipeline

    log = get_logger("test_video")
    log.info("Testing video: %s", args.source)

    detector = PPEDetector()
    log.info("Model person=%s required=%s equipment=%s",
             detector.person_class, sorted(detector.required),
             sorted(detector.equipment_classes))

    tracker = CentroidTracker(
        max_lost=config.MAX_LOST, dist_thresh=config.DIST_THRESH,
        iou_weight=config.TRACK_IOU_WEIGHT, min_iou=config.TRACK_MIN_IOU,
        dist_scale=config.TRACK_DIST_SCALE,
    )
    # Use a short cooldown so the test captures distinct violation events but
    # not every single frame.
    alerts = AlertManager(cooldown_seconds=config.ALERT_COOLDOWN_SECONDS)
    alerts._backend = None  # no pop-ups during a batch video test; log instead
    inner_logger = ViolationLogger(os.path.join(out_dir, "_pipeline_violations.csv"))
    reporter = SessionReporter(os.path.join(out_dir, "_pipeline_report.json"))
    pipeline = CompliancePipeline(detector, tracker, alerts, inner_logger, reporter)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video: {args.source}")
        sys.exit(1)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {w}x{h} @ {fps:.1f}fps, {total} frames")

    writer = cv2.VideoWriter(annotated_video, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (w, h))

    # CSV of snapshots this script captures.
    csv_fh = open(csv_path, "w", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_fh)
    csv_w.writerow(["frame", "person_id", "missing_items", "snapshot"])

    # Track which (person, missing-set) we've already snapshotted recently, so
    # one continuous violation yields one snapshot per cooldown, not hundreds.
    last_snapshot_frame = {}
    cooldown_frames = int(config.ALERT_COOLDOWN_SECONDS * fps)
    snapshots_taken = 0
    per_person = Counter()
    per_item = Counter()
    frame_no = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_no += 1
        if frame_no % config.FRAME_STRIDE != 0:
            # still write the frame so the annotated video stays in sync
            writer.write(frame)
            continue

        result = pipeline.process(frame)

        # Debug: periodic per-person assignment breakdown to the console.
        if args.debug and (frame_no // config.FRAME_STRIDE) % args.debug_every == 0:
            print_person_breakdown(result, frame_no, log)

        # Draw all persons for the annotated output video.
        annotated_frame = frame.copy()
        for pr in result.persons:
            draw_person(annotated_frame, pr)
            if args.debug:
                draw_person_debug(annotated_frame, pr)
        hud = (f"persons={result.person_count} "
               f"violations={result.violation_count} snaps={snapshots_taken}")
        cv2.putText(annotated_frame, hud, (8, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(annotated_frame)

        # For each confirmed violator, save an annotated evidence snapshot
        # (respecting a per-person cooldown so we don't flood).
        for pr in result.persons:
            if pr.status != "Non-compliant":
                continue
            pid = pr.person_id
            last = last_snapshot_frame.get(pid, -10**9)
            if frame_no - last < cooldown_frames:
                continue
            last_snapshot_frame[pid] = frame_no

            miss = "_".join(sorted(pr.missing)) or "PPE"
            ts = datetime.now().strftime("%H%M%S")
            fname = f"violation_id{pid}_{miss}_f{frame_no}_{ts}.jpg"
            fpath = os.path.join(snap_dir, fname)
            snap = annotate_violation(frame, pr, result.persons, frame_no)
            cv2.imwrite(fpath, snap)

            csv_w.writerow([frame_no, pid, "|".join(sorted(pr.missing)), fpath])
            csv_fh.flush()
            snapshots_taken += 1
            per_person[pid] += 1
            for it in pr.missing:
                per_item[it] += 1
            log.warning("Snapshot saved: ID=%s missing=%s -> %s",
                        pid, sorted(pr.missing), fpath)

        if not args.no_preview:
            cv2.imshow("PPE Video Test", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

    cap.release()
    writer.release()
    csv_fh.close()
    if not args.no_preview:
        cv2.destroyAllWindows()

    summary = {
        "source": args.source,
        "frames": frame_no,
        "required_equipment": sorted(detector.required),
        "snapshots_taken": snapshots_taken,
        "unique_violators": len(per_person),
        "violations_by_person": dict(per_person),
        "missing_item_counts": dict(per_item),
        "annotated_video": annotated_video,
        "snapshots_dir": snap_dir,
        "csv": csv_path,
    }
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    inner_logger.close()

    print("\n==== TEST SUMMARY ====")
    print(json.dumps(summary, indent=2))
    print(f"\nAnnotated video : {annotated_video}")
    print(f"Snapshots       : {snap_dir}  ({snapshots_taken} saved)")
    print(f"Violations CSV  : {csv_path}")
    print(f"Summary JSON    : {summary_path}")


if __name__ == "__main__":
    main()
