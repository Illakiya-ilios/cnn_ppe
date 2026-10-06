"""
PPE Safety Violation Detection System - entry point.

Reads live camera footage (webcam / RTSP CCTV) or a video file, detects people
and their PPE, tracks each person, assesses compliance, raises a screen pop-up
alert on confirmed violations, and writes a CSV log + JSON session report.

Usage:
    python main.py                         # use config.py / env defaults
    python main.py --source 0              # webcam
    python main.py --source rtsp://cam/stream1
    python main.py --source clip.mp4 --no-preview
    python main.py --model ppe_model.pt --conf 0.4

Any setting in config.py can also be set via a PPE_<NAME> environment variable.
Press 'q' in the preview window (or Ctrl+C in the console) to stop.
"""

import argparse
import sys
import time

import cv2

import config


def parse_args():
    p = argparse.ArgumentParser(description="PPE Safety Violation Detection System")
    p.add_argument("--source", help="Video source: 0 (webcam), RTSP URL, or file path")
    p.add_argument("--model", help="Path to YOLO .pt weights")
    p.add_argument("--device", help="Inference device: cpu, 0, cuda:0 ...")
    p.add_argument("--conf", type=float, help="Detection confidence threshold")
    p.add_argument("--stride", type=int, help="Process every Nth frame")
    p.add_argument("--output", help="Annotated output video path ('none' to disable)")
    p.add_argument("--no-preview", action="store_true", help="Run headless (no window)")
    p.add_argument("--log-level", help="DEBUG / INFO / WARNING / ERROR")
    return p.parse_args()


def apply_cli(args):
    overrides = {}
    if args.source is not None:
        overrides["VIDEO_SOURCE"] = args.source
    if args.model is not None:
        overrides["MODEL_PATH"] = args.model
    if args.device is not None:
        overrides["DEVICE"] = args.device
    if args.conf is not None:
        overrides["CONF_THRESH"] = args.conf
    if args.stride is not None:
        overrides["FRAME_STRIDE"] = max(1, args.stride)
    if args.output is not None:
        overrides["OUTPUT_VIDEO"] = None if args.output.lower() == "none" else args.output
    if args.no_preview:
        overrides["SHOW_PREVIEW"] = False
    if args.log_level is not None:
        overrides["LOG_LEVEL"] = args.log_level
    config.apply_overrides(overrides)


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
        label += f" (missing: {', '.join(sorted(pr.missing))})"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, config.COLOR_TEXT, 1, cv2.LINE_AA)


def draw_hud(frame, text, color, fps):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_TEXT, 2, cv2.LINE_AA)
    fps_txt = f"{fps:4.1f} FPS"
    cv2.putText(frame, fps_txt, (w - 120, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_TEXT, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    apply_cli(args)

    # Imports after config overrides so modules read final settings.
    from logging_setup import get_logger
    from detector import PPEDetector
    from tracker import CentroidTracker
    from alerts import AlertManager, ViolationLogger, SessionReporter
    from pipeline import CompliancePipeline
    import video_source

    log = get_logger("main")
    log.info("Starting PPE Safety Violation Detection System")
    log.info("Config: %s", {k: config.as_dict()[k] for k in
                             ("MODEL_PATH", "VIDEO_SOURCE", "DEVICE", "CONF_THRESH",
                              "FRAME_STRIDE", "SHOW_PREVIEW")})

    detector = PPEDetector()
    if not detector.has_ppe_classes():
        log.warning(
            "Loaded model exposes NONE of the configured PPE classes %s. "
            "It detects persons only, so every person will read as a violation. "
            "Set MODEL_PATH to a PPE-trained model and update EQUIPMENT_CLASSES.",
            sorted(detector.equipment_classes),
        )

    tracker = CentroidTracker(
        max_lost=config.MAX_LOST,
        dist_thresh=config.DIST_THRESH,
        iou_weight=config.TRACK_IOU_WEIGHT,
        min_iou=config.TRACK_MIN_IOU,
        dist_scale=config.TRACK_DIST_SCALE,
    )
    alerts = AlertManager(cooldown_seconds=config.ALERT_COOLDOWN_SECONDS)
    logger = ViolationLogger(config.VIOLATION_LOG)
    reporter = SessionReporter(config.SESSION_REPORT)
    pipeline = CompliancePipeline(detector, tracker, alerts, logger, reporter)

    source = video_source.from_config()
    if not source.open():
        log.error("Could not open video source: %r. Set --source or PPE_VIDEO_SOURCE.",
                  config.VIDEO_SOURCE)
        logger.close()
        sys.exit(1)

    writer = None
    if config.OUTPUT_VIDEO:
        import os
        os.makedirs(os.path.dirname(config.OUTPUT_VIDEO) or ".", exist_ok=True)
        w = source.width or 1280
        h = source.height or 720
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(config.OUTPUT_VIDEO, fourcc, source.fps, (w, h))

    frame_idx = 0
    processed = 0
    last_result = None
    fps = 0.0
    t_prev = time.time()

    log.info("Running. Press 'q' in the preview to quit, or Ctrl+C in console.")
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                log.info("End of stream.")
                break

            frame_idx += 1
            # Frame stride: run the model on every Nth frame, reuse last result
            # for display on skipped frames to keep the preview smooth.
            if frame_idx % config.FRAME_STRIDE == 0:
                last_result = pipeline.process(frame)
                processed += 1

            if last_result is not None:
                for pr in last_result.persons:
                    draw_person(frame, pr)
                violations = last_result.violation_count
                person_count = last_result.person_count
                uncertain = sum(1 for p in last_result.persons if p.status == "Uncertain")
            else:
                violations = person_count = uncertain = 0

            now = time.time()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt) if fps else (1.0 / dt)

            # Four distinct states. Critically, "no person detected" is NOT
            # reported as compliant.
            if person_count == 0:
                draw_hud(frame, "NO PERSON DETECTED", config.COLOR_NOPERSON, fps)
            elif violations:
                draw_hud(frame,
                         f"SAFETY VIOLATION: {violations} confirmed | logged {logger.total}",
                         config.COLOR_NONCOMPLIANT, fps)
            elif uncertain:
                draw_hud(frame,
                         f"UNCERTAIN: {uncertain} person(s) - insufficient evidence",
                         config.COLOR_UNCERTAIN, fps)
            else:
                draw_hud(frame, f"COMPLIANT: {person_count} person(s)",
                         config.COLOR_COMPLIANT, fps)

            if writer is not None:
                writer.write(frame)

            # Show any queued pop-ups from the MAIN thread (thread-safe).
            alerts.pump()

            if config.SHOW_PREVIEW:
                cv2.imshow("PPE Safety Violation Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    log.info("Quit requested from preview window.")
                    break

    except KeyboardInterrupt:
        log.info("Interrupted by user (Ctrl+C).")
    finally:
        source.release()
        if writer is not None:
            writer.release()
        if config.SHOW_PREVIEW:
            cv2.destroyAllWindows()
        logger.close()
        summary = reporter.write(frames_processed=processed)
        log.info("Shutdown complete. Violations=%s unique_violators=%s frames=%s",
                 summary["total_violations"], summary["unique_violators"], processed)
        print(f"\nSession report -> {config.SESSION_REPORT}")
        print(f"Violation log  -> {config.VIOLATION_LOG}")


if __name__ == "__main__":
    main()
