"""
Alerting and violation reporting.

- AlertManager    : screen pop-up for confirmed violations (per-person cooldown),
                    optional snapshot capture at the moment of alert.
- ViolationLogger : append every confirmed violation to a CSV report.
- SessionReporter : aggregate per-person counts and write a JSON summary on exit.
"""

import csv
import json
import os
import queue
import threading
import time
from collections import Counter
from datetime import datetime

import config
from logging_setup import get_logger

log = get_logger("alerts")


class AlertManager:
    """
    Screen pop-up alerts for confirmed violations, with a per-person cooldown.

    Thread-safety: the previous implementation created tkinter windows from
    worker threads, which crashed ("main thread is not in main loop" /
    "Tcl_AsyncDelete"). This version is designed so the pipeline thread only
    enqueues alerts (maybe_alert), and the MAIN thread shows them (pump()).

    Pop-up backends, in order of preference:
      - "win32": native Win32 MessageBox via ctypes. This is a plain system
        call with no Tcl/event-loop state, so it is safe to launch in a short
        daemon thread (keeps the video loop non-blocking) without crashing.
      - "tkinter": only used on non-Windows, and only from the main thread
        (via pump), so no cross-thread Tcl access occurs.
      - console: structured log line.
    """

    def __init__(self, cooldown_seconds=15):
        self.cooldown_seconds = cooldown_seconds
        self._last_alert = {}  # person_id -> last alert epoch seconds
        self._queue = queue.Queue()
        self._backend = self._detect_popup_backend()
        if config.SAVE_ALERT_SNAPSHOTS:
            os.makedirs(config.SNAPSHOT_DIR, exist_ok=True)
        log.info("Alert pop-up backend: %s", self._backend or "console")

    def _detect_popup_backend(self):
        # Prefer the native Win32 box on Windows (thread-safe, no Tcl).
        if os.name == "nt":
            try:
                import ctypes  # noqa: F401
                return "win32"
            except Exception:
                pass
        try:
            import tkinter  # noqa: F401
            return "tkinter"
        except Exception:
            return None

    def _save_snapshot(self, person_id, frame):
        if not config.SAVE_ALERT_SNAPSHOTS or frame is None:
            return None
        try:
            import cv2
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(config.SNAPSHOT_DIR, f"violation_id{person_id}_{ts}.jpg")
            cv2.imwrite(path, frame)
            return path
        except Exception as exc:
            log.error("Failed to save snapshot: %s", exc)
            return None

    def maybe_alert(self, person_id, missing_items, frame=None):
        """
        Record a violation and queue a pop-up if the per-person cooldown has
        elapsed. Safe to call from any thread; no GUI work happens here.
        Returns True if an alert was queued, False if suppressed by cooldown.
        """
        now = time.time()
        last = self._last_alert.get(person_id, 0.0)
        if now - last < self.cooldown_seconds:
            return False

        self._last_alert[person_id] = now
        snapshot = self._save_snapshot(person_id, frame)
        missing = ", ".join(sorted(missing_items)) if missing_items else "ALL PPE"
        title = "SAFETY VIOLATION"
        message = (
            f"PPE violation detected!\n\n"
            f"Person ID: {person_id}\n"
            f"Missing: {missing}\n"
            f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        if snapshot:
            message += f"\nSnapshot: {snapshot}"
        self._queue.put((title, message))
        return True

    def pump(self):
        """
        Display any queued pop-ups. MUST be called from the main thread
        (e.g. once per frame in the video loop). Non-blocking.
        """
        while True:
            try:
                title, message = self._queue.get_nowait()
            except queue.Empty:
                return
            self._display(title, message)

    def _display(self, title, message):
        if self._backend == "win32":
            # Native MessageBox in a daemon thread so the video loop never
            # blocks. ctypes MessageBoxW has no Tcl state -> no crash.
            def _box():
                try:
                    import ctypes
                    # 0x30 = MB_ICONWARNING, 0x1000 = MB_SYSTEMMODAL (topmost)
                    ctypes.windll.user32.MessageBoxW(0, message, title, 0x30 | 0x1000)
                except Exception as exc:
                    log.error("win32 popup failed: %s | %s: %s", exc, title, message)

            threading.Thread(target=_box, daemon=True).start()

        elif self._backend == "tkinter":
            # Runs on the main thread (pump is main-thread only) -> safe.
            try:
                import tkinter as tk
                from tkinter import messagebox
                root = tk.Tk()
                root.withdraw()
                root.attributes("-topmost", True)
                messagebox.showwarning(title, message)
                root.destroy()
            except Exception as exc:
                log.error("tkinter popup failed: %s | %s: %s", exc, title, message)

        else:
            log.warning("ALERT (no GUI) | %s: %s", title, message.replace("\n", " "))


class ViolationLogger:
    """Appends confirmed violations to a CSV report."""

    def __init__(self, path):
        self.path = path
        self._count = 0
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        new_file = not os.path.exists(path)
        self._fh = open(path, "a", newline="", encoding="utf-8")
        self._writer = csv.writer(self._fh)
        if new_file:
            self._writer.writerow(
                ["timestamp", "person_id", "status", "missing_items", "detected_items"]
            )
            self._fh.flush()

    def log(self, person_id, status, missing_items, detected_items):
        self._count += 1
        self._writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                person_id,
                status,
                "|".join(sorted(missing_items)),
                "|".join(sorted(detected_items)),
            ]
        )
        self._fh.flush()

    @property
    def total(self):
        return self._count

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


class SessionReporter:
    """Aggregates violations for the run and writes a JSON summary on close."""

    def __init__(self, path):
        self.path = path
        self.started = datetime.now()
        self.per_person = Counter()
        self.by_status = Counter()
        self.missing_items = Counter()
        self.total = 0

    def record(self, person_id, status, missing_items):
        self.total += 1
        self.per_person[str(person_id)] += 1
        self.by_status[status] += 1
        for item in missing_items:
            self.missing_items[item] += 1

    def write(self, frames_processed=0):
        ended = datetime.now()
        duration = (ended - self.started).total_seconds()
        summary = {
            "session_start": self.started.strftime("%Y-%m-%d %H:%M:%S"),
            "session_end": ended.strftime("%Y-%m-%d %H:%M:%S"),
            "duration_seconds": round(duration, 1),
            "frames_processed": frames_processed,
            "total_violations": self.total,
            "unique_violators": len(self.per_person),
            "violations_by_person": dict(self.per_person),
            "violations_by_status": dict(self.by_status),
            "most_missing_items": dict(self.missing_items),
        }
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            log.info("Session report written: %s", self.path)
        except Exception as exc:
            log.error("Failed to write session report: %s", exc)
        return summary
