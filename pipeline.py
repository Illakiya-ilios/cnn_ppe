"""
Compliance monitoring pipeline.

Encapsulates the per-frame processing stages as a single reusable component,
mirroring a clean inference pipeline:

    [Frame]
       -> YOLO Detection          (persons + equipment)
       -> Equipment Assignment    (IoU + center-in-box)
       -> Person Tracking         (persistent IDs)
       -> Compliance Assessment   (Compliant / Uncertain / Non-compliant)
       -> Temporal Confirmation    (per-person evidence buffer)
       -> Alerting + Reporting     (debounced pop-up, CSV, snapshots)
       -> Annotated frame

A violation is only CONFIRMED (and alertable) when it is the dominant state
across a person's recent frames, so a single missed detection does not raise a
false alarm. "Uncertain" (not enough evidence) is never treated as a violation.

Keeping this separate from main.py means the same pipeline can be driven by a
different front-end later (service, batch processor, web app) without change.
"""

from collections import defaultdict, deque

import config
from logging_setup import get_logger

log = get_logger("pipeline")


from dataclasses import dataclass, field


@dataclass
class PersonResult:
    person_id: int
    box: tuple
    status: str                                   # confirmed/display status
    raw_status: str = ""                          # this-frame status
    missing: set = field(default_factory=set)
    detected: set = field(default_factory=set)


@dataclass
class FrameResult:
    persons: list = field(default_factory=list)   # list[PersonResult]
    violation_count: int = 0                       # CONFIRMED violations
    person_count: int = 0                          # people detected this frame


class CompliancePipeline:
    def __init__(self, detector, tracker, alerts, logger, reporter):
        self.detector = detector
        self.tracker = tracker
        self.alerts = alerts
        self.logger = logger
        self.reporter = reporter
        # Per-person rolling window of this-frame statuses for temporal
        # confirmation, and the set of missing items most recently seen.
        self._history = defaultdict(lambda: deque(maxlen=config.TEMPORAL_WINDOW))
        self._last_missing = defaultdict(set)
        self._alerted_streak = defaultdict(int)

    def process(self, frame):
        """Run the full pipeline on one frame and return a FrameResult."""
        persons, equipment = self.detector.detect(frame)
        assigned = self.detector.assign_equipment(persons, equipment)
        tracked = self.tracker.update(persons)

        result = FrameResult()
        result.person_count = len(persons)

        for idx, pbox in enumerate(persons):
            detected_items = assigned[idx]
            raw_status, missing = self.detector.compliance(detected_items)
            person_id = self._match_track_id(tracked, pbox)

            # Temporal confirmation: record this frame's status and decide the
            # person's confirmed status from the recent window.
            hist = self._history[person_id]
            hist.append(raw_status)
            if missing:
                self._last_missing[person_id] = set(missing)
            confirmed = self._confirmed_status(hist)

            pr = PersonResult(
                person_id=person_id,
                box=pbox,
                status=confirmed,
                raw_status=raw_status,
                missing=set(self._last_missing[person_id]) if confirmed == "Non-compliant" else set(),
                detected=set(detected_items),
            )
            result.persons.append(pr)

            if confirmed == "Non-compliant":
                result.violation_count += 1
                streak = self._alerted_streak[person_id] + 1
                self._alerted_streak[person_id] = streak
                # Alert once the confirmed violation has also persisted the
                # configured number of frames (debounce on top of temporal).
                if streak == config.VIOLATION_FRAMES_BEFORE_ALERT:
                    miss = pr.missing
                    fired = self.alerts.maybe_alert(person_id, miss, frame)
                    if fired:
                        self.logger.log(person_id, confirmed, miss, detected_items)
                        self.reporter.record(person_id, confirmed, miss)
                        log.warning("CONFIRMED VIOLATION id=%s missing=%s",
                                    person_id, sorted(miss))
            else:
                self._alerted_streak[person_id] = 0

        return result

    @staticmethod
    def _confirmed_status(history):
        """
        Decide a confirmed status from a window of per-frame statuses.

        A "Non-compliant" is only confirmed when non-compliant frames dominate
        the recent window (>= TEMPORAL_CONFIRM_FRAC), so a single missed
        detection does not flip a compliant worker into a violation.
        """
        if not history:
            return "Uncertain"
        n = len(history)
        nc = sum(1 for s in history if s == "Non-compliant")
        if nc / n >= config.TEMPORAL_CONFIRM_FRAC and nc >= config.TEMPORAL_MIN_FRAMES:
            return "Non-compliant"
        comp = sum(1 for s in history if s == "Compliant")
        if comp / n >= config.TEMPORAL_CONFIRM_FRAC:
            return "Compliant"
        return "Uncertain"

    @staticmethod
    def _match_track_id(tracked, pbox, tol=1.0):
        for tid, tbox in tracked.items():
            if all(abs(a - b) <= tol for a, b in zip(tbox, pbox)):
                return tid
        return -1
