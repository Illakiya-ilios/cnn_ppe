"""
End-to-end pipeline check (no camera needed).
Builds a synthetic frame, runs the real YOLO model, and exercises
assignment + compliance + alert/logger/reporter wiring.
Run: python test_pipeline.py
"""

import os
import numpy as np

import config
from detector import PPEDetector
from alerts import AlertManager, ViolationLogger, SessionReporter


def main():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    detector = PPEDetector()
    persons, equipment = detector.detect(frame)
    print(f"detect() ran. persons={len(persons)} equipment={len(equipment)}")
    print(f"has_ppe_classes() -> {detector.has_ppe_classes()}")

    assigned = detector.assign_equipment(persons, equipment)
    print(f"assign_equipment() -> {assigned}")

    status_full, miss_full = detector.compliance(set(config.REQUIRED_EQUIPMENT))
    print(f"compliance(all required) -> {status_full}, missing={miss_full}")
    assert status_full == "Compliant"

    # "No gear": any required item with reliable absence evidence (a negative
    # class) is ABSENT -> Non-compliant; otherwise everything is UNKNOWN ->
    # Uncertain. Both are valid depending on config; assert it's never a
    # false "Compliant".
    status_none, _ = detector.compliance(set())
    print(f"compliance(no gear)  -> {status_none}")
    assert status_none in ("Non-compliant", "Uncertain")
    assert status_none != "Compliant"

    # Alert (no snapshot) is queued, drained by pump(), logger/reporter wiring.
    config.SAVE_ALERT_SNAPSHOTS = False
    alerts = AlertManager(cooldown_seconds=0)
    alerts._backend = None  # headless/console
    fired = alerts.maybe_alert(1, {"helmet"}, frame=None)
    assert fired is True
    alerts.pump()  # should not raise
    print("AlertManager queue/pump fired OK")

    test_log = "_test_violations.csv"
    if os.path.exists(test_log):
        os.remove(test_log)
    logger = ViolationLogger(test_log)
    logger.log(1, "Non-compliant", {"helmet"}, set())
    assert logger.total == 1
    logger.close()
    with open(test_log, encoding="utf-8") as fh:
        assert len(fh.read().strip().splitlines()) == 2
    os.remove(test_log)
    print("ViolationLogger wrote header + row OK")

    test_report = "_test_report.json"
    reporter = SessionReporter(test_report)
    reporter.record(1, "Non-compliant", {"helmet"})
    reporter.record(1, "Non-compliant", {"vest"})
    summary = reporter.write(frames_processed=10)
    assert summary["total_violations"] == 2
    assert summary["unique_violators"] == 1
    assert summary["violations_by_person"]["1"] == 2
    os.remove(test_report)
    print("SessionReporter aggregated + wrote JSON OK")

    print("\nPipeline test passed.")


if __name__ == "__main__":
    main()
