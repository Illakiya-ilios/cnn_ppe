"""
PPE detection and per-person compliance assessment.

Wraps the YOLO model, splits detections into persons vs equipment, assigns
each equipment item to the nearest person (IoU + center-in-box bonus), and
computes a compliance status for every person.
"""

import config
from logging_setup import get_logger

log = get_logger("detector")


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_in_box(inner, outer):
    cx = (inner[0] + inner[2]) / 2.0
    cy = (inner[1] + inner[3]) / 2.0
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


class PPEDetector:
    def __init__(self):
        # Imported lazily so the module can be imported without ultralytics
        # installed (e.g. for unit tests of the geometry helpers).
        from ultralytics import YOLO

        log.info("Loading model: %s (device=%s)", config.MODEL_PATH, config.DEVICE)
        self.model = YOLO(config.MODEL_PATH)
        self.device = config.DEVICE
        self.names = self.model.names  # id -> class name

        # Start from config defaults, then auto-select a matching model profile
        # based on the loaded class names (unless the operator pinned env vars).
        self.person_class = config.PERSON_CLASS
        self.equipment_classes = set(config.EQUIPMENT_CLASSES)
        self.required = set(config.REQUIRED_EQUIPMENT)
        self.negative_classes = dict(getattr(config, "NEGATIVE_CLASSES", {}))
        self._auto_select_profile()

        # Optional dedicated person-detection model (two-model pipeline).
        self.person_model = None
        self.person_model_class = config.PERSON_MODEL_CLASS
        pm_path = getattr(config, "PERSON_MODEL_PATH", "") or ""
        if pm_path:
            log.info("Loading dedicated person model: %s", pm_path)
            self.person_model = YOLO(pm_path)
            log.info("Two-model pipeline: persons from '%s', PPE from '%s'.",
                     pm_path, config.MODEL_PATH)

    def _auto_select_profile(self):
        """Match the loaded model's class names to a known profile."""
        if config.CLASS_CONFIG_FROM_ENV:
            log.info("Class config pinned via env; skipping profile auto-detect.")
            return
        model_classes = set(self.names.values())
        for name, prof in config.MODEL_PROFILES.items():
            sig = prof.get("signature", set())
            if sig and sig.issubset(model_classes):
                self.person_class = prof["person_class"]
                self.equipment_classes = set(prof["equipment_classes"])
                self.required = set(prof["required_equipment"])
                self.negative_classes = dict(prof["negative_classes"])
                log.info("Auto-selected model profile '%s': person=%s required=%s",
                         name, self.person_class, sorted(self.required))
                return
        log.info("No known profile matched; using config defaults. "
                 "Model classes: %s", sorted(model_classes))

    def has_ppe_classes(self):
        """True if the loaded model exposes any configured PPE / negative class."""
        model_classes = set(self.names.values())
        return bool(
            (model_classes & self.equipment_classes)
            or (model_classes & set(self.negative_classes))
        )

    def detect(self, frame):
        """
        Run detection on a frame.

        Returns:
            persons:   list of (x1, y1, x2, y2)
            equipment: list of (class_name, (x1, y1, x2, y2))

        Negative-signal detections (e.g. "no-helmet") are emitted into
        `equipment` under a reserved "!<required_item>" name so the assignment
        stage attributes them to a person, and compliance treats them as a hard
        missing-item signal.
        """
        frame_area = float(frame.shape[0] * frame.shape[1])
        min_person_area = frame_area * config.MIN_PERSON_AREA_FRAC

        # Equipment (and single-model persons) from the main PPE model.
        equipment, main_persons = self._detect_equipment(frame, min_person_area)

        if self.person_model is not None:
            persons = self._detect_persons(frame, min_person_area)
        else:
            persons = main_persons
        return persons, equipment

    def _detect_persons(self, frame, min_person_area):
        """Detect people using the dedicated person model."""
        persons = []
        res = self.person_model(
            frame, conf=config.PERSON_CONF_THRESH, device=self.device,
            verbose=False,
        )
        if res and res[0].boxes is not None:
            names = self.person_model.names
            for box in res[0].boxes:
                if names.get(int(box.cls[0])) != self.person_model_class:
                    continue
                xyxy = box.xyxy[0].tolist()
                coords = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                if area < min_person_area:
                    continue
                persons.append(coords)
        return persons

    def _detect_equipment(self, frame, min_person_area):
        """
        Detect PPE/equipment from the main model. Also returns persons found by
        the main model, used only when no dedicated person model is configured.
        """
        results = self.model(
            frame, conf=config.CONF_THRESH, device=self.device, verbose=False
        )
        equipment = []
        main_persons = []
        if not results or results[0].boxes is None:
            return equipment, main_persons

        for box in results[0].boxes:
            name = self.names.get(int(box.cls[0]), str(int(box.cls[0])))
            conf = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()
            coords = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])

            if name == self.person_class:
                if conf < config.PERSON_CONF_THRESH:
                    continue
                area = (coords[2] - coords[0]) * (coords[3] - coords[1])
                if area < min_person_area:
                    continue
                main_persons.append(coords)
            elif name in self.equipment_classes:
                equipment.append((name, coords))
            elif name in self.negative_classes:
                equipment.append(("!" + self.negative_classes[name], coords))
        return equipment, main_persons

    def assign_equipment(self, persons, equipment):
        """
        Assign each equipment item to the best-matching person.

        Returns a list, one entry per person, of the set of equipment class
        names assigned to that person.
        """
        assigned = [set() for _ in persons]
        if not persons:
            return assigned

        for name, ebox in equipment:
            best_idx = -1
            best_score = 0.0
            for i, pbox in enumerate(persons):
                score = iou(ebox, pbox)
                if center_in_box(ebox, pbox):
                    score += 1.0  # strong bonus for spatial containment
                if score > best_score:
                    best_score = score
                    best_idx = i
            if best_idx >= 0 and best_score >= config.IOU_THRESH:
                assigned[best_idx].add(name)
        return assigned

    def item_states(self, detected_items):
        """
        Compute a three-state estimate (PRESENT / ABSENT / UNKNOWN) for every
        required PPE item, given a person's assigned detections.

        `detected_items` may contain:
          - positive equipment names present on the person (e.g. "helmet")
          - negative markers "!<item>" meaning that item was explicitly
            detected as absent (e.g. "!helmet" from a "no-helmet" box)

        State logic per required item:
          - PRESENT: the item was detected on the person.
          - ABSENT:  the item was NOT detected AND we have reliable absence
                     evidence for it -- either an explicit negative detection
                     ("!helmet"), or the item is configured as reliably-absent-
                     when-missing. ABSENT is a genuine violation signal.
          - UNKNOWN: the item was not detected and we have no trustworthy
                     absence evidence (e.g. vest, where a missed detection is
                     indistinguishable from a truly absent vest). UNKNOWN must
                     NOT be treated as a violation.

        Items that have a configured negative class (self.negative_classes
        values) are trusted on absence-by-omission too, because the model
        actively reasons about that item.
        """
        positives = {d for d in detected_items if not d.startswith("!")}
        negatives = {d[1:] for d in detected_items if d.startswith("!")}

        # Items for which "not detected" is reliable evidence of absence.
        reliable_absence = set(self.negative_classes.values())

        states = {}
        for item in self.required:
            if item in positives and item not in negatives:
                states[item] = "PRESENT"
            elif item in negatives:
                states[item] = "ABSENT"            # explicit no-X detection
            elif item in reliable_absence:
                states[item] = "ABSENT"            # model reasons about this item
            else:
                states[item] = "UNKNOWN"           # missed detection vs truly absent
        return states

    def compliance(self, detected_items):
        """
        Classify a person's compliance from their assigned detections.

        Returns (status, missing_items) where status is one of:
          - "Compliant":     every required item is PRESENT.
          - "Non-compliant": at least one required item is ABSENT.
          - "Uncertain":     no item is ABSENT, but at least one is UNKNOWN
                             (not enough evidence to confirm compliance OR a
                             violation). Not an alertable violation on its own.

        `missing_items` are the items in the ABSENT state (the actionable ones).
        """
        states = self.item_states(detected_items)
        absent = {i for i, s in states.items() if s == "ABSENT"}
        unknown = {i for i, s in states.items() if s == "UNKNOWN"}

        if absent:
            return "Non-compliant", absent
        if unknown:
            return "Uncertain", set()
        return "Compliant", set()
