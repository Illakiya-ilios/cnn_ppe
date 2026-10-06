"""
Multi-person tracker with persistent IDs.

Matches detections to existing tracks using a combined score of bounding-box
IoU and centroid distance, with a simple constant-velocity motion prediction.
This is far more stable than pure centroid distance for people who move a
meaningful amount between processed frames (common on CPU inference).

Public interface is unchanged: construct with (max_lost, dist_thresh) and call
update(boxes) -> {track_id: box}.
"""

import math


def _centroid(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _diag(box):
    """Bounding-box diagonal length; used to scale distance to person size."""
    x1, y1, x2, y2 = box
    return math.hypot(x2 - x1, y2 - y1)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
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


class CentroidTracker:
    def __init__(self, max_lost=30, dist_thresh=80.0, iou_weight=0.5,
                 min_iou=0.1, dist_scale=1.5):
        """
        Args:
            max_lost:    frames a track may go unmatched before retirement.
            dist_thresh: baseline centroid-distance tolerance (px). The actual
                         tolerance adapts to person size and time unseen.
            iou_weight:  how much IoU contributes to the match score (0..1).
            min_iou:     IoU above this alone is enough to accept a match,
                         regardless of distance (handles large but overlapping
                         movement).
            dist_scale:  distance tolerance as a multiple of the person's
                         bounding-box diagonal (adaptive to scale/zoom).
        """
        self.max_lost = max_lost
        self.dist_thresh = dist_thresh
        self.iou_weight = iou_weight
        self.min_iou = min_iou
        self.dist_scale = dist_scale
        self.next_id = 0
        # id -> {centroid, box, vel:(vx,vy), lost:int}
        self.tracks = {}

    def _register(self, box):
        tid = self.next_id
        self.tracks[tid] = {
            "centroid": _centroid(box),
            "box": box,
            "vel": (0.0, 0.0),
            "lost": 0,
        }
        self.next_id += 1
        return tid

    def _deregister(self, tid):
        del self.tracks[tid]

    def _predicted_centroid(self, track):
        """Constant-velocity prediction, extended by how long it's been lost."""
        cx, cy = track["centroid"]
        vx, vy = track["vel"]
        steps = track["lost"] + 1
        return (cx + vx * steps, cy + vy * steps)

    def _tolerance(self, track):
        """Adaptive distance tolerance: scales with person size and time lost."""
        base = max(self.dist_thresh, self.dist_scale * _diag(track["box"]))
        # A track unseen for several frames may have moved further; widen it.
        return base * (1.0 + 0.5 * track["lost"])

    def update(self, boxes):
        """
        Update the tracker with the current frame's person boxes.

        Returns a dict mapping track_id -> box for this frame.
        """
        if not boxes:
            for tid in list(self.tracks.keys()):
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    self._deregister(tid)
            return {}

        if not self.tracks:
            return {self._register(b): b for b in boxes}

        track_ids = list(self.tracks.keys())
        input_centroids = [_centroid(b) for b in boxes]

        # Build candidate matches with a combined cost. Lower cost = better.
        # cost = (normalized distance) * (1 - iou_weight) + (1 - IoU) * iou_weight
        candidates = []
        for ti, tid in enumerate(track_ids):
            track = self.tracks[tid]
            pred = self._predicted_centroid(track)
            tol = self._tolerance(track)
            for ii, ic in enumerate(input_centroids):
                dist = math.hypot(pred[0] - ic[0], pred[1] - ic[1])
                iou = _iou(track["box"], boxes[ii])

                # Reject only if BOTH signals are poor: too far AND no overlap.
                if dist > tol and iou < self.min_iou:
                    continue

                norm_dist = min(dist / tol, 1.0) if tol > 0 else 1.0
                cost = (1.0 - self.iou_weight) * norm_dist + self.iou_weight * (1.0 - iou)
                candidates.append((cost, ti, ii))

        candidates.sort(key=lambda c: c[0])

        used_tracks = set()
        used_inputs = set()
        result = {}

        for cost, ti, ii in candidates:
            if ti in used_tracks or ii in used_inputs:
                continue
            tid = track_ids[ti]
            track = self.tracks[tid]
            new_centroid = input_centroids[ii]
            # Update velocity (smoothed) from the real last position.
            old_cx, old_cy = track["centroid"]
            steps = track["lost"] + 1
            vx = (new_centroid[0] - old_cx) / steps
            vy = (new_centroid[1] - old_cy) / steps
            track["vel"] = (0.5 * track["vel"][0] + 0.5 * vx,
                            0.5 * track["vel"][1] + 0.5 * vy)
            track["centroid"] = new_centroid
            track["box"] = boxes[ii]
            track["lost"] = 0
            result[tid] = boxes[ii]
            used_tracks.add(ti)
            used_inputs.add(ii)

        # Unmatched existing tracks -> age / retire.
        for ti, tid in enumerate(track_ids):
            if ti not in used_tracks:
                self.tracks[tid]["lost"] += 1
                if self.tracks[tid]["lost"] > self.max_lost:
                    self._deregister(tid)

        # Unmatched detections -> new tracks.
        for ii, box in enumerate(boxes):
            if ii not in used_inputs:
                result[self._register(box)] = box

        return result
