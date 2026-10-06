"""
Smoke tests for the pure-logic components (no camera / model required).
Run: python test_logic.py
"""

from detector import iou, center_in_box
from tracker import CentroidTracker


def approx(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_iou():
    # Identical boxes -> IoU 1.0
    assert approx(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
    # Disjoint boxes -> IoU 0.0
    assert approx(iou((0, 0, 10, 10), (20, 20, 30, 30)), 0.0)
    # Half overlap
    val = iou((0, 0, 10, 10), (5, 0, 15, 10))
    assert 0.0 < val < 1.0
    print("test_iou passed")


def test_center_in_box():
    assert center_in_box((4, 4, 6, 6), (0, 0, 10, 10)) is True
    assert center_in_box((40, 40, 60, 60), (0, 0, 10, 10)) is False
    print("test_center_in_box passed")


def test_tracker_persistent_ids():
    t = CentroidTracker(max_lost=3, dist_thresh=50)
    # Frame 1: two people
    r1 = t.update([(0, 0, 20, 40), (100, 0, 120, 40)])
    assert len(r1) == 2
    ids1 = set(r1.keys())

    # Frame 2: both moved slightly -> same IDs
    r2 = t.update([(2, 1, 22, 41), (103, 2, 123, 42)])
    assert set(r2.keys()) == ids1, "IDs should persist across small movement"
    print("test_tracker_persistent_ids passed")


def test_tracker_persists_through_large_movement():
    """A person walking across the frame must keep the same ID.

    Box is ~60x120 (diag ~134). Each step moves 60px, which exceeds the old
    fixed 80px-only logic over several frames, but overlap + size-scaled
    tolerance should keep the ID stable.
    """
    t = CentroidTracker(max_lost=5, dist_thresh=50)
    box = (0, 0, 60, 120)
    r = t.update([box])
    pid = next(iter(r.keys()))

    for step in range(1, 10):
        x = 60 * step
        box = (x, 0, x + 60, 120)
        r = t.update([box])
        assert list(r.keys()) == [pid], (
            f"ID changed at step {step}: expected {pid}, got {list(r.keys())}"
        )
    print("test_tracker_persists_through_large_movement passed")


def test_tracker_persists_through_brief_occlusion():
    """A person missing for a couple of frames (occlusion) keeps their ID."""
    t = CentroidTracker(max_lost=5, dist_thresh=50)
    box = (100, 100, 160, 220)
    pid = next(iter(t.update([box]).keys()))

    # Disappear for 2 frames (within max_lost)...
    t.update([])
    t.update([])
    # ...reappear slightly further along the motion path -> same ID.
    r = t.update([(130, 100, 190, 220)])
    assert list(r.keys()) == [pid], "ID should survive brief occlusion"
    print("test_tracker_persists_through_brief_occlusion passed")


def test_tracker_distinct_people_get_distinct_ids():
    t = CentroidTracker(max_lost=3, dist_thresh=50)
    r = t.update([(0, 0, 60, 120), (400, 0, 460, 120)])
    assert len(set(r.keys())) == 2, "two separate people must have two IDs"
    print("test_tracker_distinct_people_get_distinct_ids passed")


def test_tracker_new_and_retire():
    t = CentroidTracker(max_lost=2, dist_thresh=50)
    t.update([(0, 0, 20, 40)])
    # New far-away person appears -> new id
    r = t.update([(0, 0, 20, 40), (500, 500, 520, 540)])
    assert len(r) == 2
    # Person disappears for > max_lost frames -> retired
    t.update([(0, 0, 20, 40)])
    t.update([(0, 0, 20, 40)])
    t.update([(0, 0, 20, 40)])
    assert len(t.tracks) == 1, "stale track should be retired"
    print("test_tracker_new_and_retire passed")


if __name__ == "__main__":
    test_iou()
    test_center_in_box()
    test_tracker_persistent_ids()
    test_tracker_persists_through_large_movement()
    test_tracker_persists_through_brief_occlusion()
    test_tracker_distinct_people_get_distinct_ids()
    test_tracker_new_and_retire()
    print("\nAll logic tests passed.")
