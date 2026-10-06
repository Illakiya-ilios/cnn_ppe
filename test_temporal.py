"""
Tests for temporal violation confirmation.

A single or sparse non-compliant frame must NOT confirm a violation; a
sustained run of non-compliant frames must. Run: python test_temporal.py
"""

from collections import deque

import config
from pipeline import CompliancePipeline


def _window(statuses):
    d = deque(maxlen=config.TEMPORAL_WINDOW)
    for s in statuses:
        d.append(s)
    return d


def main():
    cs = CompliancePipeline._confirmed_status

    # Empty -> Uncertain
    assert cs(_window([])) == "Uncertain"

    # One bad frame among many good -> NOT confirmed as violation
    seq = ["Compliant"] * 9 + ["Non-compliant"]
    assert cs(_window(seq)) == "Compliant", cs(_window(seq))

    # Sustained non-compliance -> confirmed
    seq = ["Non-compliant"] * config.TEMPORAL_WINDOW
    assert cs(_window(seq)) == "Non-compliant"

    # Majority non-compliant beyond the fraction + min frames -> confirmed
    nc = max(config.TEMPORAL_MIN_FRAMES,
             int(config.TEMPORAL_WINDOW * config.TEMPORAL_CONFIRM_FRAC) + 1)
    seq = ["Non-compliant"] * nc + ["Compliant"] * (config.TEMPORAL_WINDOW - nc)
    assert cs(_window(seq)) == "Non-compliant", (nc, cs(_window(seq)))

    # Flickery / mixed with no clear majority -> Uncertain
    seq = ["Non-compliant", "Uncertain", "Compliant", "Uncertain",
           "Non-compliant", "Uncertain"]
    assert cs(_window(seq)) == "Uncertain", cs(_window(seq))

    # A couple of non-compliant frames below min -> not confirmed
    seq = ["Non-compliant", "Non-compliant"] + ["Uncertain"] * 8
    assert cs(_window(seq)) != "Non-compliant"

    print("All temporal tests passed.")


if __name__ == "__main__":
    main()
