"""
Unit tests for the three-state compliance logic.

With REQUIRED_EQUIPMENT = {helmet, vest} and NEGATIVE_CLASSES = {no-helmet:helmet}:
  - helmet has reliable absence evidence (no-helmet class) -> absence is ABSENT
  - vest has NO negative class -> absence is UNKNOWN, not a violation

Run: python test_compliance.py
"""


class _Stub:
    """Minimal object exposing item_states/compliance with controlled config."""
    required = {"helmet", "vest"}
    negative_classes = {"no-helmet": "helmet"}  # helmet reliably reasoned about

    from detector import PPEDetector
    item_states = PPEDetector.item_states
    compliance = PPEDetector.compliance


def main():
    s = _Stub()

    # Both present -> Compliant
    assert s.compliance({"helmet", "vest"}) == ("Compliant", set())

    # Helmet present, vest NOT detected -> vest is UNKNOWN (no neg class),
    # so NOT a violation; overall Uncertain.
    status, missing = s.compliance({"helmet"})
    assert status == "Uncertain" and missing == set(), (status, missing)

    # Explicit no-helmet -> helmet ABSENT -> Non-compliant for helmet.
    status, missing = s.compliance({"vest", "!helmet"})
    assert status == "Non-compliant" and missing == {"helmet"}, (status, missing)

    # Nothing detected at all: helmet is reliably-reasoned (ABSENT), vest UNKNOWN
    # -> an ABSENT item exists -> Non-compliant on helmet only.
    status, missing = s.compliance(set())
    assert status == "Non-compliant" and missing == {"helmet"}, (status, missing)

    # Vest present, helmet not detected -> helmet ABSENT (reliable) -> violation.
    status, missing = s.compliance({"vest"})
    assert status == "Non-compliant" and missing == {"helmet"}, (status, missing)

    # item_states sanity
    states = s.item_states({"helmet"})
    assert states["helmet"] == "PRESENT"
    assert states["vest"] == "UNKNOWN"

    print("All compliance tests passed.")


if __name__ == "__main__":
    main()
