"""
Unit tests for the three-state compliance logic.

Corrected rule: ABSENT requires an EXPLICIT negative detection ("!helmet").
"Neither the item nor its negative class detected" is UNKNOWN, never a
violation -- because the model simply produced no evidence either way.

With REQUIRED_EQUIPMENT = {helmet, vest} and NEGATIVE_CLASSES = {no-helmet:helmet}:
  - helmet detected        -> PRESENT
  - !helmet (no-helmet box) -> ABSENT  (violation)
  - helmet not detected, no !helmet -> UNKNOWN (NOT a violation)
  - vest has no negative class -> absence is always UNKNOWN

Run: python test_compliance.py
"""


class _Stub:
    required = {"helmet", "vest"}
    negative_classes = {"no-helmet": "helmet"}

    from detector import PPEDetector
    item_states = PPEDetector.item_states
    compliance = PPEDetector.compliance


def main():
    s = _Stub()

    # Both detected -> Compliant
    assert s.compliance({"helmet", "vest"}) == ("Compliant", set())

    # Helmet present, vest not detected -> vest UNKNOWN -> Uncertain (not violation)
    status, missing = s.compliance({"helmet"})
    assert status == "Uncertain" and missing == set(), (status, missing)

    # Explicit no-helmet -> helmet ABSENT -> Non-compliant
    status, missing = s.compliance({"vest", "!helmet"})
    assert status == "Non-compliant" and missing == {"helmet"}, (status, missing)

    # THE KEY FIX: nothing detected, NO explicit negative -> everything UNKNOWN
    # -> Uncertain, NOT a violation. (Previously this wrongly returned
    # Non-compliant for helmet because helmet was in reliable_absence.)
    status, missing = s.compliance(set())
    assert status == "Uncertain" and missing == set(), (status, missing)

    # Vest detected but helmet not detected (no !helmet) -> helmet UNKNOWN,
    # vest PRESENT -> Uncertain, NOT a violation.
    status, missing = s.compliance({"vest"})
    assert status == "Uncertain" and missing == set(), (status, missing)

    # item_states sanity
    states = s.item_states({"helmet"})
    assert states["helmet"] == "PRESENT"
    assert states["vest"] == "UNKNOWN"

    states = s.item_states({"!helmet"})
    assert states["helmet"] == "ABSENT"
    assert states["vest"] == "UNKNOWN"

    print("All compliance tests passed.")


if __name__ == "__main__":
    main()
