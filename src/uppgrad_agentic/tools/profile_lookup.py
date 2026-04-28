"""Map a form-field label to a value pulled from the student's profile.

Pure helper — no LangGraph imports, no global state. Given a label string and
a profile dict, returns the best-match value or None.

Key→profile mapping is keyword-based, ordered. The first ordered tuple whose
keywords ALL appear in the label (case-insensitive) wins. We use ordered
matching so e.g. "address from which you plan on working" matches "location"
(via the disambiguator phrase) before generic "address" rules below.

Profile dict shape — minimum keys we look up:
    first_name, last_name, full_name, email, phone, country, city,
    location, linkedin, github, website
The dict can carry any other keys; we just ignore them.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple


# Ordered: more-specific phrases come first so they shadow broader matches.
# Each entry: (label_keywords_all_required, profile_key)
_RULES: Tuple[Tuple[Tuple[str, ...], str], ...] = (
    # Personal-name fields
    (("first name",), "first_name"),
    (("given name",), "first_name"),
    (("last name",), "last_name"),
    (("surname",), "last_name"),
    (("family name",), "last_name"),
    (("full name",), "full_name"),

    # Contact
    (("email",), "email"),
    (("phone",), "phone"),
    (("mobile",), "phone"),
    (("telephone",), "phone"),

    # Profile URLs
    (("linkedin",), "linkedin"),
    (("github",), "github"),
    (("personal website",), "website"),
    (("portfolio url",), "website"),
    (("personal site",), "website"),
    (("website",), "website"),

    # Location — "address from which you plan on working" should land on
    # location, not on "address" specifically. Keep "location" before "address".
    (("address from which",), "location"),
    (("current location",), "location"),
    (("city",), "city"),
    (("country",), "country"),
    (("location",), "location"),
    # "address" is broad — kept last among location rules so the more specific
    # phrases shadow it.
    (("home address",), "location"),
    (("mailing address",), "location"),
    (("address",), "location"),
)


def _normalize(label: str) -> str:
    return (label or "").lower().strip()


def lookup(label: str, profile: Dict[str, str]) -> Optional[str]:
    """Return profile value for `label` or None when no rule matches.

    Defensive: returns None for empty labels, missing profile keys, or empty
    profile values."""
    if not label or not profile:
        return None
    label_norm = _normalize(label)
    for keywords, key in _RULES:
        if all(kw in label_norm for kw in keywords):
            value = profile.get(key)
            if value:
                return str(value)
            return None
    return None


def lookup_many(labels: Iterable[str], profile: Dict[str, str]) -> Dict[str, Optional[str]]:
    """Convenience: bulk lookup. Returns {label: value-or-None}."""
    return {label: lookup(label, profile) for label in labels}
