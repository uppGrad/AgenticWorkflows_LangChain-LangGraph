"""Unit tests for profile_lookup — pure keyword matcher."""
from uppgrad_agentic.tools.profile_lookup import lookup, lookup_many


_PROFILE = {
    "first_name": "Koray", "last_name": "Sevil", "full_name": "Koray Sevil",
    "email": "koraysevil@gmail.com", "phone": "+90 555 1234567",
    "country": "Turkey", "city": "Istanbul", "location": "Istanbul, Turkey",
    "linkedin": "https://www.linkedin.com/in/koraysevil",
    "github": "https://github.com/koraysevil",
    "website": "https://koraysevil.com",
}


def test_first_name():
    assert lookup("First Name", _PROFILE) == "Koray"
    assert lookup("first name *", _PROFILE) == "Koray"


def test_last_name_variants():
    assert lookup("Last Name", _PROFILE) == "Sevil"
    assert lookup("Surname", _PROFILE) == "Sevil"
    assert lookup("Family Name", _PROFILE) == "Sevil"


def test_email():
    assert lookup("Email", _PROFILE) == "koraysevil@gmail.com"
    assert lookup("Email Address", _PROFILE) == "koraysevil@gmail.com"


def test_phone_variants():
    assert lookup("Phone", _PROFILE) == "+90 555 1234567"
    assert lookup("Phone Number", _PROFILE) == "+90 555 1234567"
    assert lookup("Mobile", _PROFILE) == "+90 555 1234567"


def test_country():
    assert lookup("Country", _PROFILE) == "Turkey"
    assert lookup("Country of residence", _PROFILE) == "Turkey"


def test_linkedin_github_website():
    assert lookup("LinkedIn Profile", _PROFILE) == "https://www.linkedin.com/in/koraysevil"
    assert lookup("LinkedIn URL", _PROFILE) == "https://www.linkedin.com/in/koraysevil"
    assert lookup("GitHub", _PROFILE) == "https://github.com/koraysevil"
    assert lookup("Website", _PROFILE) == "https://koraysevil.com"
    assert lookup("Personal Website", _PROFILE) == "https://koraysevil.com"
    assert lookup("Portfolio URL", _PROFILE) == "https://koraysevil.com"


def test_address_disambiguator_lands_on_location():
    """The Greenhouse 'Address from which you plan on working' phrase should
    map to location, not address."""
    assert lookup(
        'What is the address from which you plan on working?',
        _PROFILE,
    ) == "Istanbul, Turkey"


def test_unmatched_label_returns_none():
    assert lookup("Why do you want to join us?", _PROFILE) is None
    assert lookup("Pronouns", _PROFILE) is None


def test_empty_label_returns_none():
    assert lookup("", _PROFILE) is None
    assert lookup(None, _PROFILE) is None  # type: ignore[arg-type]


def test_empty_profile_returns_none():
    assert lookup("First Name", {}) is None
    assert lookup("First Name", None) is None  # type: ignore[arg-type]


def test_missing_key_in_profile_returns_none():
    """Match on label but the profile key isn't there → None, no crash."""
    sparse_profile = {"email": "x@y.com"}
    assert lookup("First Name", sparse_profile) is None


def test_empty_value_in_profile_returns_none():
    """Profile has the key but value is empty → None (defensive)."""
    profile = {"first_name": "", "email": "x@y.com"}
    assert lookup("First Name", profile) is None
    assert lookup("Email", profile) == "x@y.com"


def test_lookup_many():
    out = lookup_many(["First Name", "Email", "Pronouns"], _PROFILE)
    assert out == {
        "First Name": "Koray",
        "Email": "koraysevil@gmail.com",
        "Pronouns": None,
    }
