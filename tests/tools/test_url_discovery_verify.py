from datetime import datetime, timedelta, timezone

from uppgrad_agentic.tools.url_discovery import (
    score_candidate, VerifyInputs, _extract_distinctive_keywords,
)


def _job(title="Senior Backend Engineer", company="Acme Corp",
         posted_iso=None, location="London, UK", description=""):
    return {
        "id": 1,
        "title": title,
        "company": company,
        "posted_time": posted_iso or datetime.now(timezone.utc).isoformat(),
        "location": location,
        "description": description,
    }


# ─── Title fuzzy gate ────────────────────────────────────────────────────────

def test_title_below_fuzz_threshold_fails():
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Marketing Coordinator",
        candidate_text="Marketing role at Acme.",
        candidate_posted_at=None,
        job=_job(),  # title = "Senior Backend Engineer"
        tier="ats",
    )
    assert score_candidate(inputs).passed is False


# ─── Multi-factor: title + ≥2 corroborators required for ATS ─────────────────

def test_ats_title_only_no_corroborators_fails():
    """Title fuzz passes but no other signal — ATS needs 2 corroborators."""
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer",
        candidate_text="A great backend role somewhere.",
        candidate_posted_at=None,
        job=_job(),  # company "Acme Corp", location "London, UK"
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is False
    assert "corroborators 0/2" in " ".join(score.reasons)


def test_ats_title_plus_company_only_fails():
    """One corroborator is not enough for ATS tier (needs 2)."""
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring",  # company match only
        candidate_posted_at=None,
        job=_job(location=""),  # location empty so it can't corroborate
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is False


def test_ats_title_plus_company_plus_location_passes():
    """Two corroborators clear the ATS bar."""
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring in London, UK.",  # company + location
        candidate_posted_at=None,
        job=_job(),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True
    assert score.confidence >= 0.7


def test_ats_title_plus_recent_posting_plus_company_passes():
    """Recent posted-time counts as a corroborator."""
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring",
        candidate_posted_at=datetime.now(timezone.utc),
        job=_job(posted_iso=datetime.now(timezone.utc).isoformat(), location=""),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True


def test_old_posting_does_not_corroborate():
    """A posting >180 days off doesn't count as a corroborator."""
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring",  # company only
        candidate_posted_at=datetime.now(timezone.utc) - timedelta(days=400),
        job=_job(posted_iso=datetime.now(timezone.utc).isoformat(), location=""),
        tier="ats",
    )
    score = score_candidate(inputs)
    # Only 1 corroborator (company) — fails ATS bar of 2
    assert score.passed is False


# ─── Description-keyword corroborator ────────────────────────────────────────

def test_description_keyword_overlap_corroborates():
    """≥3 distinctive keywords from the description appearing in the candidate page → corroborator."""
    description = (
        "We are hiring a Senior Backend Engineer to build kubernetes infrastructure. "
        "Strong python, postgres, redis experience required. Familiarity with airflow "
        "and grafana is a plus."
    )
    candidate_text = (
        "Senior Backend Engineer role at Acme Corp. You will work with kubernetes, "
        "python, postgres, and grafana to build distributed systems."
    )
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer",
        candidate_text=candidate_text,
        candidate_posted_at=None,
        job=_job(description=description, location=""),  # no company in text
        tier="ats",
    )
    # Title (passes) + company-in-text (yes Acme) + keywords (yes ≥3) = 2 corroborators
    score = score_candidate(inputs)
    assert score.passed is True
    assert any("keywords" in r for r in score.reasons)


def test_description_keyword_below_threshold_does_not_corroborate():
    """<3 keyword hits doesn't count as a corroborator."""
    description = "We are hiring a Senior Backend Engineer with kubernetes, postgres, and redis."
    candidate_text = "Senior Backend Engineer position. Apply now."  # no kw overlap
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer",
        candidate_text=candidate_text,
        candidate_posted_at=None,
        job=_job(description=description, location="", company=""),  # no other corroborators
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is False


def test_extract_distinctive_keywords_drops_stopwords():
    description = (
        "We are looking for a candidate with experience. The position requires "
        "kubernetes, python, postgres, and redis. Strong communication skills."
    )
    keywords = _extract_distinctive_keywords(description)
    # Stopwords filtered
    assert "candidate" not in keywords
    assert "experience" not in keywords
    assert "position" not in keywords
    # Distinctive terms present
    assert "kubernetes" in keywords or "postgres" in keywords or "python" in keywords


# ─── Careers tier requires only 1 corroborator ───────────────────────────────

def test_careers_tier_passes_with_one_corroborator():
    """site:<company-domain> already proves the company; 1 extra corroborator suffices."""
    inputs = VerifyInputs(
        candidate_url="https://acmecorp.com/careers/role",
        candidate_title="Senior Backend Engineer",
        candidate_text="Backend engineer in London. Apply via this form.",  # location + brief
        candidate_posted_at=None,
        job=_job(),
        tier="careers",
    )
    score = score_candidate(inputs)
    assert score.passed is True


def test_careers_tier_zero_corroborators_fails():
    """Even careers tier needs at least 1 corroborator beyond title fuzz."""
    inputs = VerifyInputs(
        candidate_url="https://acmecorp.com/careers/role",
        candidate_title="Senior Backend Engineer",
        candidate_text="Backend engineer position. Apply.",  # nothing else
        candidate_posted_at=None,
        job=_job(location="", company="", description=""),
        tier="careers",
    )
    score = score_candidate(inputs)
    assert score.passed is False


# ─── Generic tier (Tier 3) ───────────────────────────────────────────────────

def test_generic_tier_needs_two_corroborators():
    """Generic tier (no domain trust) needs 2 corroborators like ATS."""
    inputs = VerifyInputs(
        candidate_url="https://random-board.com/1",
        candidate_title="Senior Backend Engineer",
        candidate_text="Senior Backend Engineer at Acme Corp",  # company only
        candidate_posted_at=None,
        job=_job(location=""),
        tier="generic",
    )
    score = score_candidate(inputs)
    assert score.passed is False  # only 1 corroborator
