from datetime import datetime, timedelta, timezone

from uppgrad_agentic.tools.url_discovery import score_candidate, VerifyInputs


def _job(title="Senior Backend Engineer", company="Acme Corp",
         posted_iso=None, location="London, UK"):
    return {
        "id": 1,
        "title": title,
        "company": company,
        "posted_time": posted_iso or datetime.now(timezone.utc).isoformat(),
        "location": location,
    }


def test_perfect_match_passes():
    inputs = VerifyInputs(
        candidate_url="https://boards.greenhouse.io/acme/jobs/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp hiring Senior Backend Engineer in London. Apply now.",
        candidate_posted_at=datetime.now(timezone.utc),
        job=_job(),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True
    assert score.confidence >= 0.7


def test_title_mismatch_fails():
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Marketing Coordinator",
        candidate_text="Marketing role at Acme.",
        candidate_posted_at=None,
        job=_job(),
        tier="ats",
    )
    assert score_candidate(inputs).passed is False


def test_company_missing_fails_for_tier1():
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer",
        candidate_text="A great backend role somewhere.",
        candidate_posted_at=None,
        job=_job(),
        tier="ats",
    )
    assert score_candidate(inputs).passed is False


def test_company_missing_ok_for_careers_tier():
    inputs = VerifyInputs(
        candidate_url="https://acmecorp.com/careers/role",
        candidate_title="Senior Backend Engineer",
        candidate_text="Backend engineer position. Apply via this form.",
        candidate_posted_at=None,
        job=_job(),
        tier="careers",
    )
    assert score_candidate(inputs).passed is True


def test_old_posting_lowers_confidence():
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring",
        candidate_posted_at=datetime.now(timezone.utc) - timedelta(days=400),
        job=_job(posted_iso=datetime.now(timezone.utc).isoformat()),
        tier="ats",
    )
    assert score_candidate(inputs).confidence < 0.85


def test_tier3_uses_stricter_threshold():
    inputs = VerifyInputs(
        candidate_url="https://random-board.com/1",
        candidate_title="Senior Backend Engineer at Acme",
        candidate_text="Acme is hiring backend engineer",
        candidate_posted_at=None,
        job=_job(title="Senior Backend Engineer (Platform)", company="Acme Corp"),
        tier="generic",
    )
    assert score_candidate(inputs).passed is False
