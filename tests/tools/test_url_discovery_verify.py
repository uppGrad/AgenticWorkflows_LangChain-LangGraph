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


# ─── ATS slug-mismatch guard (Bug #9 — surfaced live on GitHub 199838) ──────

def test_ats_url_slug_mismatch_rejects_match():
    """For ATS tier, the URL host+path identifies the employer definitively
    (`boards.greenhouse.io/<company-slug>/...`, `jobs.lever.co/<slug>/...`).
    A candidate whose ATS slug does NOT match the queried company must NOT
    pass verification — even when title fuzz is 100, the company name appears
    in the body as a required-tools mention, AND description keywords overlap.
    Live failure shape (GitHub 199838 → Forma.ai posting):
        passed=True, confidence=0.80
        reasons=['title fuzzy 100.0', 'company match', 'keywords 5/10', 'corroborators 2/2']"""
    description = (
        "We are hiring a Senior Solutions Engineer to support sales and customer "
        "success teams. Strong experience with kubernetes, terraform, and "
        "databricks integration is a plus. Familiarity with sales engineering "
        "workflows and partner programs required."
    )
    inputs = VerifyInputs(
        candidate_url="https://job-boards.greenhouse.io/formaaiinc/jobs/4687346005",
        candidate_title="Senior Solutions Engineer at Forma.ai",
        candidate_text=(
            "Senior Solutions Engineer at Forma.ai. Build working knowledge of "
            "our data architecture, repositories, and tooling (e.g., Databricks, "
            "S3, GitHub). Strong experience with kubernetes and terraform. "
            "Sales engineering workflows. Customer success integration. "
            + "Detailed role content. " * 200
        ),
        candidate_posted_at=None,
        job=_job(title="Senior Solutions Engineer", company="GitHub",
                 location="Germany", description=description),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is False, (
        f"slug 'formaaiinc' does not match company 'GitHub'; expected reject. "
        f"reasons={score.reasons}"
    )


def test_ats_url_slug_match_allows_match():
    """When the ATS slug matches the company, ATS verification proceeds normally."""
    inputs = VerifyInputs(
        candidate_url="https://boards.greenhouse.io/github/jobs/4554047",
        candidate_title="Senior Solutions Engineer at GitHub",
        candidate_text="Senior Solutions Engineer at GitHub. Berlin, Germany. " * 50,
        candidate_posted_at=None,
        job=_job(title="Senior Solutions Engineer", company="GitHub", location="Germany"),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True


def test_ats_url_slug_match_with_normalized_company_name():
    """The slug check should be tolerant to common normalization: lowercase,
    strip non-alphanumeric, partial substring. `notionhq` slug should match
    company `Notion`."""
    inputs = VerifyInputs(
        candidate_url="https://jobs.ashbyhq.com/notionhq/role-1",
        candidate_title="Solutions Engineer, EMEA at Notion",
        candidate_text="Solutions Engineer, EMEA at Notion. Dublin, Ireland. " * 50,
        candidate_posted_at=None,
        job=_job(title="Solutions Engineer, EMEA", company="Notion",
                 location="Dublin, County Dublin, Ireland"),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True


def test_ats_url_with_unknown_host_falls_back_to_text_check():
    """If the candidate URL's host isn't a known ATS pattern we recognize, the
    slug-extraction can't run — we must fall back to the existing text-based
    company corroborator without rejecting outright."""
    inputs = VerifyInputs(
        candidate_url="https://unknown-board.example.com/role/123",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring in London, UK. " * 50,
        candidate_posted_at=None,
        job=_job(),  # company "Acme Corp"
        tier="ats",
    )
    score = score_candidate(inputs)
    # Falls back to text-based verification; with company + location
    # corroborators that's 2/2 — passes.
    assert score.passed is True


def test_careers_tier_unaffected_by_slug_check():
    """Careers tier already constrains to the company's own domain via
    `site:<company-domain>` at search time. Slug-check shouldn't be applied."""
    inputs = VerifyInputs(
        candidate_url="https://acmecorp.com/careers/role",
        candidate_title="Senior Backend Engineer",
        candidate_text="Backend engineer in London. Apply via this form.",
        candidate_posted_at=None,
        job=_job(),
        tier="careers",
    )
    score = score_candidate(inputs)
    assert score.passed is True


# ─── Location-mismatch hard reject (multi-region postings) ──────────────────

def test_location_mismatch_rejects_when_countries_disjoint():
    """The exact live failure: user clicks the Istanbul, Türkiye listing
    of a job posted in multiple countries. The Czech Republic instance
    of the same role passes title + company-in-text + keywords. Location-
    mismatch must reject it before any of those corroborators count."""
    inputs = VerifyInputs(
        candidate_url="https://jobs.fromjimmy.com/o/senior-ai-engineer/c/new",
        candidate_title="Senior AI Engineer",
        candidate_text=(
            "Senior AI Engineer at Jimmy. Location: Prague, Czech Republic. "
            "Apply now to join our Prague-based team. Required: Python, ML."
        ),
        candidate_posted_at=None,
        job=_job(
            title="Senior AI Engineer",
            company="Jimmy",
            location="Istanbul, Türkiye",
            description="Senior AI Engineer with Python and ML.",
        ),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is False
    assert any("location mismatch" in r for r in score.reasons), score.reasons
    # Confirms 'turkey' / 'czech republic' are normalised even though the
    # source spelt it "Türkiye".
    joined = " ".join(score.reasons).lower()
    assert "turkey" in joined and "czech republic" in joined


def test_location_match_passes_when_country_overlaps():
    """Source Istanbul, Türkiye + candidate page mentions Türkiye →
    countries overlap → no rejection (existing corroborator scoring
    decides pass/fail)."""
    inputs = VerifyInputs(
        candidate_url="https://jobs.fromjimmy.com/o/senior-ai-engineer/c/new",
        candidate_title="Senior AI Engineer",
        candidate_text=(
            "Senior AI Engineer at Jimmy. Location: Istanbul, Türkiye. "
            "Required: Python, ML."
        ),
        candidate_posted_at=None,
        job=_job(
            title="Senior AI Engineer",
            company="Jimmy",
            location="Istanbul, Türkiye",
            description="Senior AI Engineer with Python and ML.",
        ),
        tier="ats",
    )
    score = score_candidate(inputs)
    assert score.passed is True


def test_remote_source_skips_location_check():
    """Source location 'Remote' / 'Worldwide' shouldn't constrain — the
    candidate page is allowed to be in any country."""
    inputs = VerifyInputs(
        candidate_url="https://acmecorp.com/careers/role",
        candidate_title="Senior Backend Engineer",
        candidate_text=(
            "Senior Backend Engineer at Acme Corp. Location: Berlin, Germany."
        ),
        candidate_posted_at=None,
        job=_job(location="Remote"),
        tier="careers",
    )
    score = score_candidate(inputs)
    assert score.passed is True


def test_location_check_skips_when_candidate_has_no_country():
    """Candidate page that's location-agnostic (no recognised country
    mentioned in the first 4000 chars) shouldn't be rejected — falls
    through to corroborator scoring."""
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring a backend engineer.",
        candidate_posted_at=None,
        job=_job(location="Istanbul, Türkiye"),
        tier="ats",
    )
    score = score_candidate(inputs)
    # Title fuzz passes, company corroborates, no location signal in
    # candidate → no rejection. Whether it ultimately passes depends on
    # corroborator count; we just assert it WASN'T rejected for location.
    if not score.passed:
        assert not any("location mismatch" in r for r in score.reasons)


def test_location_check_skips_when_source_country_unknown():
    """Source location uses a country we don't have in our canonical
    list — the check falls through (false negative is acceptable; over-
    rejecting common cases isn't)."""
    inputs = VerifyInputs(
        candidate_url="https://x.com/1",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Acme Corp is hiring in Quito, Ecuador.",
        candidate_posted_at=None,
        # 'Ecuador' isn't in _COUNTRY_VARIANTS
        job=_job(location="Lima, Peru"),
        tier="ats",
    )
    score = score_candidate(inputs)
    # No mismatch reason — the check returned None.
    assert not any("location mismatch" in r for r in score.reasons)


# ─── Location LLM rescue / disambiguation ──────────────────────────────────

def _fake_llm_returning(verdict_dict):
    """Minimal fake LLM whose `with_structured_output(...).invoke(...)`
    returns a `_LocationVerdict` instance built from the dict."""
    from unittest.mock import MagicMock
    from uppgrad_agentic.tools.url_discovery import _LocationVerdict
    structured = MagicMock()
    structured.invoke.return_value = _LocationVerdict(**verdict_dict)
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    return llm


def test_location_passes_skip_returns_no_constraint():
    """Agnostic source ('Remote') → skip verdict, no LLM call."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    llm = _fake_llm_returning({"is_same_location": False, "reason": "ignored"})
    passes, reason = _location_passes(
        "Remote", "We're hiring in Berlin, Germany.", llm=llm,
    )
    assert passes is True
    # LLM should NOT have been consulted for "skip"
    llm.with_structured_output.assert_not_called()


def test_location_passes_pass_clear_skips_llm():
    """Single-country overlap → pass_clear, no LLM call."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    llm = _fake_llm_returning({"is_same_location": False, "reason": "ignored"})
    passes, reason = _location_passes(
        "Istanbul, Türkiye",
        "Senior Engineer in Türkiye. Apply now.",
        llm=llm,
    )
    assert passes is True
    llm.with_structured_output.assert_not_called()


def test_location_passes_reject_lifted_by_llm():
    """Deterministic reject (no source country in cand) gets rescued
    when the LLM says it's actually the same location — covers
    alternative spellings / variants missing from `_COUNTRY_VARIANTS`."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    # Source mentions a country we DO have, candidate uses a variant
    # we don't (mock the LLM saying "same location" anyway).
    llm = _fake_llm_returning({
        "is_same_location": True,
        "reason": "Candidate page says 'Istanbul' which IS Türkiye",
    })
    passes, reason = _location_passes(
        "Istanbul, Türkiye",
        "We're hiring in Tokyo, Japan.",  # disjoint deterministic
        llm=llm,
    )
    assert passes is True
    assert "lifted by LLM" in reason


def test_location_passes_reject_confirmed_by_llm():
    """LLM agrees with the deterministic reject → reject stays."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    llm = _fake_llm_returning({
        "is_same_location": False,
        "reason": "Candidate is for Czech Republic only",
    })
    passes, reason = _location_passes(
        "Istanbul, Türkiye",
        "Position: Prague, Czech Republic. Apply now.",
        llm=llm,
    )
    assert passes is False
    assert "confirmed by LLM" in reason


def test_location_passes_pass_ambiguous_tightened_by_llm():
    """3+ countries on candidate page + source country among them →
    pass_ambiguous. LLM can tighten to reject when the source country
    mention is incidental rather than the actual hiring location."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    llm = _fake_llm_returning({
        "is_same_location": False,
        "reason": "Türkiye mentioned only as customer market, role is Czech",
    })
    passes, reason = _location_passes(
        "Istanbul, Türkiye",
        (
            "Senior AI Engineer. Location: Prague, Czech Republic. "
            "We serve customers across Türkiye, Germany, Japan, Brazil. "
            "Our team is in Czech Republic."
        ),
        llm=llm,
    )
    assert passes is False
    assert "tightened by LLM" in reason


def test_location_passes_pass_ambiguous_kept_by_llm():
    """3+ countries on candidate page; LLM confirms it's a legitimate
    multi-region hiring page."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    llm = _fake_llm_returning({
        "is_same_location": True,
        "reason": "Multi-region role explicitly hiring in Türkiye",
    })
    passes, reason = _location_passes(
        "Istanbul, Türkiye",
        (
            "Senior AI Engineer. Hiring across: Türkiye, Germany, "
            "Czech Republic, Brazil. Apply for any region."
        ),
        llm=llm,
    )
    assert passes is True
    assert "verified by LLM" in reason


def test_location_passes_falls_back_to_deterministic_when_llm_none():
    """`llm=None` → no LLM call; deterministic verdict stands.
    Reject case still rejects."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    passes, reason = _location_passes(
        "Istanbul, Türkiye",
        "Position: Prague, Czech Republic. Apply now.",
        llm=None,
    )
    assert passes is False
    assert "location mismatch" in reason
    # No LLM-flavour annotation
    assert "LLM" not in reason


def test_location_passes_handles_llm_exception():
    """LLM throws (network blip / structured-output unsupported / etc.)
    → _ask_llm_location_match returns None → caller falls back to
    deterministic verdict."""
    from uppgrad_agentic.tools.url_discovery import _location_passes
    from unittest.mock import MagicMock
    structured = MagicMock()
    structured.invoke.side_effect = RuntimeError("OpenAI 503")
    llm = MagicMock()
    llm.with_structured_output.return_value = structured
    passes, reason = _location_passes(
        "Istanbul, Türkiye",
        "Position: Prague, Czech Republic.",
        llm=llm,
    )
    # Falls back to deterministic reject
    assert passes is False
    assert "LLM" not in reason


def test_score_candidate_invokes_llm_for_pass_ambiguous():
    """End-to-end through score_candidate: a multi-region candidate with
    source-country mention should trigger LLM disambiguation. Patch
    get_llm to provide a controllable fake."""
    from unittest.mock import patch
    from uppgrad_agentic.tools.url_discovery import score_candidate, VerifyInputs

    fake_llm = _fake_llm_returning({
        "is_same_location": True,
        "reason": "Genuine multi-region listing covering Türkiye too",
    })
    inputs = VerifyInputs(
        candidate_url="https://job-boards.greenhouse.io/co/jobs/1",
        candidate_title="Senior AI Engineer",
        candidate_text=(
            "Senior AI Engineer at Co. Hiring across: Türkiye, "
            "Germany, Czech Republic, Brazil. Apply now."
        ),
        candidate_posted_at=None,
        job=_job(
            title="Senior AI Engineer",
            company="Co",
            location="Istanbul, Türkiye",
            description="Senior AI Engineer with Python and ML.",
        ),
        tier="ats",
    )
    with patch("uppgrad_agentic.tools.url_discovery.get_llm",
               return_value=fake_llm):
        score = score_candidate(inputs)
    assert score.passed is True
    fake_llm.with_structured_output.assert_called_once()


def test_score_candidate_skips_llm_for_clear_pass():
    """Single-country overlap (pass_clear) shouldn't waste an LLM call."""
    from unittest.mock import patch
    from uppgrad_agentic.tools.url_discovery import score_candidate, VerifyInputs

    fake_llm = _fake_llm_returning({"is_same_location": False, "reason": "x"})
    inputs = VerifyInputs(
        candidate_url="https://job-boards.greenhouse.io/co/jobs/1",
        candidate_title="Senior AI Engineer at Co",
        candidate_text="Senior AI Engineer at Co. Location: Istanbul, Türkiye.",
        candidate_posted_at=None,
        job=_job(
            title="Senior AI Engineer", company="Co",
            location="Istanbul, Türkiye",
        ),
        tier="ats",
    )
    with patch("uppgrad_agentic.tools.url_discovery.get_llm",
               return_value=fake_llm):
        score = score_candidate(inputs)
    fake_llm.with_structured_output.assert_not_called()
    # And it still passes on the regular corroborator scoring
    assert score.passed is True


def test_location_check_handles_us_uk_aliases():
    """USA / U.S. / America / UK / Britain are common variants; each
    should canonicalise to the right country and overlap correctly."""
    src_us = _job(location="San Francisco, USA")
    cand_us = "Position: San Francisco, U.S.A. Apply now."
    inputs = VerifyInputs(
        candidate_url="https://x.com/us",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text=cand_us,
        candidate_posted_at=None,
        job=src_us, tier="ats",
    )
    score = score_candidate(inputs)
    assert not any("location mismatch" in r for r in score.reasons)

    # Now disjoint US vs UK
    inputs_disjoint = VerifyInputs(
        candidate_url="https://x.com/uk",
        candidate_title="Senior Backend Engineer at Acme Corp",
        candidate_text="Position: London, United Kingdom. Apply now.",
        candidate_posted_at=None,
        job=src_us, tier="ats",
    )
    score = score_candidate(inputs_disjoint)
    assert score.passed is False
    assert any("location mismatch" in r for r in score.reasons)
