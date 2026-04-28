"""BeautifulSoup-based form-area extractor.

Given rendered HTML from a Crawl4AI run (or any rendered page), pull out
the `<form>` subtree, strip noise (`<script>`, `<style>`, hidden fields,
analytics elements), and return a compact HTML string suitable for LLM
extraction. Preserves the structural metadata the LLM needs:
input/select/textarea/option tags with their type, name, required, and
visible labels.
"""
from uppgrad_agentic.tools.form_extractor import extract_form_html


def test_returns_form_subtree_only():
    html = """
    <html>
      <head><title>Apply</title><style>.x{}</style></head>
      <body>
        <h1>About the role</h1>
        <p>marketing copy here</p>
        <form id="apply" action="/submit" method="POST">
          <label>Resume</label>
          <input type="file" name="resume" required />
        </form>
        <footer>copyright</footer>
      </body>
    </html>
    """
    out = extract_form_html(html)
    assert "<form" in out
    assert 'name="resume"' in out
    assert "<h1" not in out
    assert "<footer" not in out
    assert "marketing copy here" not in out


def test_strips_scripts_styles_and_meta():
    html = """
    <form>
      <script>alert('xss')</script>
      <style>.hide{display:none}</style>
      <meta charset="utf-8" />
      <label>Email</label>
      <input type="email" name="email" required />
    </form>
    """
    out = extract_form_html(html)
    assert "<script" not in out
    assert "<style" not in out
    assert "<meta" not in out
    assert 'name="email"' in out


def test_drops_hidden_inputs():
    """Hidden inputs are framework plumbing (CSRF tokens, form state) — not
    user-fillable fields. They'd just confuse the LLM."""
    html = """
    <form>
      <input type="hidden" name="csrf_token" value="abc123" />
      <input type="hidden" name="form_id" value="42" />
      <label>Full name</label>
      <input type="text" name="full_name" required />
    </form>
    """
    out = extract_form_html(html)
    assert 'name="csrf_token"' not in out
    assert 'name="form_id"' not in out
    assert 'name="full_name"' in out


def test_preserves_select_options():
    """The LLM needs the option list to know what dropdown values are valid."""
    html = """
    <form>
      <label for="country">Country</label>
      <select name="country" id="country" required>
        <option value="">Choose…</option>
        <option value="US">United States</option>
        <option value="IE">Ireland</option>
        <option value="DE">Germany</option>
      </select>
    </form>
    """
    out = extract_form_html(html)
    assert 'name="country"' in out
    assert "United States" in out
    assert "Ireland" in out
    assert "Germany" in out


def test_preserves_textarea_for_long_form_questions():
    """Textareas are 'user_answer' fields (e.g. 'Why do you want to join?').
    The LLM needs to see them to know what to answer."""
    html = """
    <form>
      <label>Why do you want to join us?</label>
      <textarea name="motivation" rows="5" required></textarea>
    </form>
    """
    out = extract_form_html(html)
    assert "<textarea" in out
    assert 'name="motivation"' in out


def test_preserves_radio_and_checkbox_groups():
    html = """
    <form>
      <fieldset>
        <legend>Are you authorized to work in the EU?</legend>
        <label><input type="radio" name="eu_auth" value="yes" /> Yes</label>
        <label><input type="radio" name="eu_auth" value="no" /> No</label>
      </fieldset>
      <label><input type="checkbox" name="agree_tos" required /> I agree to the terms</label>
    </form>
    """
    out = extract_form_html(html)
    assert 'type="radio"' in out
    assert 'name="eu_auth"' in out
    assert "Yes" in out
    assert "No" in out
    assert 'type="checkbox"' in out
    assert 'name="agree_tos"' in out


def test_returns_empty_string_when_no_inputs_anywhere():
    """Pages with no inputs at all (description-only, no form/inputs in body)
    return empty so caller can skip the LLM call entirely."""
    html = "<html><body><h1>Apply via email</h1><p>Send us your CV.</p></body></html>"
    out = extract_form_html(html)
    assert out == ""


def test_falls_back_to_body_when_no_form_tag_but_inputs_exist():
    """Modern React ATSes (Ashby, newer Workday) don't use native <form>
    elements — they use <div>s with click handlers and submit via fetch.
    The body still contains the <input>/<select>/<textarea> tags the LLM
    needs. We must fall back to the body when no <form> is present."""
    html = """
    <html>
      <head><style>body{}</style></head>
      <body>
        <div class="application-shell">
          <div class="page-header">Apply for the Role</div>
          <div class="field-group">
            <label>Resume</label>
            <input type="file" name="resume" required />
          </div>
          <div class="field-group">
            <label>Country</label>
            <select name="country" required>
              <option>Ireland</option>
              <option>Germany</option>
            </select>
          </div>
          <div class="field-group">
            <label>Why us?</label>
            <textarea name="why" required></textarea>
          </div>
        </div>
      </body>
    </html>
    """
    out = extract_form_html(html)
    assert 'name="resume"' in out
    assert 'name="country"' in out
    assert 'name="why"' in out
    assert "Ireland" in out
    assert "Germany" in out
    # noise still stripped
    assert "<style" not in out


def test_form_tag_preferred_over_body_fallback():
    """When BOTH a <form> tag AND extra body inputs exist, the <form> wins
    (more focused, less noise for the LLM)."""
    html = """
    <body>
      <input type="search" name="header_search" />
      <form>
        <label>Resume</label><input type="file" name="resume" required />
        <label>Email</label><input type="email" name="email" required />
      </form>
      <footer><input type="email" name="newsletter" /></footer>
    </body>
    """
    out = extract_form_html(html)
    assert 'name="resume"' in out
    assert 'name="email"' in out
    # body-level inputs (search bar, footer newsletter) should NOT be in the form output
    assert 'name="header_search"' not in out
    assert 'name="newsletter"' not in out


def test_returns_largest_form_when_multiple_present():
    """Pages sometimes have a tiny search/login form alongside the apply form;
    we pick the one with the most input fields as the application form."""
    html = """
    <body>
      <form id="search"><input type="search" name="q" /></form>
      <form id="apply">
        <label>Resume</label><input type="file" name="resume" required />
        <label>Cover letter</label><input type="file" name="cover" />
        <label>Email</label><input type="email" name="email" required />
        <label>Why us?</label><textarea name="why"></textarea>
      </form>
    </body>
    """
    out = extract_form_html(html)
    assert 'name="resume"' in out
    assert 'name="cover"' in out
    assert 'name="why"' in out
    assert 'name="q"' not in out


def test_handles_empty_or_invalid_html_gracefully():
    assert extract_form_html("") == ""
    assert extract_form_html("<not html>") == ""
    assert extract_form_html(None) == ""  # type: ignore[arg-type]


# ─── ATS iframe detection (for company-direct careers pages that embed
#      a third-party ATS form via cross-origin iframe — MongoDB→Greenhouse,
#      Stripe→Greenhouse, etc.) ──────────────────────────────────────────

from uppgrad_agentic.tools.form_extractor import extract_ats_iframe_src


def test_finds_greenhouse_embed_iframe():
    """Companies like MongoDB / Stripe embed Greenhouse via
    <iframe id="grnhse_iframe" src="https://job-boards.greenhouse.io/embed/...">.
    The form lives in that cross-origin iframe; we must follow it."""
    html = """
    <html><body>
      <h1>Senior Engineer</h1>
      <p>Job description ...</p>
      <iframe id="grnhse_iframe" title="Greenhouse Job Board"
              src="https://job-boards.greenhouse.io/embed/job_app?for=mongodb&token=7484657"
              height="3142"></iframe>
    </body></html>
    """
    src = extract_ats_iframe_src(html)
    assert src == "https://job-boards.greenhouse.io/embed/job_app?for=mongodb&token=7484657"


def test_finds_lever_embed_iframe():
    html = """
    <html><body>
      <iframe src="https://jobs.lever.co/acme/abc-123" width="100%"></iframe>
    </body></html>
    """
    src = extract_ats_iframe_src(html)
    assert src is not None
    assert "lever.co" in src


def test_finds_ashby_embed_iframe():
    html = """
    <html><body>
      <iframe src="https://jobs.ashbyhq.com/acme/role-1/application"></iframe>
    </body></html>
    """
    src = extract_ats_iframe_src(html)
    assert src is not None
    assert "ashbyhq.com" in src


def test_ignores_non_ats_iframes():
    """GTM, Optimizely, ad-tracking iframes are noise. We only follow iframes
    whose src points at a known ATS domain."""
    html = """
    <html><body>
      <iframe src="https://www.googletagmanager.com/ns.html?id=GTM-X"></iframe>
      <iframe src="https://insight.adsrvr.org/track/cei?advertiser_id=foo"></iframe>
      <iframe src="about:blank"></iframe>
    </body></html>
    """
    assert extract_ats_iframe_src(html) is None


def test_returns_none_when_no_iframes_at_all():
    assert extract_ats_iframe_src("<html><body>no iframes</body></html>") is None
    assert extract_ats_iframe_src("") is None


def test_prefers_first_ats_iframe_when_multiple():
    """If multiple ATS iframes exist (rare, but defensive), pick the first one
    in document order."""
    html = """
    <html><body>
      <iframe src="https://job-boards.greenhouse.io/embed/job_app?for=acme&token=1"></iframe>
      <iframe src="https://jobs.lever.co/acme/2"></iframe>
    </body></html>
    """
    src = extract_ats_iframe_src(html)
    assert src is not None
    assert "greenhouse" in src
