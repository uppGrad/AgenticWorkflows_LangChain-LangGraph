"""Validator semantics for the gate-1 resume payload.

Locks in the choice-allowed table so the rules don't drift:

  - skip           → optional fields only
  - ignore_for_now → always valid; on REQUIRED items it's the kill-switch
                     signal for the auto-fill module (handled at fill time,
                     not here)
  - upload         → only valid for documents (not text)
  - auto_generate  → valid for documents (unless USER_SUPPLIED type) and texts
"""
from uppgrad_agentic.workflows.auto_apply.nodes.human_gate_1 import _validate_resume


def _doc_item(*, id, required, document_type="CV"):
    return {
        "id": id, "category": "document", "label": document_type,
        "required": required, "document_type": document_type,
    }


def _text_item(*, id, required):
    return {"id": id, "category": "text", "label": "Why us?", "required": required}


def _misc_item(*, id):
    return {"id": id, "category": "misc", "label": "Profile/identity (3)", "required": False}


def _payload(reqs, misc_strategy="ignore"):
    return {"requirements": reqs, "misc_strategy": misc_strategy}


# ─── Required documents ──────────────────────────────────────────────────────

def test_required_document_accepts_upload():
    items = [_doc_item(id=0, required=True)]
    errors = _validate_resume(
        _payload({"0": {"choice": "upload", "uploaded_text": "actual cv body text"}}),
        items,
    )
    assert errors == []


def test_required_document_accepts_auto_generate():
    items = [_doc_item(id=0, required=True)]
    errors = _validate_resume(_payload({"0": {"choice": "auto_generate"}}), items)
    assert errors == []


def test_required_document_rejects_skip():
    items = [_doc_item(id=0, required=True)]
    errors = _validate_resume(_payload({"0": {"choice": "skip"}}), items)
    assert errors and any("not allowed for required document" in e for e in errors)


def test_required_document_accepts_ignore_for_now_kill_switch_signal():
    """ignore_for_now on a required document is the user's "package and
    bounce" signal — validator must NOT reject it. The auto-fill module
    later observes this in adapter code and skips the browser launch."""
    items = [_doc_item(id=0, required=True)]
    errors = _validate_resume(_payload({"0": {"choice": "ignore_for_now"}}), items)
    assert errors == []


def test_user_supplied_required_doc_rejects_auto_generate():
    items = [_doc_item(id=0, required=True, document_type="Transcript")]
    errors = _validate_resume(_payload({"0": {"choice": "auto_generate"}}), items)
    assert errors and any("Transcript" in e and "must be uploaded" in e for e in errors)


def test_required_document_upload_requires_uploaded_text():
    items = [_doc_item(id=0, required=True)]
    errors = _validate_resume(_payload({"0": {"choice": "upload"}}), items)
    assert errors and any("uploaded_text" in e for e in errors)


# ─── Optional documents ──────────────────────────────────────────────────────

def test_optional_document_accepts_skip():
    items = [_doc_item(id=0, required=False, document_type="Portfolio")]
    errors = _validate_resume(_payload({"0": {"choice": "skip"}}), items)
    assert errors == []


def test_optional_document_accepts_ignore_for_now():
    items = [_doc_item(id=0, required=False, document_type="Portfolio")]
    errors = _validate_resume(_payload({"0": {"choice": "ignore_for_now"}}), items)
    assert errors == []


# ─── Text items ──────────────────────────────────────────────────────────────

def test_required_text_rejects_upload_text_answers_are_not_files():
    """A text answer is LLM-drafted, not uploaded — `upload` is a file-side
    action and shouldn't be allowed on text RequirementItems."""
    items = [_text_item(id=0, required=True)]
    errors = _validate_resume(
        _payload({"0": {"choice": "upload", "uploaded_text": "x"}}),
        items,
    )
    assert errors and any("not allowed for required text" in e for e in errors)


def test_required_text_rejects_skip():
    items = [_text_item(id=0, required=True)]
    errors = _validate_resume(_payload({"0": {"choice": "skip"}}), items)
    assert errors and any("not allowed for required text" in e for e in errors)


def test_required_text_accepts_ignore_for_now_kill_switch_signal():
    items = [_text_item(id=0, required=True)]
    errors = _validate_resume(_payload({"0": {"choice": "ignore_for_now"}}), items)
    assert errors == []


def test_optional_text_accepts_skip():
    items = [_text_item(id=0, required=False)]
    errors = _validate_resume(_payload({"0": {"choice": "skip"}}), items)
    assert errors == []


def test_optional_text_rejects_upload():
    items = [_text_item(id=0, required=False)]
    errors = _validate_resume(
        _payload({"0": {"choice": "upload", "uploaded_text": "x"}}),
        items,
    )
    assert errors and any("not allowed for optional text" in e for e in errors)


# ─── Misc — out of scope of per-id validation ────────────────────────────────

def test_misc_per_id_entries_are_a_no_op_by_contract():
    """Frontends route misc verdicts via misc_strategy, not requirements[<id>].
    A stray per-id entry passes through silently (no allowed-choices row for
    misc). Documented for clarity; not a hard reject."""
    items = [_misc_item(id=0)]
    errors = _validate_resume(_payload({"0": {"choice": "skip"}}), items)
    assert errors == []


# ─── Cross-cutting ───────────────────────────────────────────────────────────

def test_unknown_id_rejected():
    items = [_doc_item(id=0, required=True)]
    errors = _validate_resume(
        _payload({"99": {"choice": "auto_generate"}}),
        items,
    )
    assert errors and any("unknown requirement id" in e for e in errors)


def test_invalid_misc_strategy_rejected():
    items = []
    errors = _validate_resume(_payload({}, misc_strategy="bogus"), items)
    assert errors and any("misc_strategy" in e for e in errors)


def test_user_prompt_max_200_chars():
    items = [_doc_item(id=0, required=False, document_type="Portfolio")]
    errors = _validate_resume(
        _payload({"0": {"choice": "auto_generate", "user_prompt": "x" * 201}}),
        items,
    )
    assert errors and any("exceeds" in e and "200" in e for e in errors)
