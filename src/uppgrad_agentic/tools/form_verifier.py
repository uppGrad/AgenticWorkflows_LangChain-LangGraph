"""Vision-LLM verifier for the deterministic form discoverer's output.

The deterministic walker (`tools/form_discoverer.py`) is fast and right
on most ATS forms, but it over-confidently mis-classifies non-Greenhouse
markup: an OpenAI-style radio group laid out as a sibling list often
gets folded into a single "first radio" entry that the frontend renders
as a Yes/I-confirm checkbox. We can't hand-code a guard for every ATS,
so we ask a vision-capable LLM to compare a screenshot of the form
against the walker's JSON output and propose corrections.

The verifier runs ONCE per session, in `extract_form_fields`, BEFORE
the field list is committed to state. No background loop, no
free-form tool-using agent — just a single structured-output call that
returns edits / additions / removals against specific walker_ids.
Bounded vocabulary, deterministic execution.

Cost: ~$0.01-0.05 per session (gpt-4o-mini vision, ~1000 input tokens
+ 1 image). Latency: ~5-10s.

Toggle via env: `UPPGRAD_FORM_DISCOVERY_VERIFY=1` enables the pass.
When unset / disabled / LLM unavailable, returns the walker output
unchanged — the verifier never blocks discovery.
"""
from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


_VALID_FIELD_TYPES = (
    "file", "text", "textarea", "select", "checkbox", "radio",
    "number", "email", "url", "date", "tel",
)


class FieldEdit(BaseModel):
    """Edit one field captured by the walker. Identified by walker_id."""
    walker_id: str = Field(..., description="The _walker_id of the field to edit")
    new_label: Optional[str] = Field(
        default=None,
        description="Replacement label. Only set when the walker's label is wrong.",
    )
    new_field_type: Optional[str] = Field(
        default=None,
        description=(
            "Replacement field_type. Use only when the walker classified the "
            "wrong shape (e.g. walker said 'text' but the screenshot shows a "
            "radio group). Must be one of: file, text, textarea, select, "
            "checkbox, radio, number, email, url, date, tel."
        ),
    )
    new_options: Optional[List[str]] = Field(
        default=None,
        description=(
            "Replacement options list — set when the walker missed visible "
            "options on a select / radio / combobox (e.g. screenshot shows "
            "Yes/No buttons but the walker captured options=[])."
        ),
    )
    new_required: Optional[bool] = Field(
        default=None,
        description="Replacement required flag.",
    )
    reason: str = Field(
        default="",
        description="Short explanation. Mention the visual cue you used.",
    )


class FieldAdd(BaseModel):
    """Add a new field the walker missed entirely. Inserted at the end."""
    label: str = Field(..., description="Visible question label")
    field_type: str = Field(
        ...,
        description=(
            "Type as it would appear on the form. Must be one of: file, "
            "text, textarea, select, checkbox, radio, number, email, url, "
            "date, tel."
        ),
    )
    name: str = Field(default="", description="Best-guess `name`/`id` attr if visible")
    options: List[str] = Field(default_factory=list)
    required: bool = Field(default=False)
    reason: str = Field(default="")


class FieldRemove(BaseModel):
    """Mark a walker entry as a phantom — heading text or tooltip the
    walker wrongly captured as an input."""
    walker_id: str = Field(...)
    reason: str = Field(default="")


class VerificationVerdict(BaseModel):
    """Structured output from the vision LLM. Empty lists when the
    walker output looked correct."""
    edits: List[FieldEdit] = Field(default_factory=list)
    adds: List[FieldAdd] = Field(default_factory=list)
    removes: List[FieldRemove] = Field(default_factory=list)
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    notes: str = Field(default="")


_SYSTEM = """You are a form-extraction verifier. You receive:
1. A screenshot of an HTML application form.
2. A JSON list of fields a deterministic DOM walker extracted from the form.

Your job: identify mistakes in the walker's output by comparing it to the screenshot. Return a structured list of corrections.

The walker can fail in these ways:
- LABEL ERROR: walker label doesn't match the question heading visible on screen (e.g. walker says "Attach" because it picked up the upload button text; the actual heading above is "Resume/CV"). Generic ATS section headings ("Application", "Personal Information", "Voluntary Disclosures", "EEOC", "Demographics") are NEVER valid labels — every field must have a specific question label visible in the screenshot.
- TYPE ERROR: walker classified a field's shape wrong (e.g. walker said `field_type="text"` but the screenshot shows a clear radio group with multiple visible options; or walker said "checkbox" for what is actually a Yes/No combobox).
- MISSING OPTIONS: walker captured `options=[]` (or a meaningless single-element list like `['on']` from the underlying HTML value attribute) for a field whose options are clearly visible in the screenshot (Yes/No, Male/Female, country list, etc.). Read the visible option labels and provide them.
- MISSING FIELD: a visible interactive control (input, dropdown, radio group, checkbox group, file picker) is in the screenshot but absent from the walker's JSON. Add it.
- PHANTOM FIELD: walker included an entry that's not a user-fillable input (a heading, instructions, a tooltip). Remove it.

CRITICAL DECISION RULES:

1. Prefer EDITS over REMOVES. If a walker entry has a generic / wrong label but the screenshot clearly shows a real user-fillable control at that position, EDIT the label rather than removing the entry. Removing means the field disappears from auto-fill entirely; editing repairs it. Only `remove` when you're confident the walker captured something that is NOT an input (e.g. a static heading or paragraph).

2. NO FIELD should remain labeled with a generic ATS section heading. If the walker emitted N fields all labeled "Application" (or "Personal Information", "EEOC", etc.), every one of them needs an EDIT with the specific visible label — not a remove.

3. If you can't identify a specific label for a walker entry from the screenshot, but the entry's `field_type` looks like a real input (file, checkbox, radio, select), still emit an edit with your best-guess label rather than leaving the generic one in place. Better a slightly imperfect label than a phantom-looking generic one.

4. For radio groups: the walker often captures each radio option as a separate entry with the bare HTML value attribute (`options=['on']`). The CORRECT shape is ONE field with `field_type="radio"` and the full `options` list. If you see N walker entries that all share a label and `field_type="radio"`, edit the FIRST one to have the full options list and `remove` the duplicates.

Other constraints:
- Be CONSERVATIVE on adds. Only add a field if it's clearly visible AND missing from the walker output entirely. Don't duplicate.
- For every correction, cite the visual cue in `reason` (e.g. "screenshot shows three radio buttons with labels Yes/No/Maybe under heading 'Are you authorized?'").
- Use walker_id from the JSON exactly as given to target edits and removes.
- For adds, try to read the field's visible label verbatim.
- Set `overall_confidence` low (≤0.5) if large parts of the form are obscured / cropped / not rendered.
- Use `notes` for caveats (e.g. "screenshot only shows top half of the form").

Output schema:
- `edits`: changes to existing walker entries (PRIMARY tool)
- `adds`: new fields the walker missed entirely
- `removes`: phantom entries to drop (use sparingly; prefer edit)
- `overall_confidence`: float 0-1
- `notes`: free-text caveats
"""


def _shape_for_llm(fields: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Compact field summary for the LLM. Only the attributes that affect
    classification — drop walker internals like `_walker_id` (we re-key
    it as a top-level `walker_id` so the LLM sees a clean label)."""
    out = []
    for f in fields:
        out.append({
            "walker_id": f.get("_walker_id", "") or "",
            "label": f.get("label", ""),
            "field_type": f.get("field_type", ""),
            "name": f.get("name", ""),
            "required": bool(f.get("required")),
            "options": list(f.get("options") or [])[:20],
            "role": f.get("role", ""),
            "aria_autocomplete": f.get("aria_autocomplete", ""),
        })
    return out


def _apply_corrections(
    fields: List[Dict[str, Any]], verdict: VerificationVerdict,
) -> List[Dict[str, Any]]:
    """Apply edits / removes / adds to the walker's field list."""
    by_walker_id: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        wid = f.get("_walker_id") or ""
        if wid:
            by_walker_id[wid] = f

    # Removes first.
    remove_ids = {r.walker_id for r in (verdict.removes or [])}
    if remove_ids:
        logger.info(
            "form_verifier: removing %d phantom field(s) flagged by LLM",
            len(remove_ids),
        )

    # Edits (mutate in place so list order is preserved).
    edit_count = 0
    for edit in (verdict.edits or []):
        target = by_walker_id.get(edit.walker_id)
        if target is None:
            logger.debug(
                "form_verifier: edit references unknown walker_id=%s — ignoring",
                edit.walker_id,
            )
            continue
        if edit.new_label is not None and edit.new_label.strip():
            target["label"] = edit.new_label.strip()
        if edit.new_field_type is not None:
            ft = edit.new_field_type.strip().lower()
            if ft in _VALID_FIELD_TYPES:
                target["field_type"] = ft
        if edit.new_options is not None:
            cleaned = [o.strip() for o in edit.new_options if isinstance(o, str) and o.strip()]
            if cleaned:
                target["options"] = cleaned[:50]
        if edit.new_required is not None:
            target["required"] = bool(edit.new_required)
        edit_count += 1
    if edit_count:
        logger.info("form_verifier: applied %d edit(s) from LLM", edit_count)

    # Build the post-remove list (preserving order).
    out = [f for f in fields if (f.get("_walker_id") or "") not in remove_ids]

    # Adds (appended at the end; we don't try to insert in DOM order).
    for add in (verdict.adds or []):
        ft = (add.field_type or "").strip().lower()
        if ft not in _VALID_FIELD_TYPES:
            logger.debug(
                "form_verifier: add with invalid field_type=%r — ignoring",
                add.field_type,
            )
            continue
        label = (add.label or "").strip()
        if not label:
            continue
        out.append({
            "label": label,
            "field_type": ft,
            "name": (add.name or "").strip(),
            "required": bool(add.required),
            "options": [o.strip() for o in (add.options or []) if isinstance(o, str) and o.strip()][:50],
            "accepts_file": [],
            "expected_source": "user_answer" if ft != "file" else "user_document",
            "canonical_document_type": "",
            "role": "",
            "aria_haspopup": "",
            "aria_controls": "",
            "aria_owns": "",
            "aria_autocomplete": "",
            "list_id": "",
            "_walker_id": f"verifier_add_{len(out)}",
        })
    if verdict.adds:
        logger.info("form_verifier: appended %d new field(s) from LLM", len(verdict.adds))

    return out


def _is_enabled() -> bool:
    val = (os.environ.get("UPPGRAD_FORM_DISCOVERY_VERIFY") or "").strip().lower()
    return val in ("1", "true", "yes", "on")


def verify_fields_with_vision(
    fields: List[Dict[str, Any]],
    screenshot_bytes: Optional[bytes],
) -> List[Dict[str, Any]]:
    """Sync entrypoint. Sends the walker's output + a screenshot to a
    vision-capable LLM and applies returned corrections.

    Returns the (possibly corrected) field list. Strips any
    `_walker_id` keys before returning so callers persist clean data.
    Best-effort: returns the input unchanged on any failure.
    """
    # Strip walker_ids from the output regardless of whether verification
    # ran — they're an internal-only marker.
    def _strip(fs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [{k: v for k, v in f.items() if not k.startswith("_")} for f in fs]

    if not fields:
        return _strip(fields)
    if not _is_enabled():
        return _strip(fields)
    if not screenshot_bytes:
        logger.info(
            "form_verifier: no screenshot — returning walker output unchanged",
        )
        return _strip(fields)

    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError as exc:
        logger.warning(
            "form_verifier: langchain_openai missing — cannot verify (%s)", exc,
        )
        return _strip(fields)

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.info(
            "form_verifier: OPENAI_API_KEY unset — returning walker output unchanged",
        )
        return _strip(fields)

    # Vision-capable, cheap. Uses the openai client directly rather than
    # `common.llm.get_llm()` so the verifier model is independent of
    # UPPGRAD_OPENAI_MODEL (the agent's text model may not have vision).
    model = os.environ.get("UPPGRAD_FORM_VERIFIER_MODEL", "gpt-4o-mini")
    try:
        llm = ChatOpenAI(model=model, temperature=0, max_tokens=2000)
        structured = llm.with_structured_output(VerificationVerdict)
    except Exception as exc:
        logger.warning("form_verifier: failed to init ChatOpenAI — %s", exc)
        return _strip(fields)

    summary = _shape_for_llm(fields)
    img_b64 = base64.b64encode(screenshot_bytes).decode("utf-8")

    msg = HumanMessage(content=[
        {
            "type": "text",
            "text": (
                "Walker output (JSON):\n" + repr(summary) + "\n\n"
                "Compare the walker's output to the form screenshot below. "
                "Return only the corrections you're confident about."
            ),
        },
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        },
    ])

    try:
        verdict: VerificationVerdict = structured.invoke([
            SystemMessage(content=_SYSTEM), msg,
        ])
    except Exception as exc:
        logger.warning(
            "form_verifier: vision LLM call failed (non-blocking) — %s", exc,
        )
        return _strip(fields)

    logger.info(
        "form_verifier: verdict — edits=%d adds=%d removes=%d confidence=%.2f notes=%r",
        len(verdict.edits or []),
        len(verdict.adds or []),
        len(verdict.removes or []),
        verdict.overall_confidence,
        (verdict.notes or "")[:120],
    )

    if verdict.overall_confidence < 0.3:
        # Verifier itself flagged low confidence (e.g. screenshot
        # cropped). Apply only removes — they're the lowest-risk
        # corrections — and leave edits / adds for a future
        # re-run. Keeps the walker output mostly intact.
        verdict_safe = VerificationVerdict(
            edits=[], adds=[], removes=verdict.removes,
            overall_confidence=verdict.overall_confidence,
            notes=verdict.notes,
        )
        corrected = _apply_corrections(fields, verdict_safe)
    else:
        corrected = _apply_corrections(fields, verdict)
    return _strip(corrected)
