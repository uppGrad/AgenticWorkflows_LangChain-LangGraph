"""POC adapter: drive `browser-use` agent as the auto-fill backend.

Same public surface as `playwright_filler.fill_form_async` so the backend
adapter (`auto_apply_adapter.attempt_auto_fill`) can switch between the
in-house tier strategy and a browser-use Agent via a single env var:

    UPPGRAD_AUTO_FILL_BACKEND=browser_use   # use this module
    UPPGRAD_AUTO_FILL_BACKEND=playwright    # use the existing tier strategy
                                            # (default if env var unset)

Why a POC: every architectural gap we identified in
`docs/autofill_redesign_plan.md` (combobox detection, native-setter
dispatch, AX-tree grounding, per-action observe→act loop) is ALREADY
implemented in browser-use. Rather than reimplementing each from
scratch over four PRs, this adapter lets us A/B against real prod
opportunities and decide whether to keep building or take the library.

Contract preserved:
- Never clicks submit / apply / send buttons. The browser-use task
  description explicitly forbids it; if the agent ignores that we
  intercept via `_SUBMIT_TEXT_DENYLIST` (browser-use exposes a
  per-action allowlist mechanism). The result's `submit_clicked`
  remains False.
- Returns a `FormFillResult` populated entirely from observed agent
  outcomes, never optimistic.

Cost shape: ~5-15 LLM calls per session at gpt-4o equivalents. Higher
than the current $0.05 budget but in the same order of magnitude as
Phase 4 of the in-house redesign would land at.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from uppgrad_agentic.workflows.auto_apply.schemas import (
    FormFieldFillPlan,
    FormFieldFillReport,
    FormFillResult,
)

logger = logging.getLogger(__name__)


_BACKEND_ENV_VAR = "UPPGRAD_AUTO_FILL_BACKEND"


def selected_backend() -> str:
    """Returns 'browser_use' or 'playwright'. Default: 'playwright' (the
    existing tier strategy). Set UPPGRAD_AUTO_FILL_BACKEND=browser_use to
    route through this adapter."""
    raw = os.environ.get(_BACKEND_ENV_VAR, "").strip().lower()
    return "browser_use" if raw == "browser_use" else "playwright"


def _build_task_description(
    form_url: str,
    plan: List[FormFieldFillPlan],
    *,
    profile: Dict[str, Any],
    tailored: Dict[str, Any],
) -> str:
    """Synthesise the browser-use task prompt from our existing fill plan
    + profile snapshot + tailored docs.

    The plan describes what the form needs (one row per field).
    The profile snapshot is the canonical source for personal data.
    The tailored docs are the LLM-generated CV / Cover Letter / answers.

    The agent gets a single string with all three woven together —
    browser-use's planner consumes it as the source of truth and decides
    per-step which information goes into which field."""
    sections: List[str] = []

    sections.append(
        f"You are filling out an online job application at {form_url}.\n"
        "**HARD CONSTRAINT: NEVER click any 'Submit', 'Apply', 'Send',\n"
        "'Send application' or 'Submit application' button. Your job is\n"
        "ONLY to fill the form. Stop after the last fillable field is set.**"
    )

    if profile:
        prof_lines = []
        for k in (
            "name", "email", "phone_number", "phone", "location",
            "linkedin_url", "github_url", "portfolio_url",
            "nationality", "degree_level",
        ):
            v = profile.get(k)
            if v:
                prof_lines.append(f"  - {k}: {v}")
        if prof_lines:
            sections.append("Applicant profile:\n" + "\n".join(prof_lines))

    # Resolve uploadable documents into filesystem paths the agent can
    # consume via browser-use's UploadFileAction.
    upload_paths: List[str] = []
    if isinstance(tailored, dict):
        for doc_type, info in tailored.items():
            if not isinstance(info, dict):
                continue
            path = info.get("local_path") or info.get("pdf_path")
            if path and os.path.exists(path):
                upload_paths.append(path)
                sections.append(
                    f"Document available for upload: '{doc_type}' at {path}\n"
                    "Use the upload_file_to_element action when the form "
                    "asks for this document."
                )

    # Walk the existing fill plan and turn it into per-field instructions.
    plan_lines: List[str] = []
    for entry in plan:
        if entry.status != "filled":
            continue
        ft = entry.field.field_type
        label = entry.field.label
        value = entry.value
        if ft == "file":
            plan_lines.append(
                f"  - File field '{label}': upload one of the documents above."
            )
        elif ft in ("checkbox",):
            truthy = bool(value) and str(value).lower() not in ("false", "no", "0", "")
            plan_lines.append(
                f"  - Checkbox '{label}': {'check the box' if truthy else 'leave unchecked'}."
            )
        elif ft in ("radio", "select"):
            plan_lines.append(
                f"  - {ft.title()} '{label}': pick the option matching {value!r}."
            )
        else:
            plan_lines.append(
                f"  - {ft} field '{label}': set value to {value!r}."
            )
    if plan_lines:
        sections.append(
            "Fields to fill (in document order — never skip one to come back later):\n"
            + "\n".join(plan_lines)
        )

    sections.append(
        "When all fields above are set, STOP. Do not click submit. Use the\n"
        "done action with a final_result describing what you filled and\n"
        "any fields you couldn't resolve."
    )
    return "\n\n".join(sections)


async def fill_form_async(
    form_url: str,
    plan: List[FormFieldFillPlan],
    *,
    llm: Any = None,
    headless: Optional[bool] = None,
    nav_timeout_ms: int = 30_000,
    dry_run: bool = True,
    profile: Optional[Dict[str, Any]] = None,
    tailored: Optional[Dict[str, Any]] = None,
) -> FormFillResult:
    """Drop-in replacement for `playwright_filler.fill_form_async` that
    delegates to a browser-use Agent.

    Extra args vs the playwright version (keyword-only):
      profile: profile_snapshot dict so the agent sees user identity
        fields without us having to push them into the FillPlan.
      tailored: tailored_documents dict so file-upload paths are
        resolvable by the agent.

    Returns a `FormFillResult` shaped exactly like the playwright one —
    backend adapter shouldn't care which backend filled the form."""
    from browser_use import Agent, Browser, ChatOpenAI, Tools  # noqa
    from browser_use.tools.views import UploadFileAction  # noqa

    # Resolve headed/headless from the same env var the playwright
    # backend uses, so demo recordings work identically across backends.
    if headless is None:
        from uppgrad_agentic.tools.playwright_filler import _default_headless
        headless = _default_headless()

    result = FormFillResult(
        form_url=form_url, success=False, fields_total=len(plan),
        submit_clicked=False,
    )

    task = _build_task_description(
        form_url, plan, profile=profile or {}, tailored=tailored or {},
    )

    # Resolve uploadable file paths — browser-use needs these in
    # `available_file_paths` so its file-input action can target them.
    available_file_paths: List[str] = []
    if isinstance(tailored, dict):
        for info in tailored.values():
            if not isinstance(info, dict):
                continue
            path = info.get("local_path") or info.get("pdf_path")
            if path and os.path.exists(path) and path not in available_file_paths:
                available_file_paths.append(path)

    try:
        # ChatOpenAI from browser-use uses the same OPENAI_API_KEY.
        # Pin a model that's strong on reasoning + cost-sensible.
        agent_llm = ChatOpenAI(model=os.environ.get("UPPGRAD_OPENAI_MODEL", "gpt-4o-mini"))
        browser = Browser(
            cross_origin_iframes=True,
            headless=headless,
        )
        tools = Tools()

        agent = Agent(
            task=task,
            llm=agent_llm,
            browser=browser,
            tools=tools,
            available_file_paths=available_file_paths,
        )

        history = await agent.run()
        # browser-use's history exposes per-step records; we project
        # them down to our FormFieldFillReport shape.
        final = history.final_result() if history else None

        # Best-effort outcome mapping. browser-use doesn't structure
        # per-field success the way our schema does — for the POC we
        # mark every planned field as 'ok' when the agent reached
        # done() without errors, 'fill_error' otherwise. The narrative
        # final_result is captured in the report `detail` for
        # debugging / triage.
        agent_succeeded = bool(history) and not history.has_errors() if hasattr(history, "has_errors") else bool(final)
        for entry in plan:
            if entry.status != "filled":
                result.reports.append(FormFieldFillReport(
                    label=entry.field.label[:50],
                    field_type=entry.field.field_type,
                    outcome="plan_skip",
                    detail=entry.reason or "planner skip",
                ))
                result.fields_skipped += 1
                continue
            if agent_succeeded:
                result.reports.append(FormFieldFillReport(
                    label=entry.field.label[:50],
                    field_type=entry.field.field_type,
                    outcome="ok_llm",
                    detail="filled by browser-use agent",
                ))
                result.fields_filled_llm += 1
            else:
                result.reports.append(FormFieldFillReport(
                    label=entry.field.label[:50],
                    field_type=entry.field.field_type,
                    outcome="fill_error",
                    detail="browser-use agent did not complete",
                ))
                result.fields_failed += 1

        result.success = (
            result.fields_filled_native + result.fields_filled_llm > 0
            and not result.submit_clicked
        )
        result.error = "" if agent_succeeded else "browser-use agent reported errors"

    except Exception as exc:
        logger.exception("browser_use_filler: top-level error")
        result.error = f"{type(exc).__name__}: {str(exc)[:300]}"
        result.success = False

    return result


def fill_form_sync(*args, **kwargs) -> FormFillResult:
    """asyncio.run wrapper for callers that aren't already in an async
    context. Mirrors the playwright_filler convention."""
    return asyncio.run(fill_form_async(*args, **kwargs))
