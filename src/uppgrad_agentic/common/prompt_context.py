"""Render parsed_instructions and profile_snapshot as compact prompt blocks.

Used by the parallel analysis nodes (analyze_structure, analyze_style,
analyze_content_gaps, analyze_ats, analyze_opportunity_alignment) so each
analysis is informed by the user's stated focus and applicant profile rather
than producing generic feedback.

Both helpers return an empty string when there's no signal worth including,
so callers can safely concat them into a HumanMessage without padding.
"""
from __future__ import annotations

from typing import Any, Mapping


# Intents that carry no usable signal — produced by parse_user_instructions
# when the user typed nothing or something unparseable.
_NULL_INTENTS = {"", "general document feedback"}


def format_user_focus(parsed_instructions: Mapping[str, Any] | None) -> str:
    if not parsed_instructions:
        return ""

    intent = (parsed_instructions.get("intent") or "").strip()
    tone = parsed_instructions.get("tone_preferences") or []
    role = (parsed_instructions.get("target_role") or "").strip()
    program = (parsed_instructions.get("target_program") or "").strip()
    constraints = parsed_instructions.get("explicit_constraints") or []

    lines: list[str] = []
    if intent and intent.lower() not in _NULL_INTENTS:
        lines.append(f"- intent: {intent}")
    if tone:
        lines.append(f"- tone preferences: {', '.join(str(t) for t in tone)}")
    if role:
        lines.append(f"- target role: {role}")
    if program:
        lines.append(f"- target program: {program}")
    if constraints:
        # parse_user_instructions appends "[parse fallback: ...]" on LLM failure;
        # that's diagnostic noise the analysis nodes shouldn't see.
        clean = [str(c) for c in constraints if not str(c).startswith("[parse fallback")]
        if clean:
            lines.append(f"- constraints: {'; '.join(clean)}")

    if not lines:
        return ""

    return "User focus (prioritise findings that serve these goals):\n" + "\n".join(lines)


def format_profile_brief(
    profile_snapshot: Mapping[str, Any] | None,
    *,
    max_chars: int = 1500,
) -> str:
    if not profile_snapshot:
        return ""

    # Order matters: highest-signal fields first, since max_chars truncation
    # will drop the tail.
    parts: list[str] = []

    target_roles = profile_snapshot.get("target_roles") or []
    target_programs = profile_snapshot.get("target_programs") or []
    if target_roles:
        parts.append(f"- target roles: {', '.join(str(r) for r in target_roles)}")
    if target_programs:
        parts.append(f"- target programs: {', '.join(str(p) for p in target_programs)}")

    experience_level = profile_snapshot.get("experience_level")
    if experience_level:
        parts.append(f"- career stage: {experience_level}")

    location = profile_snapshot.get("location")
    if location:
        parts.append(f"- location: {location}")

    education = profile_snapshot.get("education") or []
    if education:
        ed_lines: list[str] = []
        for ed in education[:3]:
            if not isinstance(ed, Mapping):
                continue
            bits: list[str] = []
            if ed.get("degree"):
                bits.append(str(ed["degree"]))
            if ed.get("major") and str(ed.get("major")) != str(ed.get("degree")):
                bits.append(f"in {ed['major']}")
            if ed.get("institution"):
                bits.append(str(ed["institution"]))
            if ed.get("year"):
                bits.append(str(ed["year"]))
            if bits:
                ed_lines.append(", ".join(bits))
        if ed_lines:
            parts.append("- education: " + " | ".join(ed_lines))

    experience = profile_snapshot.get("experience") or []
    if experience:
        exp_lines: list[str] = []
        for exp in experience[:4]:
            if not isinstance(exp, Mapping):
                continue
            title = exp.get("title") or "?"
            company = exp.get("company") or "?"
            if title != "?" or company != "?":
                exp_lines.append(f"{title} @ {company}")
        if exp_lines:
            parts.append("- experience: " + " | ".join(exp_lines))

    projects = profile_snapshot.get("projects") or []
    if projects:
        proj_titles = [
            str(p.get("title"))
            for p in projects[:5]
            if isinstance(p, Mapping) and p.get("title")
        ]
        if proj_titles:
            parts.append("- projects: " + " | ".join(proj_titles))

    publications = profile_snapshot.get("publications") or []
    if publications:
        pub_titles = [
            str(p.get("title"))
            for p in publications[:5]
            if isinstance(p, Mapping) and p.get("title")
        ]
        if pub_titles:
            parts.append("- publications: " + " | ".join(pub_titles))

    achievements = profile_snapshot.get("achievements") or []
    if achievements:
        ach_titles = [
            str(a.get("title"))
            for a in achievements[:5]
            if isinstance(a, Mapping) and a.get("title")
        ]
        if ach_titles:
            parts.append("- achievements: " + " | ".join(ach_titles))

    skills = profile_snapshot.get("skills") or []
    if skills:
        parts.append("- skills: " + ", ".join(str(s) for s in skills[:20]))

    languages = profile_snapshot.get("languages") or []
    if languages:
        parts.append("- languages: " + ", ".join(str(l) for l in languages[:10]))

    interests = profile_snapshot.get("interests") or []
    if interests:
        parts.append("- interests: " + ", ".join(str(i) for i in interests[:10]))

    work_bits: list[str] = []
    if profile_snapshot.get("work_style"):
        work_bits.append(str(profile_snapshot["work_style"]))
    if profile_snapshot.get("work_type"):
        work_bits.append(str(profile_snapshot["work_type"]))
    if work_bits:
        parts.append("- work preferences: " + " / ".join(work_bits))

    bio = profile_snapshot.get("bio")
    if bio:
        # Bio rendered last so truncation cuts the most verbose field first.
        parts.append(f"- bio: {str(bio).strip()}")

    if not parts:
        return ""

    body = (
        "Applicant profile (use to ground findings; never invent details not present):\n"
        + "\n".join(parts)
    )
    if len(body) > max_chars:
        body = body[: max_chars - 3].rstrip() + "..."
    return body
