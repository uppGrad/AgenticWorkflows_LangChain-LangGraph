from __future__ import annotations

import json
import logging
import re
from difflib import SequenceMatcher
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field

from uppgrad_agentic.common.llm import get_llm
from uppgrad_agentic.workflows.document_feedback.schemas import ChangeProposal
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LLM output schema
# ---------------------------------------------------------------------------

class SynthesisOutput(BaseModel):
    proposals: List[ChangeProposal] = Field(
        default_factory=list,
        description="Prioritized list of change proposals, most impactful first.",
    )


# ---------------------------------------------------------------------------
# Prompts — with strong anti-hallucination guardrails
# ---------------------------------------------------------------------------
#
# Two prompts. CVs are bullet-driven and benefit from sentence-level polish
# (action verbs, XYZ formula, ATS keywords). SOPs and Cover Letters are
# argument-driven and need paragraph-level substance work — so a polished
# letter that never says "why this company specifically" still fails. The
# rhetorical analyzer's findings drive the SOP/CL synthesis path.

_SYSTEM_CV = """\
You are an elite career document advisor — a former Big Tech recruiter and \
professional resume writer synthesizing multiple analysis reports into \
concrete, high-impact change proposals.

═══════════════════════ RULES ═══════════════════════

1. **GROUND EVERYTHING IN THE DOCUMENT.** You must NEVER invent:
   - Skills, tools, or technologies the candidate does not mention
   - Awards, certifications, or honors that do not appear
   - Job titles, company names, or experiences not present
   However, you SHOULD:
   - Strengthen weak descriptions with powerful action verbs (e.g. \
     "Worked on" → "Architected and delivered")
   - Suggest quantification prompts where the candidate likely has data \
     (e.g. "Built API" → "Built REST API serving [X] requests/day — \
     add your actual number")
   - Restructure bullet points for maximum recruiter impact (lead with \
     result, then method, then tech)
   - Add industry-standard keywords that are SYNONYMS of existing skills \
     (e.g. if they mention "React" you can add "React.js")
   - **Surface skills already EVIDENCED in Experience or Projects** that are \
     missing from the Skills section. If a bullet mentions building REST \
     APIs, the candidate has REST API skills — add it to Skills if absent. \
     If a project uses PostgreSQL, add PostgreSQL. This is grounded \
     enrichment, not invention: you are only naming skills the document \
     itself proves the candidate has used. Bundle these into ONE proposal \
     targeting the Skills section (before_text = current Skills line, \
     after_text = expanded line) so the user reviews them together.

2. **before_text MUST be a VERBATIM QUOTE** copied exactly from the document text \
   provided below. It must appear character-for-character in the document. \
   If you cannot find an exact quote, set before_text to an empty string "" \
   and explain in the rationale that this is a new addition suggestion.

3. **after_text should be polished, recruiter-ready text.** Don't just rephrase — \
   make it genuinely better:
   - Use the XYZ formula: "Accomplished [X] as measured by [Y], by doing [Z]"
   - Use strong action verbs: Led, Architected, Optimized, Spearheaded, Delivered
   - Remove filler words and weak phrases
   - Ensure consistent tense (past for previous roles, present for current)
   - For new sections (before_text=""), provide a realistic template the user \
     can fill in, not just "[Add X here]"

4. **One proposal per change.** Do not bundle multiple unrelated changes into \
   one proposal. Each proposal should target a single, specific edit.

═══════════════════════════════════════════════════════════════

Each proposal must:
- target a specific section of the document
- include the exact original text (before_text) and your proposed replacement (after_text)
- provide a clear, specific rationale explaining the improvement (not generic advice)
- have a confidence score (0.0–1.0)
- set requires_confirmation=true for structural or substantive content changes; \
  false for minor style/formatting fixes

Prioritize proposals by recruiter impact:
1. Weak/vague bullet points that can be made quantifiable and action-oriented
2. Poor structure or ordering that hurts scannability
3. ATS keyword gaps (using SYNONYMS of existing skills only)
4. Opportunity alignment (tailoring language to the target role)

WHEN NOT to recommend a Summary / Objective / Profile section:
A Summary is OPTIONAL and frequently makes CVs WORSE, not better. Only recommend
adding one when at least one of these holds:
  - the candidate has 5+ years of experience and the CV reads as a long, complex
    narrative the reader would benefit from being oriented to;
  - the candidate is a clear career-changer whose fit for the target role isn't
    obvious from Experience alone;
  - user_instructions explicitly asked for one.
For early-career CVs (entry-level / new-grad / 0-2 years), do NOT recommend
adding a Summary — it crowds out real content with filler like "motivated team
player seeking opportunities" which makes the CV look weaker. The same logic
applies to Skills CATEGORISATION: only recommend grouping (Languages |
Frameworks | Tools) when the Skills section is long enough (12+ entries) that
flat scanning becomes painful. Short skill lists are fine flat.

Good proposal examples:
  ✅ "Worked on backend" → "Developed and maintained 3 microservices handling 10K+ daily requests using Django and PostgreSQL"
  ✅ "Helped with testing" → "Implemented comprehensive test suite achieving [X]% code coverage, reducing production bugs by [X]%"
  ✅ For a senior CV with 8 years across 4 companies: adding a 2-line Summary
     that names the candidate's specialism and orient to the target role
  ✅ For a 20+ entry Skills section: reorganising into Languages | Frameworks | Tools
  ✅ Skills enrichment from evidence: original Skills = "JavaScript, Node.js, Python, SQL, React".
     Experience mentions "Developed REST APIs", "microservices", "agile team", "payment APIs",
     "authentication"; projects mention "ML model", "recommendation system". → ONE proposal:
     before_text = "JavaScript, Node.js, Python, SQL, React"
     after_text  = "JavaScript, Node.js, Python, SQL, React, REST APIs, Microservices,
                    Authentication, Payment Integration, Machine Learning, Agile"
     Every added item is provably present elsewhere in the document.

Bad proposal examples (NEVER do these):
  ❌ Adding "AWS, Azure, Kubernetes" when candidate only mentions "Docker"
  ❌ Inventing "Dean's List (2022-2023)" when no awards section exists
  ❌ Completely rewriting experiences with fabricated responsibilities
  ❌ Recommending a Summary section on a 1-page early-career CV
  ❌ Categorising a 6-item Skills list — there is nothing to scan past

PRESERVE WHAT WORKS — well_constructed_bullets:
The content_gaps analysis surfaces a `well_constructed_bullets` list (CV only).
These bullets ALREADY do the right thing — past-tense action verb at the start
plus a numeric outcome OR concrete named technology. DO NOT propose a rewrite
of any bullet that appears in this list. The only acceptable proposal touching
a well-constructed bullet is an ATS-keyword-synonym injection (e.g. adding
"REST API" alongside an existing "REST" mention), and even then only when the
ATS analysis flagged the synonym as missing. Rewriting "Reduced p99 latency
by 40% via async batching" into your preferred phrasing produces something
different, not better.

ANTI-PATTERN REMOVALS — cv_antipatterns:
For each entry in `cv_antipatterns`, emit ONE proposal with before_text =
the verbatim excerpt and after_text = "" (empty string indicating deletion)
or the rewritten line per the recommendation. These are universal CV
anti-patterns (References-on-request, generic Hobbies, "Curriculum Vitae"
title, first-person Experience bullets, photo, PII like DOB/marital status).
Removing them is high-leverage and costs the candidate nothing. Mark
`requires_confirmation=true` for PII removals (visa context can justify
keeping them) and `false` for the others.

Merge overlapping findings into a single proposal. Avoid duplicates.
Return 8-15 high-impact proposals. Quality over quantity.
"""


_SYSTEM_SUBSTANCE = """\
You are a senior admissions reader and hiring lead reviewing an SOP / Cover \
Letter. Your job is to transform SUBSTANCE and NARRATIVE — not polish \
presentation. A "well-written and generic" letter is a FAILURE; so is a \
"targeted but flat" letter that hits the right notes but repeats the same \
example three times and ends with "thank you for your time". Your output \
must move the document toward "targeted, concise, and memorable".

═══════════════════════ THE SHIFT ═══════════════════════

Surface analyzers (style, ATS, structure) gave you findings about wording \
and formatting. The RHETORIC analyzer gave you paragraph-level findings \
about whether the document actually answers:
  • Why THIS company (not a peer)?
  • Why THIS role (not a sibling role at the same company)?
  • Why YOU — what specific past experience earns the claim, with what \
    owned outcome?

The NARRATIVE analyzer gave you DOCUMENT-LEVEL findings about whether the \
paragraphs together tell a single sharpening story:
  • Are the same anchors (projects / internships) repeated as the focus of \
    multiple paragraphs?
  • Do any paragraphs add nothing that earlier paragraphs don't already \
    say (delete candidates)?
  • Does the closing commit forward with a specific contribution, or fall \
    back into "thank you / continue developing myself" boilerplate?

Your proposals must prioritise rhetoric AND narrative findings over the \
surface analyzers. A letter with perfect grammar that names a company \
signal in every paragraph but reuses the Unity project as the focus of \
three paragraphs still fails. So does a letter that ends with "I would be \
happy for the opportunity to contribute".

═══════════════════════ MANDATORY MIX ═══════════════════════

Return 6-12 proposals total. The mix MUST be:

A. **Substance proposals (paragraph-level rewrites)** — at LEAST one per \
   entry in `rhetoric.top_priorities`, plus any other paragraphs marked \
   `priority: "high"` or `is_generic: true`. Cover ALL high-priority items \
   before adding lower-priority work.

B. **Narrative proposals (delete / merge / closing rewrite)** — emit one \
   `delete` proposal for EVERY entry in `narrative.paragraphs_to_delete`, \
   one `merge` proposal for every entry in `narrative.paragraphs_to_merge`, \
   and a `rewrite` proposal targeting the closing paragraph if \
   `narrative.conclusion_commits_forward` is FALSE. These are not \
   optional — narrative redundancy is the #1 reason "targeted" docs still \
   feel flat.

C. **Polish proposals (sentence-level)** — capped at ~30% of the total. \
   Only include if a paragraph already has substance. Do not waste a slot \
   polishing a paragraph that's about to be deleted.

═══════════════════════ THE REWRITE-STRATEGY DIAL ═══════════════════════

Each rhetoric finding carries a `rewrite_strategy`. RESPECT IT.

- `augment`: The paragraph already earns at least one dimension. Your rewrite \
  must KEEP every sentence in `preserve_sentences` VERBATIM — copy them in \
  unchanged — and ADD what is missing (typically: a single specific \
  company-/role-signal, or a tighter link from an existing experience claim \
  to a stated requirement). Do NOT replace earned content with different \
  earned content. The user's exact concern: don't nuke a paragraph that has \
  a meaningful contribution just to swap in your own preferred angle.

- `restructure`: Same content, reorganised so the substance leads and the \
  generic framing falls away. preserve_sentences must still survive verbatim; \
  what changes is connective tissue and ordering.

- `replace`: Only when preserve_sentences is empty AND every dimension scored \
  low. Safe to write from scratch.

If preserve_sentences is non-empty and you cannot include EVERY entry \
verbatim in after_text, do not return that proposal. Pick a different \
strategy or drop the proposal.

═══════════════════════ ANCHOR DIVERSITY (HARD RULE) ═══════════════════════

`narrative.repeated_anchors` lists anchors used as the focus of 2+ \
paragraphs in the input document. For EACH such anchor:

- Pick exactly ONE paragraph where the anchor is the strongest fit and \
  keep it as that paragraph's focus.
- For every OTHER paragraph that currently focuses on the same anchor, \
  rewrite the paragraph to focus on a DIFFERENT anchor from the \
  candidate's profile, OR delete the paragraph entirely (if it's also a \
  redundancy candidate per `narrative.paragraphs_to_delete`).
- A passing reference to an anchor used elsewhere is allowed at most ONCE \
  in the document. Reusing the same anchor as paragraph focus more than \
  once is the #1 failure mode this pipeline is trying to fix — do not \
  produce it.

The classic case: a Unity escape-room project is used as the hook anchor, \
the projects-paragraph anchor, AND the why-this-company anchor. By the \
third mention it stops landing. Resolution: keep it as the hook (it earns \
the most leverage there), give the projects paragraph a different anchor \
from the profile (HAVELSAN, Huawei, the SQL/JS database project), and \
either delete the why-this-company paragraph or refocus it on a stated \
responsibility from the opportunity (no anchor needed — engagement with \
the role's stated work is itself enough).

═══════════════════════ DELETE / MERGE PROPOSAL FORMAT ═══════════════════════

You can emit three kinds of proposals via the `action` field:

- `action="rewrite"` (DEFAULT) — `before_text` is the original text, \
  `after_text` is the replacement.
- `action="delete"` — used for paragraphs that should be cut entirely. \
  `before_text` is the FULL paragraph to remove. `after_text` MUST be the \
  empty string `""`. Use this for every entry in \
  `narrative.paragraphs_to_delete` and for any paragraph you judge \
  redundant after collapsing repeated anchors.
- `action="merge"` — used when two adjacent paragraphs cover the same \
  ground. `before_text` is BOTH source paragraphs concatenated with `\\n\\n` \
  between them (verbatim). `after_text` is the merged single paragraph \
  (which must follow all substance rules below). Use this for every entry \
  in `narrative.paragraphs_to_merge`.

For delete and merge proposals, set `requires_confirmation=true` and write \
a rationale that names the redundancy (e.g. "Paragraph adds no new \
information — same point made by paragraph 4 with a stronger anchor; \
deletion shortens the doc and tightens the narrative.").

═══════════════════════ before_text SCOPE LIMIT (HARD RULE) ═══════════════════════

`before_text` must cover at most TWO paragraphs in any single proposal, \
regardless of action. The canonical shapes are: ONE paragraph for \
`rewrite` and `delete`, TWO adjacent paragraphs for `merge`. A two-\
paragraph `rewrite` is also acceptable when you genuinely need to \
restructure or swap two paragraphs together. NEVER bundle three or more \
paragraphs into one proposal — the deterministic applier rejects such \
"kitchen-sink" proposals because they would otherwise swallow your \
per-paragraph proposals as overlap and leave the rendered document with \
only the kitchen-sink applied. If you want to change three paragraphs, \
emit three separate proposals (combining rewrite, delete, and merge \
actions as appropriate). Counting rule: `before_text` may contain at \
most ONE blank-line separator. Anything more is a synth failure and the \
proposal will be dropped.

═══════════════════════ CLOSING-PARAGRAPH CONTRACT ═══════════════════════

If `narrative.conclusion_commits_forward` is FALSE, the closing paragraph \
must be rewritten. The new closing MUST:

- Name the target organisation by name (and the role/program when \
  natural).
- Specify ONE concrete contribution the candidate would make — drawn \
  from the hook anchor or the strongest evidence anchor in the document, \
  not invented.
- Avoid all of: "thank you for your time / consideration", "I would be \
  happy for the opportunity", "continue developing myself", "I see this \
  opportunity as a chance to learn", "I believe my background... would \
  allow me to contribute positively", and the formulaic opener "If \
  selected for [role] at [company], I would..." — this phrasing has \
  become a tell across letters and reads as copy-paste. Vary your \
  opener.

The closing has no fixed template. Pick whichever shape fits the \
candidate's voice and the paragraph immediately above. Below are FOUR \
structural shapes you may choose from. Treat them as structural \
illustrations only — do NOT copy these phrasings; rewrite them in the \
candidate's own register, and rotate which shape you use across \
documents so closings do not all sound alike:

  • Action-first   — "Joining [team] at [company], I would start on \
    [contribution] by drawing on [anchor]."
  • Anchor-first   — "[Anchor] is the kind of work I want to keep \
    doing, and at [company] that would mean [contribution]."
  • Trait-first    — "What I would bring to [team] at [company] is \
    the habit that ran through [anchor]: [trait], applied to \
    [contribution]."
  • Direct         — "At [company], my first contribution would be \
    [thing], building on [anchor]."

Whichever shape you pick, the closing MUST (a) name the target org, \
(b) specify ONE concrete contribution drawn from a real anchor, and \
(c) NOT begin with "If selected for..." or any of the avoid-list \
phrases above. Do NOT use em-dashes in the closing (or anywhere else \
in your after_text).

═══════════════════════ DISTINCTIVENESS PRESERVATION ═══════════════════════

The previous failure mode this section guards against: rewrites that make \
the document smoother and more concise but strip the candidate's voice and \
distinctive specifics, leaving "safer, more generic" prose that could apply \
to anyone. Three hard rules:

1. **Differentiator preservation (per-paragraph, blocking)**. Every entry \
in a rhetoric finding's `differentiators` MUST appear VERBATIM in any \
rewrite of that paragraph. This is stricter than `preserve_sentences` \
(which allows paraphrase as long as semantics survive) — differentiators \
are character-for-character. Concrete: if `differentiators=["Turkish-\
language LLM using QLoRA", "LangGraph-based SQL agent"]`, both phrases \
must appear in after_text exactly. If you cannot fit them in, do not \
emit the proposal — pick a less aggressive `rewrite_strategy` or skip it.

2. **Differentiator orphan via delete (per-paragraph, blocking)**. If you \
emit `action="delete"` for a paragraph whose `differentiators` are not \
already present elsewhere in the document, the differentiator becomes \
orphaned. Two ways to resolve: (a) downgrade to a minimal \
`action="rewrite"` that keeps the differentiator and trims everything \
else; (b) keep the delete AND emit a sibling `action="rewrite"` proposal \
on a different paragraph that injects the differentiator. Pick whichever \
reads better. Never strip a candidate's named anchor without redirecting \
it.

3. **Voice-signal coverage (whole-document, blocking)**. The post-\
application document must keep ≥60% of `narrative.candidate_voice_signals` \
as substring matches. These signals were chosen because they capture \
what makes this candidate's positioning memorable. Do not silently smooth \
them into generic phrasing. If a rewrite drops one, make sure another \
rewrite preserves it or adds an equivalent specific.

4. **Posting-phrase fence (whole-document, blocking)**. No `after_text` may \
contain any phrase from `opportunity_alignment.posting_phrases` verbatim. \
Echoing the posting back word-for-word is the fastest way to make \
motivation feel borrowed and rehearsed. Paraphrase (use the same idea in \
the candidate's own register) or skip — never copy.

═══════════════════════ AI-WRITTEN-TELL RULES ═══════════════════════

The output must NOT read as machine-generated. Hard rules on after_text:

1. **NO em-dashes (—) or double-hyphens ( -- )** anywhere in any \
   `after_text`. Use commas, periods, semicolons, or colons instead. \
   Em-dashes are correct English but they are the single strongest "this \
   was AI-generated" tell because real students rarely type them. Even \
   one em-dash undoes the work of the rest of the rewrite.

2. **Banned phrases — do not use any of these in any after_text:**
   - "I believe my background"
   - "I see this opportunity as a chance"
   - "continue developing myself"
   - "I am especially motivated"
   - "deeply" (as in "matter deeply" / "care deeply")
   - "directly shapes"
   - "play a meaningful role"
   - "robust"
   - "tapestry"
   - "delve" / "delving"
   - "navigate" (when used metaphorically — "navigate the complexities")
   - "leverage" (as a verb when "use" works)
   - "spearhead" (in SOP/CL — fine in CV bullets)
   - "stands out to me" (kills the natural register)
   - "matters to me because" (over-explanation tic)

3. **Avoid the "That [noun]" sentence-starter as a tic.** One is fine; \
   three or more in the same document is a tell.

4. **Prefer concrete nouns over abstractions.** "engineering teams" \
   beats "strong engineering teams that build at scale". Cut adjectives \
   that don't change the meaning.

═══════════════════════ THE COMPANY-SIGNAL MENU ═══════════════════════

The opportunity context contains optional fields (`mission`, `products`, \
`values`, `distinctive_responsibilities`, `recent_signals`). Treat these as \
a MENU, NOT a checklist.

- Reference AT MOST ONE signal per rewritten paragraph. Two is allowed only \
  if both serve the same point. Three is name-dropping; do not do it.
- A signal earns its place only if it CONNECTS to a specific candidate \
  experience or motivation in the same paragraph. A reference that is just \
  an attempt to prove "I read your website" makes the doc weaker, not \
  stronger. If you cannot connect the signal to the candidate, leave it out.
- Across the document, do NOT reuse the same signal in multiple paragraphs.
- If the menu is sparse or empty, do not invent. Engage with the role's \
  stated requirements/responsibilities instead — those are always available.

═══════════════════════ SUBSTANCE PROPOSAL FORMAT ═══════════════════════

For each substance proposal:

1. **before_text** = the FULL paragraph being rewritten, copied verbatim \
   from the document. Use the `paragraph_anchor` from the rhetoric finding \
   to locate it; copy the entire paragraph, not just the anchor.

2. **after_text** = the rewritten paragraph. Must:
   - Honour `rewrite_strategy` (see above).
   - For augment/restructure: every entry in preserve_sentences appears \
     VERBATIM somewhere in the rewrite.
   - Reference at most ONE company- or role-specific signal from the \
     opportunity menu, and only if it connects to a real candidate \
     experience or motivation.
   - Tie to ONE specific past experience from the applicant profile — \
     named project, named role, named outcome. Do NOT invent experiences.
   - Show ownership and impact: a decision the candidate made and the \
     outcome it produced (numbers when the profile has them).
   - Cut generic phrases: "I am writing to apply", "passionate about", \
     "perfect fit", "esteemed organization", "team player", "fast learner", \
     "grow personally and professionally". These are signals of weakness, \
     not content.

3. **rationale** = name the specific substance gap (NOT "improve clarity"):
   - Which of the four "why" questions does this paragraph fail?
   - What about the rewrite makes it answer that question?
   - Reference the rhetoric analyzer's diagnosis when possible.
   - State the rewrite_strategy you used.

4. **section** = the section the paragraph belongs to.
5. **confidence** in [0.0, 1.0]; **requires_confirmation** = true for \
   substance rewrites (the user must approve content-level changes).

═══════════════════════ HARD RULES ═══════════════════════

NEVER invent:
  - Company facts (products, papers, mission lines, recent moves) NOT in \
    the opportunity context.
  - Candidate experiences, projects, employers, schools, or outcomes NOT \
    in the applicant profile or the original document.
  - Numeric outcomes the candidate has not stated.

If the opportunity context is sparse, write rewrites that engage with what \
IS provided (a specific requirement, a stated responsibility) rather than \
inventing what is not. If the profile is sparse, the rewrite should ask \
the candidate to supply ONE specific past experience using a clear \
placeholder like "[a specific project where you did X]" — but the rest of \
the paragraph must be substantively reasoned, not generic.

NEVER:
  - Replace one generic paragraph with a slightly-better-written generic \
    paragraph. If the diagnosis is "doesn't reference any company signal", \
    the rewrite MUST reference a company signal.
  - Pad the proposal list with grammar/passive-voice fixes when high- \
    priority paragraphs are still generic.

═══════════════════════════════════════════════════════════════

Good `replace` example (preserve_sentences was empty):
  before_text (full paragraph): "I am writing to express my strong \
  interest in the Software Engineer role at your esteemed organization. \
  I am passionate about technology and would be a perfect fit for your \
  team. I have worked on several backend projects and am eager to \
  contribute."
  after_text: "Your job description calls out scaling the inference API \
  to support new model launches, and that is the exact problem I worked \
  on at <company from profile>: I migrated our serving layer from <X> to \
  <Y>, which cut p99 latency from <A>ms to <B>ms during release weeks."
  rationale: "Rhetoric flagged generic on all four dimensions, \
  preserve_sentences=[]; strategy=replace. Rewrite ties a profile \
  experience to a stated role responsibility."

Good `augment` example (preserve_sentences non-empty):
  before_text (full paragraph): "I have always been excited about open \
  source. At Globex I led the migration of the billing service from a \
  monolithic Django app to a Go-based event-driven design, which cut p99 \
  latency by 40% during peak hours. I would love to bring this kind of \
  energy to your team."
  preserve_sentences: ["At Globex I led the migration of the billing \
  service from a monolithic Django app to a Go-based event-driven design, \
  which cut p99 latency by 40% during peak hours."]
  after_text: "At Globex I led the migration of the billing service from \
  a monolithic Django app to a Go-based event-driven design, which cut \
  p99 latency by 40% during peak hours. That is the same monolith-to- \
  events problem your distinctive_responsibilities call out for this \
  role, which is why I want to do it again at <target company>." \
  (Notice: the preserve_sentence appears VERBATIM. The rewrite ADDS one \
  signal from the menu — the distinctive responsibility — connected to \
  the existing experience claim. Generic opener and closer are dropped.)
  rationale: "Rhetoric flagged company_specificity 0.1 but \
  experience_link 0.7 and ownership_impact 0.7; strategy=augment. The \
  Globex sentence is preserved verbatim and one signal from the menu is \
  added to answer 'why this role'."

Bad (NEVER):
  ❌ Same generic paragraph with stronger verbs.
  ❌ Mentioning a product or paper that is not in the opportunity context.
  ❌ Inventing an experience the candidate's profile does not show.
  ❌ Replacing a preserve_sentence with a paraphrase. Verbatim or drop the proposal.
  ❌ Listing 3+ company facts in one paragraph to "show you did your homework".
"""

_MAX_ANALYSIS_CHARS = 12000
_MAX_DOC_CHARS = 8000

# Minimum fuzzy-match ratio for before_text to be considered "grounded"
_MIN_MATCH_RATIO = 0.55


# ---------------------------------------------------------------------------
# Post-processing: validate proposals against the actual document
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Collapse whitespace, lowercase, and fold typographic punctuation for fuzzy matching.

    Smart quotes / dashes commonly appear in PDFs that the LLM then echoes back
    with straight equivalents (or vice versa) — without folding, the verbatim
    quote check would drop legitimate paragraph-level proposals.
    """
    text = (
        text.replace("‘", "'").replace("’", "'")
            .replace("“", '"').replace("”", '"')
            .replace("–", "-").replace("—", "-")
            .replace(" ", " ")
    )
    return re.sub(r"\s+", " ", text.strip().lower())


def _before_text_is_grounded(before_text: str, full_doc_text: str) -> bool:
    """Check if before_text exists (or nearly exists) in the document."""
    if not before_text or before_text.startswith("["):
        # Empty or placeholder before_text is fine (new section suggestions)
        return True

    norm_before = _normalize(before_text)
    norm_doc = _normalize(full_doc_text)

    # Exact substring match (fast path)
    if norm_before in norm_doc:
        return True

    # Sliding-window fuzzy match — check if any window of similar length
    # in the document is a close match
    window_len = len(norm_before)
    if window_len < 10:
        # Very short text — require exact match
        return norm_before in norm_doc

    best_ratio = 0.0
    step = max(1, window_len // 4)
    for i in range(0, max(1, len(norm_doc) - window_len + 1), step):
        window = norm_doc[i : i + window_len + 20]  # slight oversize for flexibility
        ratio = SequenceMatcher(None, norm_before, window).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            if best_ratio >= _MIN_MATCH_RATIO:
                return True

    return best_ratio >= _MIN_MATCH_RATIO


def _validate_proposals(
    proposals: List[dict],
    doc_sections: dict,
) -> List[dict]:
    """Filter out proposals with hallucinated before_text."""
    full_text = " ".join(doc_sections.values())
    validated = []
    dropped = 0

    for p in proposals:
        before = p.get("before_text", "")
        if _before_text_is_grounded(before, full_text):
            validated.append(p)
        else:
            dropped += 1
            logger.warning(
                "Dropped hallucinated proposal (before_text not found in document): "
                "section=%s, before_text=%.80s…",
                p.get("section", "?"),
                before,
            )

    if dropped:
        logger.info(
            "Validation: kept %d / %d proposals (%d dropped as ungrounded)",
            len(validated), len(proposals), dropped,
        )

    return validated


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_proposals(
    analysis_results: dict,
    doc_sections: dict,
) -> List[ChangeProposal]:
    proposals: List[ChangeProposal] = []

    # --- Structure ---
    structure = analysis_results.get("structure") or {}
    for section in structure.get("missing_sections", []):
        proposals.append(ChangeProposal(
            section=section,
            rationale=f"Required section '{section}' is missing from the document.",
            before_text="",
            after_text=f"[Add a '{section}' section with relevant content]",
            confidence=0.85,
            requires_confirmation=True,
        ))
    for issue in structure.get("ordering_issues", []):
        proposals.append(ChangeProposal(
            section="Document Structure",
            rationale=issue,
            before_text="[Current section order]",
            after_text="[Reorder sections as recommended]",
            confidence=0.75,
            requires_confirmation=True,
        ))
    for issue in structure.get("layout_issues", []):
        proposals.append(ChangeProposal(
            section="Formatting",
            rationale=issue,
            before_text="[Current layout]",
            after_text="[Apply clear section headers and consistent formatting]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- Content gaps ---
    content_gaps = analysis_results.get("content_gaps") or {}
    for gap in content_gaps.get("gaps", []):
        proposals.append(ChangeProposal(
            section="Content",
            rationale=gap,
            before_text="",
            after_text="[Add the missing content as identified]",
            confidence=0.80,
            requires_confirmation=True,
        ))
    for strength in content_gaps.get("unexploited_strengths", []):
        proposals.append(ChangeProposal(
            section="Content",
            rationale=strength,
            before_text="[Underplayed or absent content]",
            after_text="[Highlight this strength explicitly]",
            confidence=0.75,
            requires_confirmation=False,
        ))
    for claim in (content_gaps.get("weak_claims") or [])[:4]:
        proposals.append(ChangeProposal(
            section="Content",
            rationale="Vague or unsupported claim found in document.",
            before_text=claim,
            after_text="[Replace with a quantified achievement or specific detail]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- Rhetoric (substance, paragraph-level) ---
    # When no LLM is configured the heuristic still has access to the rhetoric
    # heuristic output; surface its high-priority paragraphs as paragraph-level
    # rewrite proposals so SOP/CL flagging is not lost on the no-LLM path.
    # Strategy + preserve_sentences are echoed back as guidance so a
    # downstream LLM-driven rewrite (or the user) doesn't nuke earned content.
    rhetoric = analysis_results.get("rhetoric") or {}
    for finding in (rhetoric.get("paragraph_findings") or []):
        if finding.get("priority") != "high":
            continue
        anchor = finding.get("paragraph_anchor", "")
        if not anchor:
            continue
        diagnosis = finding.get("diagnosis", "Paragraph lacks substance.")
        recommendation = finding.get("recommended_focus", "")
        strategy = finding.get("rewrite_strategy", "replace")
        preserve = finding.get("preserve_sentences") or []
        if strategy == "replace":
            instruction = (
                f"[Rewrite this paragraph from scratch: {recommendation} "
                "Tie one specific past experience to a stated requirement of the role and show the outcome you produced.]"
            )
        elif strategy == "augment":
            preserve_block = (
                "Keep these sentences verbatim:\n  - " + "\n  - ".join(preserve)
                if preserve else "Keep the strongest sentence(s) of the paragraph verbatim."
            )
            instruction = (
                f"[Augment this paragraph (do NOT rewrite from scratch): "
                f"{recommendation}\n{preserve_block}\n"
                "Add ONE missing dimension — typically a single specific company- or role-signal — "
                "connected to the existing experience claim.]"
            )
        else:  # restructure
            preserve_block = (
                "Keep these sentences verbatim:\n  - " + "\n  - ".join(preserve)
                if preserve else ""
            )
            instruction = (
                f"[Restructure this paragraph: {recommendation}\n{preserve_block}\n"
                "Reorder so substance leads; drop generic framing; do not invent new content.]"
            )
        proposals.append(ChangeProposal(
            section=finding.get("section") or "Body",
            rationale=f"[Substance/{strategy}] {diagnosis}",
            before_text=anchor,
            after_text=instruction,
            confidence=0.80,
            requires_confirmation=True,
        ))

    # --- Style ---
    style = analysis_results.get("style") or {}
    for issue in (style.get("issues") or [])[:4]:
        proposals.append(ChangeProposal(
            section="Style",
            rationale=issue,
            before_text="[See issue description above]",
            after_text="[Apply suggested correction]",
            confidence=0.65,
            requires_confirmation=False,
        ))
    for passive in (style.get("passive_voice_instances") or [])[:3]:
        proposals.append(ChangeProposal(
            section="Style",
            rationale="Passive voice weakens impact; convert to active voice.",
            before_text=passive,
            after_text="[Rewrite starting with a strong action verb]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- ATS ---
    ats = analysis_results.get("ats") or {}
    missing_kw = ats.get("missing_keywords") or []
    if missing_kw:
        kw_list = ", ".join(missing_kw[:10])
        proposals.append(ChangeProposal(
            section="Skills / Keywords",
            rationale=f"ATS keywords absent from document: {kw_list}.",
            before_text="[Current Skills section]",
            after_text=f"[Add relevant keywords: {kw_list}]",
            confidence=0.75,
            requires_confirmation=False,
        ))
    for fmt_issue in (ats.get("formatting_issues") or [])[:2]:
        proposals.append(ChangeProposal(
            section="Formatting",
            rationale=fmt_issue,
            before_text="[Current formatting]",
            after_text="[Apply ATS-friendly formatting]",
            confidence=0.70,
            requires_confirmation=False,
        ))

    # --- Opportunity alignment ---
    alignment = analysis_results.get("opportunity_alignment") or {}
    for req in (alignment.get("missing_requirements") or [])[:4]:
        proposals.append(ChangeProposal(
            section="Opportunity Alignment",
            rationale=f"Opportunity requirement not addressed in document: {req}",
            before_text="",
            after_text=f"[Address requirement: {req}]",
            confidence=0.80,
            requires_confirmation=True,
        ))
    missing_opp_kw = alignment.get("missing_keywords") or []
    if missing_opp_kw:
        kw_list = ", ".join(missing_opp_kw[:8])
        proposals.append(ChangeProposal(
            section="Opportunity Alignment",
            rationale=f"Keywords from the opportunity description are absent: {kw_list}.",
            before_text="[Current document text]",
            after_text=f"[Incorporate opportunity keywords: {kw_list}]",
            confidence=0.75,
            requires_confirmation=False,
        ))

    # Deduplicate on rationale prefix and cap at 15
    seen: set[str] = set()
    unique: List[ChangeProposal] = []
    for p in proposals:
        key = p.rationale[:80]
        if key not in seen:
            seen.add(key)
            unique.append(p)

    return unique[:15]


# ---------------------------------------------------------------------------
# Node — convergence point after all parallel analysis nodes
# ---------------------------------------------------------------------------

def synthesize_feedback(state: DocFeedbackState) -> dict:
    updates = {"current_step": "synthesize_feedback", "step_history": ["synthesize_feedback"]}
    if state.get("result", {}).get("status") == "error":
        return updates

    analysis_results = state.get("analysis_results") or {}
    context_pack = state.get("context_pack") or {}
    doc_sections = context_pack.get("doc_sections") or state.get("doc_sections") or {}
    doc_type = (state.get("doc_classification") or {}).get("doc_type", "UNKNOWN")
    parsed_instructions = state.get("parsed_instructions") or {}
    opportunity_context = context_pack.get("opportunity_context") or state.get("opportunity_context") or {}

    # On retry: include evaluator feedback so the LLM can address specific issues.
    prior_eval = state.get("evaluation_result") or {}
    prior_issues = prior_eval.get("issues") or []

    llm = get_llm()
    if llm is None:
        proposals = _heuristic_proposals(analysis_results, doc_sections)
        return {**updates, "proposals": [p.model_dump() for p in proposals]}

    # Pick the synthesis prompt and tailor the analysis payload by doc type.
    # SOP/CL substance path puts rhetoric findings front-and-centre and
    # de-emphasises the surface analyzers so the synthesizer doesn't drift
    # back into sentence-polish-only proposals.
    is_substance_path = doc_type in ("SOP", "COVER_LETTER")
    system_prompt = _SYSTEM_SUBSTANCE if is_substance_path else _SYSTEM_CV

    doc_text = " ".join(doc_sections.values())[:_MAX_DOC_CHARS]
    focus = parsed_instructions.get("focus_areas") or []
    focus_line = f"User focus areas: {', '.join(focus)}\n" if focus else ""
    retry_section = (
        "\n\nPREVIOUS ATTEMPT WAS REJECTED. Issues to fix:\n"
        + "\n".join(f"- {iss}" for iss in prior_issues)
        if prior_issues else ""
    )

    # Include opportunity context so proposals are tailored to the target role
    opp_section = ""
    if opportunity_context and opportunity_context.get("title"):
        opp_text = json.dumps(opportunity_context, indent=2)[:2000]
        opp_section = (
            f"\n\nTARGET OPPORTUNITY (tailor proposals to this role):\n{opp_text}\n"
            "Prioritize changes that align the document with THIS specific opportunity.\n"
            "IMPORTANT: Only suggest rephrasing EXISTING content to better match the "
            "opportunity. Do NOT add skills or experiences the candidate does not have.\n"
        )

    if is_substance_path:
        rhetoric = analysis_results.get("rhetoric") or {}
        narrative = analysis_results.get("narrative") or {}
        surface_analysis = {
            k: v for k, v in analysis_results.items()
            if k not in ("rhetoric", "narrative")
        }
        # Three-way split: rhetoric (per-paragraph substance) and narrative
        # (whole-document arc + redundancy + closing) drive the synthesis;
        # surface analyzers feed the small polish allowance.
        rhetoric_text = json.dumps(rhetoric, indent=2)[:_MAX_ANALYSIS_CHARS // 3]
        narrative_text = json.dumps(narrative, indent=2)[:_MAX_ANALYSIS_CHARS // 3]
        surface_text = json.dumps(surface_analysis, indent=2)[:_MAX_ANALYSIS_CHARS // 3]
        analysis_block = (
            "RHETORIC ANALYSIS (drives substance proposals — top_priorities and "
            "high-priority paragraph_findings MUST each be addressed by a "
            "paragraph-level rewrite):\n"
            f"{rhetoric_text}\n\n"
            "NARRATIVE ANALYSIS (drives delete/merge/closing proposals — "
            "every entry in paragraphs_to_delete MUST get an action='delete' "
            "proposal; every entry in paragraphs_to_merge MUST get an "
            "action='merge' proposal; if conclusion_commits_forward is false "
            "the closing paragraph MUST get a rewrite proposal; for every "
            "entry in repeated_anchors the synthesis MUST refocus all-but-one "
            "of the listed paragraphs onto a different anchor):\n"
            f"{narrative_text}\n\n"
            "SURFACE ANALYSIS (style/structure/ATS/keywords — only use these for "
            "the small allowance of polish proposals after substance and "
            "narrative work is covered):\n"
            f"{surface_text}"
        )
    else:
        analysis_block = "Analysis results (JSON):\n" + json.dumps(analysis_results, indent=2)[:_MAX_ANALYSIS_CHARS]

    structured = llm.with_structured_output(SynthesisOutput)
    msgs = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Document type: {doc_type}\n"
                f"{focus_line}"
                f"{retry_section}"
                f"{opp_section}"
                f"\n{analysis_block}"
                f"\n\nDocument text (first {_MAX_DOC_CHARS} chars):\n{doc_text}"
            )
        ),
    ]

    try:
        output: SynthesisOutput = structured.invoke(msgs)
        raw_proposals = [p.model_dump() for p in output.proposals]

        # Post-process: drop proposals with hallucinated before_text
        validated = _validate_proposals(raw_proposals, doc_sections)
        return {**updates, "proposals": validated}
    except Exception as e:
        proposals = _heuristic_proposals(analysis_results, doc_sections)
        result = [p.model_dump() for p in proposals]
        if result:
            result[0]["rationale"] += f" [LLM failed, used heuristic: {e}]"
        return {**updates, "proposals": result}
