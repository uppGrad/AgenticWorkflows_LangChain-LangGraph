from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, List
import os


@dataclass
class ExtractResult:
    text: str
    page_count: Optional[int]
    warnings: List[str]


def _read_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# Per-page trailing footer patterns. pypdf includes the page-number footer
# in extracted text, which then renders as a stray "1" paragraph between the
# closing salutation and the real PDF page number on the rendered output.
# Real CV/SOP/CL content never ends a page with a bare digit or "Page N" line.
_PAGE_NUMBER_FOOTER = re.compile(
    r"(?:\s*\n)+\s*(?:"
    r"\d{1,3}"                     # bare "1", "12"
    r"|[Pp]age\s*\d{1,3}"          # "Page 1"
    r"|\d{1,3}\s*(?:of|/)\s*\d{1,3}"  # "1 of 2", "1/2"
    r")\s*$"
)


def _strip_page_footer(page_text: str) -> str:
    """Remove a trailing page-number-only line if present."""
    return _PAGE_NUMBER_FOOTER.sub("", page_text)


# pypdf can produce two pathological extractions on narrative documents that
# both manifest downstream as broken paragraph boundaries:
#
#   (a) "Single-newline paragraph boundary" — LaTeX article PDFs typeset
#       paragraphs with first-line indent and no blank line. pypdf renders
#       this as a lone "\n" between paragraphs. Every paragraph-aware
#       analyzer downstream splits on r"\n\s*\n", so the whole body looks
#       like ONE paragraph and the synth emits one whole-body rewrite.
#
#   (b) "Word-per-line wrap" — Google Docs / Word "Sloppy SoP" exports break
#       wrapped paragraphs into a sequence of single words separated by
#       "\n \n" (newline + space-only line + newline). The synth's regex
#       matcher uses \s+ so before_text still matches, but the finalize LLM
#       sees each word on its own input line and (intermittently) renders
#       them as separate LaTeX paragraphs — every word ends up as its own
#       block in the rendered PDF.
#
# normalize_pdf_text() handles both: it (1) marks true paragraph boundaries
# wherever a sentence-ender is followed by whitespace + a capital letter,
# (2) collapses any newline run that contains intermediate horizontal
# whitespace ("\n \n", "\n  \n") into a single space — that's the "blank
# line with just spaces" pattern that signals pypdf wrap, never a real
# paragraph break — and (3) restores the marked paragraph boundaries.
# Plain single \n (e.g. between CV bullets) is left alone so bullet
# separation survives.
_PARA_MARKER = "\x00PARA\x00"
_TRUE_PARAGRAPH_BOUNDARY = re.compile(r"([.!?])[ \t]*\n[ \t\n]*(?=[A-Z])")
_WORD_WRAP_BLANK_LINE = re.compile(r"\n[ \t]+\n[ \t\n]*")


def _collapse_mid_sentence_wrap(text: str) -> str:
    """Collapse lone "\\n" inside a paragraph into a single space.

    Conservative on purpose. We collapse when ALL of:
      - the line preceding the "\\n" is long (≥ 40 chars) — short lines are
        usually CV section headers ("Education", "Skills") or single bullets
        whose line break is structural.
      - the segment ends with a word character, comma, or semicolon — never
        a sentence terminator (those are already paragraph breaks).
      - the next non-whitespace character is lowercase or an opening
        bracket — capital letters might start a new section, so we leave
        them alone.

    CV impact: bullets and headers are typically short (< 40 chars), so a
    "\\n" preceding them is preserved. Wrapped sentence continuations in
    SOP/CL prose are typically embedded in long lines, so they collapse.
    """
    out: List[str] = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        out.append(line)
        if i == len(lines) - 1:
            continue
        # Don't touch existing paragraph markers — they look like blank lines.
        if not line.strip() or not lines[i + 1].strip():
            out.append("\n")
            continue
        if len(line) < 40:
            out.append("\n")
            continue
        # Only collapse when the line ends with word/comma/semicolon and the
        # next line begins with a lowercase letter or opening bracket.
        if not re.search(r"[\w,;:)\]'\"]\s*$", line):
            out.append("\n")
            continue
        if not re.match(r"\s*[a-z(\[]", lines[i + 1]):
            out.append("\n")
            continue
        out.append(" ")
    return "".join(out)


def normalize_paragraph_breaks(text: str) -> str:
    """Normalize PDF/DOCX-extracted text into clean paragraphs.

    Robust to two pypdf failure modes (single-newline-only paragraph breaks
    and word-per-line wrap with "\\n \\n" between every word). See module
    block comment above for the full failure-mode map.

    Guards against a third failure mode introduced by step 1 itself: in a
    document that ALREADY has well-formed blank-line paragraph structure
    (regular Word/Google Docs PDF with page-width line wraps), step 1's
    "sentence-end + newline + capital" pattern over-fires inside paragraphs.
    Any in-paragraph line wrap that happens to land at a sentence boundary
    looks identical to a paragraph break, so the heuristic injects "\\n\\n"
    between sentences within a paragraph. Downstream effect: every sentence
    renders as its own paragraph, narrative analysis sees 15+
    pseudo-paragraphs and flags them as redundant, synth emits one anchor
    per pseudo-paragraph and the same anchor lands in three "paragraphs".
    Skip step 1 when blank-line structure already exists. Word-per-line
    case still triggers it (its blank lines contain horizontal whitespace
    and don't match `\\n\\n`); LaTeX article PDFs still trigger it (no
    blank lines at all).
    """
    has_blank_paragraphs = len(re.findall(r"\n\n", text)) >= 2

    if not has_blank_paragraphs:
        # Step 1: protect true paragraph boundaries (sentence-ender + newline
        # + capital letter). Consume the surrounding whitespace so we don't
        # end up with stray spaces around the marker.
        text = _TRUE_PARAGRAPH_BOUNDARY.sub(r"\1" + _PARA_MARKER, text)
    # Step 2: collapse word-per-line wrap. Any newline run that contains a
    # space-only intermediate line is the pypdf wrap signature; replace with
    # a single space so the words rejoin into one paragraph.
    text = _WORD_WRAP_BLANK_LINE.sub(" ", text)
    # Step 3: collapse mid-sentence soft wraps inside long prose lines.
    # CV bullets and section headers are short and survive untouched.
    text = _collapse_mid_sentence_wrap(text)
    # Step 4: restore true paragraph boundaries (no-op when step 1 was
    # skipped — no markers were ever inserted).
    text = text.replace(_PARA_MARKER, "\n\n")
    # Step 5: collapse runs of horizontal whitespace inserted by earlier steps.
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text


def _extract_pdf_pypdf(path: str) -> Tuple[str, int, List[str]]:
    """pypdf path. Returns (text, page_count, warnings)."""
    from pypdf import PdfReader
    warnings: List[str] = []
    reader = PdfReader(path)
    page_count = len(reader.pages)
    texts: List[str] = []
    for i, page in enumerate(reader.pages):
        try:
            t = page.extract_text() or ""
        except Exception:
            t = ""
            warnings.append(f"pypdf failed on page {i+1}.")
        if t.strip():
            texts.append(_strip_page_footer(t))
    return "\n".join(texts), page_count, warnings


def _extract_pdf_pdfminer(path: str) -> Tuple[str, List[str]]:
    """pdfminer.six path. Returns (text, warnings)."""
    from pdfminer.high_level import extract_text as _pm_extract
    warnings: List[str] = []
    try:
        text = _pm_extract(path) or ""
    except Exception as e:
        warnings.append(f"pdfminer extraction failed: {e}")
        text = ""
    return text, warnings


def _avg_paragraph_length(text: str) -> float:
    """Average paragraph length after splitting on blank lines.

    Used as the scoring signal for picking between pypdf and pdfminer.
    Higher avg = fewer fragmentary paragraphs = better extraction. The
    failure mode this protects against is word-per-line wrap, which both
    extractors hit on different PDFs (pypdf on standard Word/Google Docs
    PDFs, pdfminer on PDFs with unusual character-spacing encoding) and
    which produces 100+ tiny "paragraphs" of one word each.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return 0.0
    return sum(len(p) for p in paragraphs) / len(paragraphs)


def extract_text_from_file(path: str) -> ExtractResult:
    """
    Best-effort extraction from TXT, DOCX, PDF.
    Returns extracted text + optional page_count + warnings.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    warnings: List[str] = []
    page_count: Optional[int] = None

    if ext in [".txt", ".md"]:
        text = _read_txt(path)
        return ExtractResult(text=text, page_count=None, warnings=warnings)

    if ext == ".docx":
        try:
            import docx  # python-docx
        except Exception as e:
            raise RuntimeError("python-docx is not installed (required for .docx).") from e

        doc = docx.Document(path)
        parts = [p.text for p in doc.paragraphs if p.text and p.text.strip()]
        text = "\n".join(parts)
        if not text.strip():
            warnings.append("DOCX had no readable paragraph text (tables/images not extracted).")
        return ExtractResult(text=text, page_count=None, warnings=warnings)

    if ext == ".pdf":
        # Two-extractor strategy. Neither pypdf nor pdfminer.six is universally
        # better at preserving paragraph structure: pypdf often produces
        # word-per-line wrap on standard Word/Google Docs PDFs (pdfminer reads
        # those cleanly), while pdfminer falls into word-per-line on PDFs with
        # unusual character-spacing encoding (pypdf reads those better, with
        # `normalize_paragraph_breaks` recovering paragraph boundaries from
        # the heuristic). Run both and pick the one whose paragraph-length
        # distribution looks healthier.
        try:
            text_pypdf, page_count, w_pypdf = _extract_pdf_pypdf(path)
        except Exception as e:
            raise RuntimeError(f"pypdf is not installed or failed: {e}") from e
        warnings.extend(w_pypdf)

        try:
            text_pdfminer, w_pdfminer = _extract_pdf_pdfminer(path)
        except Exception as e:
            text_pdfminer = ""
            warnings.append(f"pdfminer fallback unavailable: {e}")

        warnings.extend(w_pdfminer if text_pdfminer else [])

        # Score after normalization (pypdf needs it; pdfminer's blank-line
        # structure passes through and the line-wrap collapsing in step 3
        # cleans up internal soft wraps).
        norm_pypdf = normalize_paragraph_breaks(text_pypdf) if text_pypdf else ""
        norm_pdfminer = normalize_paragraph_breaks(text_pdfminer) if text_pdfminer else ""

        score_pypdf = _avg_paragraph_length(norm_pypdf) if norm_pypdf else 0.0
        score_pdfminer = _avg_paragraph_length(norm_pdfminer) if norm_pdfminer else 0.0

        if score_pdfminer > score_pypdf and norm_pdfminer:
            text = text_pdfminer
        else:
            text = text_pypdf

        if not text.strip():
            warnings.append("PDF text extraction returned empty. File may be scanned or image-based.")
        return ExtractResult(text=text, page_count=page_count, warnings=warnings)

    raise ValueError(f"Unsupported file type: {ext}. Please upload a PDF, DOCX, or TXT.")
