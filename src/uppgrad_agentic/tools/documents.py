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


# pypdf and python-docx both emit a single "\n" between paragraphs in narrative
# documents (LaTeX article PDFs typeset paragraphs with first-line indent and
# no blank line, which pypdf renders as just \n). Every paragraph-aware
# analyzer downstream — analyze_rhetoric, analyze_narrative, finalize's
# kitchen-sink guard — splits on r"\n\s*\n" and therefore sees the whole body
# as a single paragraph when blank lines are missing. The synth then emits one
# whole-body rewrite because it was told there is only one paragraph to fix.
# This heuristic restores the blank-line separator: when a line ends with
# terminal punctuation and the next line begins with a capital letter, that
# single \n is treated as a paragraph boundary.
_PARAGRAPH_BOUNDARY_HEURISTIC = re.compile(r"([.!?])\n(?=[A-Z])")


def normalize_paragraph_breaks(text: str) -> str:
    """Convert single-newline paragraph boundaries into blank-line separators."""
    return _PARAGRAPH_BOUNDARY_HEURISTIC.sub(r"\1\n\n", text)


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
        # prefer pypdf (lightweight)
        try:
            from pypdf import PdfReader
        except Exception as e:
            raise RuntimeError("pypdf is not installed (required for .pdf).") from e

        reader = PdfReader(path)
        page_count = len(reader.pages)
        texts = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
                warnings.append(f"Failed to extract text from page {i+1}.")
            if t.strip():
                texts.append(_strip_page_footer(t))

        text = "\n".join(texts)
        if not text.strip():
            warnings.append("PDF text extraction returned empty. File may be scanned or image-based.")
        return ExtractResult(text=text, page_count=page_count, warnings=warnings)

    raise ValueError(f"Unsupported file type: {ext}. Please upload a PDF, DOCX, or TXT.")
