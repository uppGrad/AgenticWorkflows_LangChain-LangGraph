from __future__ import annotations

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
                texts.append(t)

        text = "\n".join(texts)
        if not text.strip():
            warnings.append("PDF text extraction returned empty. File may be scanned or image-based.")
        return ExtractResult(text=text, page_count=page_count, warnings=warnings)

    raise ValueError(f"Unsupported file type: {ext}. Please upload a PDF, DOCX, or TXT.")
