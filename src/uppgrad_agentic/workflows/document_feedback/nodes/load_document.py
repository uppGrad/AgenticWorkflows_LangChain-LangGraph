from __future__ import annotations

from uppgrad_agentic.tools.documents import extract_text_from_file
from uppgrad_agentic.workflows.document_feedback.state import DocFeedbackState


MIN_CHARS = 200


def load_document(state: DocFeedbackState) -> dict:
    file = state.get("file") or {}
    path = file.get("path")
    name = file.get("name") or (path.split("/")[-1] if path else "uploaded_file")
    mime = file.get("mime", "")

    if not path:
        return {
            "result": {
                "status": "error",
                "error_code": "FILE_MISSING",
                "user_message": "No file was provided. Please upload a PDF or DOCX.",
            }
        }

    try:
        res = extract_text_from_file(path)
    except Exception as e:
        return {
            "result": {
                "status": "error",
                "error_code": "FILE_UNREADABLE",
                "user_message": "We couldn't read your file. Please upload a valid PDF/DOCX/TXT.",
                "details": {"exception": str(e)},
            }
        }

    text = (res.text or "").strip()
    meta = {
        "file_name": name,
        "mime": mime,
        "char_count": len(text),
        "page_count": res.page_count,
        "extraction_warnings": res.warnings,
    }

    if len(text) < MIN_CHARS:
        return {
            "doc_meta": meta,
            "raw_text": text,
            "result": {
                "status": "error",
                "error_code": "EMPTY_OR_TOO_SHORT",
                "user_message": "The uploaded document seems empty or too short. Please upload a complete CV/SOP/cover letter.",
                "details": {"char_count": len(text), "warnings": res.warnings},
            },
        }

    return {"raw_text": text, "doc_meta": meta}
