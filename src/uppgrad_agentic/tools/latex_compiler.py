"""Compile LaTeX source to PDF using the tectonic engine."""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def compile_latex(source: str, *, timeout: int = 60) -> Optional[bytes]:
    """Compile *source* (a complete LaTeX document) to PDF.

    Returns the PDF bytes on success, or ``None`` on failure.
    Requires the ``tectonic`` binary to be on ``$PATH``.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="uppgrad_latex_") as tmpdir:
            tex_path = Path(tmpdir) / "document.tex"
            tex_path.write_text(source, encoding="utf-8")

            result = subprocess.run(
                [
                    "tectonic",
                    "-X", "compile",
                    "--untrusted",          # sandboxed mode
                    "--keep-logs",
                    str(tex_path),
                ],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )

            pdf_path = Path(tmpdir) / "document.pdf"

            if result.returncode != 0 or not pdf_path.exists():
                logger.warning(
                    "tectonic compilation failed (rc=%d):\nSTDOUT:\n%s\nSTDERR:\n%s",
                    result.returncode,
                    result.stdout[-2000:] if result.stdout else "",
                    result.stderr[-2000:] if result.stderr else "",
                )
                return None

            return pdf_path.read_bytes()

    except FileNotFoundError:
        logger.warning("tectonic binary not found on $PATH — skipping PDF compilation")
        return None
    except subprocess.TimeoutExpired:
        logger.warning("tectonic compilation timed out after %ds", timeout)
        return None
    except Exception:
        logger.exception("Unexpected error during LaTeX compilation")
        return None


def is_tectonic_available() -> bool:
    """Quick check whether tectonic is installed."""
    try:
        r = subprocess.run(
            ["tectonic", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False
