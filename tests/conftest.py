"""Pytest configuration — ensures src/ is on sys.path.

The editable install via uv sometimes fails to activate its .pth file in
conda-base venvs (see Phase A debugging, 2026-04-26). Adding src/ here makes
test discovery deterministic regardless of the editable-install state.
"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
