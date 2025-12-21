from __future__ import annotations

import argparse
import json

from uppgrad_agentic.workflows.document_feedback.graph import build_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to PDF/DOCX/TXT")
    ap.add_argument("--instructions", default="", help="Optional user instructions")
    args = ap.parse_args()

    graph = build_graph()
    out = graph.invoke(
        {
            "file": {"path": args.file, "name": args.file.split("/")[-1]},
            "user_instructions": args.instructions,
        }
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
