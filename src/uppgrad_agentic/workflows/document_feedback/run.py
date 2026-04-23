from __future__ import annotations

import argparse
import json
import uuid

from uppgrad_agentic.workflows.document_feedback.graph import build_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Path to PDF/DOCX/TXT")
    ap.add_argument("--instructions", default="", help="Optional user instructions")
    ap.add_argument(
        "--thread-id",
        default=None,
        help="Checkpointer thread ID (auto-generated if not provided)",
    )
    args = ap.parse_args()

    thread_id = args.thread_id or str(uuid.uuid4())

    graph = build_graph()
    out = graph.invoke(
        {
            "file": {"path": args.file, "name": args.file.split("/")[-1]},
            "user_instructions": args.instructions,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
