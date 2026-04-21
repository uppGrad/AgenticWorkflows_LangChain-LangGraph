from __future__ import annotations

import argparse
import json
import uuid

from uppgrad_agentic.workflows.auto_apply.graph import build_graph


def main():
    ap = argparse.ArgumentParser(
        description="Run the UppGrad auto-apply workflow against a single opportunity."
    )
    ap.add_argument(
        "--opportunity-type",
        required=True,
        choices=["job", "masters", "phd", "scholarship"],
        help="Type of opportunity",
    )
    ap.add_argument(
        "--opportunity-id",
        required=True,
        help="ID of the opportunity record in the database",
    )
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
            "opportunity_type": args.opportunity_type,
            "opportunity_id": args.opportunity_id,
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    print(json.dumps(out, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
