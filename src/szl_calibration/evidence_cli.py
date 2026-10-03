"""Plan one evidence collection step locally; never executes it or calls a provider."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .evidence_planner import MAX_PLAN_BYTES, parse_plan, plan_evidence
from .receipts import ReceiptChain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.plan.open("rb") as stream:
            raw = stream.read(MAX_PLAN_BYTES + 1)
        plan = plan_evidence(parse_plan(raw))
        receipt = ReceiptChain().append("evidence.plan.v1", plan)
        print(json.dumps({"plan": plan, "receipt": asdict(receipt)}, indent=2, allow_nan=False))
        # Zero means the planning calculation completed, never permission to act.
        return 0
    except (OSError, TypeError, ValueError):
        print(json.dumps({"evidence_class": "BLOCKED", "error": "invalid_evidence_plan",
                          "execution_authorized": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
