"""Evaluate a local decision study; never sends data to a provider."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .decisions import assess_decisions, parse_study, MAX_STUDY_BYTES
from .receipts import ReceiptChain


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", type=Path)
    args = parser.parse_args(argv)
    try:
        with args.study.open("rb") as stream:
            raw = stream.read(MAX_STUDY_BYTES + 1)
        assessment = assess_decisions(parse_study(raw))
        receipt = ReceiptChain().append("decision.study.v1", assessment)
        print(json.dumps({"assessment": assessment, "receipt": asdict(receipt)}, indent=2, allow_nan=False))
        return 0 if assessment["verdict"] == "ELIGIBLE_FOR_SHADOW" else 2
    except (OSError, TypeError, ValueError):
        print(json.dumps({"status": "INVALID_STUDY", "execution_authorized": False}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
