"""CI entrypoint: python -m szl_calibration.gate_cli <file.safetensors> [--expect VERDICT]."""
from __future__ import annotations

import argparse
import sys

from .gates import validate_safetensors

EXIT = {"ALLOW": 0, "REVIEW": 2, "BLOCK": 1}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--expect", choices=["ALLOW", "REVIEW", "BLOCK"], default=None)
    args = ap.parse_args(argv)
    rep = validate_safetensors(args.path)
    print(rep.to_json())
    code = EXIT[rep.verdict]
    if args.expect and args.expect != rep.verdict:
        print(f"GATE MISMATCH: expected {args.expect}, got {rep.verdict}", file=sys.stderr)
        return 3
    return code


if __name__ == "__main__":
    raise SystemExit(main())
