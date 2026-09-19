"""Offline by default: python -m szl_calibration.jev_cli --request request.json."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

from .jev import JevError, MAX_REQUEST_BYTES, evaluate, parse_json, request_summary, validate_request


def _write_receipt(path: Path, receipt: dict) -> None:
    """Publish a complete local receipt atomically without overwriting any file."""
    temporary = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=".jev-receipt-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        # A same-directory hard link is atomic and fails if the requested name exists.
        os.link(temporary, path)
    except (OSError, ValueError):
        raise JevError("receipt_write_failed") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except OSError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a pinned Jev request; --live explicitly sends the provided input.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="Send one live request; may consume provider credit.")
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--receipt", type=Path, help="Atomically create a local receipt; an existing file is never overwritten.")
    args = parser.parse_args(argv)
    receipt = None
    try:
        with args.request.open("rb") as stream:
            raw = stream.read(MAX_REQUEST_BYTES + 1)
        if len(raw) > MAX_REQUEST_BYTES:
            raise JevError("request_too_large")
        request = validate_request(parse_json(raw))
        if args.live:
            receipt = evaluate(request, timeout=args.timeout)
        else:
            receipt = {"status": "REQUEST_VALIDATED_OFFLINE", "live_request_performed": False,
                       **request_summary(request), "calibration_status": "NOT_ASSESSED", "action_authorized": False}
        if args.receipt is not None:
            _write_receipt(args.receipt, receipt)
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
        return 0
    except (JevError, OSError) as error:
        code = error.code if isinstance(error, JevError) else "request_read_failed"
        # Preserve a successfully obtained response if only writing the receipt failed.
        safe = {"status": "ERROR", "error": code}
        if receipt is not None:
            safe["observation"] = receipt
        print(json.dumps(safe, sort_keys=True, allow_nan=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
