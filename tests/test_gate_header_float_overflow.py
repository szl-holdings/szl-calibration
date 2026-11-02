"""Reject JSON numbers that overflow Python floats before tensor admission."""

import struct

import pytest

from szl_calibration.gates import ALLOW, BLOCK, validate_safetensors


def write_header_number(path, literal):
    # Unknown tensor fields still belong to the JSON header. A clean synthetic
    # scalar ensures that a malformed number cannot hide behind a clean scan.
    header = ('{"w":{"dtype":"F32","shape":[],"data_offsets":[0,4],'
              '"extra":' + literal + '}}').encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0))


@pytest.mark.parametrize("literal", ["1e999", "-1e999"])
def test_overflowing_header_number_blocks(tmp_path, literal):
    path = tmp_path / "overflow.safetensors"
    write_header_number(path, literal)
    report = validate_safetensors(str(path))
    assert report.verdict == BLOCK
    assert report.scan == "UNAVAILABLE"
    assert "finite" in report.reasons[0]


def test_finite_header_number_preserves_existing_behavior(tmp_path):
    path = tmp_path / "finite.safetensors"
    write_header_number(path, "1e308")
    report = validate_safetensors(str(path))
    assert report.verdict == ALLOW
    assert report.scan == "CLEAN"
