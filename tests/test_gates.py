import json
import struct

from szl_calibration.gates import ALLOW, BLOCK, validate_safetensors


def write_st(path, tensors, extra=b""):
    header = json.dumps(tensors).encode()
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(header)))
        fh.write(header)
        fh.write(extra)


def test_clean_fixture_allows(tmp_path):
    import numpy as np
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tobytes()
    hdr = {"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}
    p = tmp_path / "tiny.safetensors"
    write_st(p, hdr, extra=data)
    rep = validate_safetensors(str(p))
    assert rep.verdict == ALLOW
    assert rep.tensors == 1 and rep.parameters == 4
    assert rep.scan == "CLEAN"


def test_truncated_blocks(tmp_path):
    p = tmp_path / "trunc.safetensors"
    p.write_bytes(struct.pack("<Q", 500) + b"{}")
    assert validate_safetensors(str(p)).verdict == BLOCK


def test_bad_offsets_block(tmp_path):
    hdr = {"w": {"dtype": "F32", "shape": [4], "data_offsets": [0, 4096]}}
    p = tmp_path / "bad.safetensors"
    write_st(p, hdr, extra=b"\x00" * 16)
    assert validate_safetensors(str(p)).verdict == BLOCK


def test_negative_offsets_block(tmp_path):
    hdr = {"w": {"dtype": "I32", "shape": [1], "data_offsets": [-4, 0]}}
    p = tmp_path / "negative.safetensors"
    write_st(p, hdr, extra=b"\x00" * 4)
    rep = validate_safetensors(str(p))
    assert rep.verdict == BLOCK
    assert "data_offsets" in rep.reasons[0]


def test_non_object_header_blocks_without_raising(tmp_path):
    p = tmp_path / "list-header.safetensors"
    write_st(p, [{"dtype": "I32", "shape": [1], "data_offsets": [0, 4]}], extra=b"\x00" * 4)
    rep = validate_safetensors(str(p))
    assert rep.verdict == BLOCK
    assert rep.reasons == ["header root must be a JSON object"]


def test_missing_shape_blocks(tmp_path):
    hdr = {"w": {"dtype": "I32", "data_offsets": [0, 4]}}
    p = tmp_path / "missing-shape.safetensors"
    write_st(p, hdr, extra=b"\x00" * 4)
    rep = validate_safetensors(str(p))
    assert rep.verdict == BLOCK
    assert rep.reasons == ["w: shape must be a list"]


def test_gap_or_overlap_blocks(tmp_path):
    gap = {
        "a": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
        "b": {"dtype": "I32", "shape": [1], "data_offsets": [8, 12]},
    }
    overlap = {
        "a": {"dtype": "I32", "shape": [1], "data_offsets": [0, 4]},
        "b": {"dtype": "I32", "shape": [1], "data_offsets": [2, 6]},
    }
    for name, hdr, data in (
        ("gap", gap, b"\x00" * 12),
        ("overlap", overlap, b"\x00" * 6),
    ):
        p = tmp_path / f"{name}.safetensors"
        write_st(p, hdr, extra=data)
        rep = validate_safetensors(str(p))
        assert rep.verdict == BLOCK
        assert "expected contiguous offset" in rep.reasons[0]


def test_nan_blocks(tmp_path):
    import numpy as np
    data = np.array([1.0, float("nan")], dtype="<f4").tobytes()
    hdr = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    p = tmp_path / "nan.safetensors"
    write_st(p, hdr, extra=data)
    assert validate_safetensors(str(p)).verdict == BLOCK
