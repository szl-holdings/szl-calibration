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


def test_nan_blocks(tmp_path):
    import numpy as np
    data = np.array([1.0, float("nan")], dtype="<f4").tobytes()
    hdr = {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    p = tmp_path / "nan.safetensors"
    write_st(p, hdr, extra=data)
    assert validate_safetensors(str(p)).verdict == BLOCK
