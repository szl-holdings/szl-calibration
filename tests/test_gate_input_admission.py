"""Adversarial file admission: malformed weights yield reports, never ALLOW."""
import io
import json
import struct

import pytest

from szl_calibration.gates import ALLOW, BLOCK, validate_safetensors


def framed(header):
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 1.0)


@pytest.mark.parametrize("dtype", [[], {}, None, 32, True])
def test_malformed_dtype_returns_block(tmp_path, dtype):
    path = tmp_path / "bad.safetensors"
    header = json.dumps({"w": {"dtype": dtype, "shape": [1], "data_offsets": [0, 4]}}).encode()
    path.write_bytes(framed(header))
    assert validate_safetensors(str(path)).verdict == BLOCK


@pytest.mark.parametrize("header", [
    b'{"w":{"dtype":"BAD","dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
    b'{"w":{},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
    b'{"__metadata__":{"a":"one","a":"two"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}',
])
def test_duplicate_header_keys_block(tmp_path, header):
    path = tmp_path / "duplicate.safetensors"
    path.write_bytes(framed(header))
    assert validate_safetensors(str(path)).verdict == BLOCK


@pytest.mark.parametrize("metadata", [None, [], "text", {"value": 1}, {"value": {}}])
def test_metadata_requires_string_map(tmp_path, metadata):
    header = json.dumps({"__metadata__": metadata, "w": {
        "dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}).encode()
    path = tmp_path / "metadata.safetensors"
    path.write_bytes(framed(header))
    assert validate_safetensors(str(path)).verdict == BLOCK


@pytest.mark.parametrize("constant", [b"NaN", b"Infinity", b"-Infinity"])
def test_non_json_numbers_block_even_in_unused_fields(tmp_path, constant):
    header = b'{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4],"extra":' + constant + b'}}'
    path = tmp_path / "constant.safetensors"
    path.write_bytes(framed(header))
    assert validate_safetensors(str(path)).verdict == BLOCK


def test_valid_metadata_and_exact_size_limit_allow(tmp_path):
    header = b'{"__metadata__":{"source":"synthetic"},"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}'
    raw = framed(header)
    path = tmp_path / "valid.safetensors"
    path.write_bytes(raw)
    assert validate_safetensors(str(path), max_bytes=len(raw)).verdict == ALLOW
    assert validate_safetensors(str(path), max_bytes=len(raw) - 1).verdict == BLOCK


def test_oversized_input_stops_after_one_detection_byte(monkeypatch):
    class ObservedFile(io.BytesIO):
        consumed = 0

        def read(self, size=-1):
            assert size >= 0, "input must never be read without a bound"
            data = super().read(size)
            self.consumed += len(data)
            return data

    source = ObservedFile(b"x" * 4096)
    monkeypatch.setattr("builtins.open", lambda *args, **kwargs: source)
    result = validate_safetensors("synthetic", max_bytes=64)
    assert result.verdict == BLOCK
    assert source.consumed == 65


@pytest.mark.parametrize("limit", [-1, True, 1.5, "64", None])
def test_invalid_limit_blocks_before_open(monkeypatch, limit):
    def forbidden_open(*args, **kwargs):
        raise AssertionError("invalid bounds must not open the input")

    monkeypatch.setattr("builtins.open", forbidden_open)
    assert validate_safetensors("synthetic", max_bytes=limit).verdict == BLOCK
