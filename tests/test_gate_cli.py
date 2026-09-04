from pathlib import Path

from szl_calibration.gate_cli import main


FIXTURES = Path(__file__).parent / "fixtures"


def test_expected_allow_is_success():
    assert main([str(FIXTURES / "tiny.safetensors"), "--expect", "ALLOW"]) == 0


def test_expected_block_is_success():
    assert main([str(FIXTURES / "truncated.safetensors"), "--expect", "BLOCK"]) == 0


def test_expectation_mismatch_is_distinct_failure():
    assert main([str(FIXTURES / "truncated.safetensors"), "--expect", "ALLOW"]) == 3


def test_block_without_expectation_remains_fail_closed():
    assert main([str(FIXTURES / "truncated.safetensors")]) == 1
