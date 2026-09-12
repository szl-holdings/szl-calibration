from szl_calibration.receipts import ReceiptChain
import pytest


def test_chain_verifies_and_links():
    c = ReceiptChain()
    r1 = c.append("a", {"x": 1})
    r2 = c.append("b", {"y": 2})
    assert r2.prev_hash == r1.hash
    assert c.verify()


def test_tamper_detected():
    c = ReceiptChain()
    c.append("a", {"x": 1})
    c.append("b", {"y": 2})
    victim = c._items[1]
    forged = type(victim)(victim.index, victim.timestamp_utc, victim.kind,
                          {"y": 999}, victim.prev_hash, victim.signature, victim.hash)
    c._items[1] = forged
    assert not c.verify()


def test_jsonl_roundtrip_shape():
    c = ReceiptChain()
    c.append("a", {"x": 1})
    line = c.to_jsonl().splitlines()[0]
    assert '"signature":"UNSIGNED_HONEST"' in line


def test_payload_and_returned_receipt_cannot_mutate_history():
    c = ReceiptChain()
    original = {"metrics": {"ece": 0.1}, "samples": [1, 2]}
    receipt = c.append("score", original)
    original["metrics"]["ece"] = 999
    original["samples"].append(3)
    receipt.payload["metrics"]["ece"] = 1000
    assert c.verify()
    assert c._items[0].payload == {"metrics": {"ece": 0.1}, "samples": [1, 2]}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_payload_never_appends(value):
    c = ReceiptChain()
    with pytest.raises(ValueError):
        c.append("invalid", {"value": value})
    assert len(c) == 0


def test_append_refuses_broken_chain():
    c = ReceiptChain()
    c.append("score", {"x": 1})
    c._items[0].payload["x"] = 2
    with pytest.raises(ValueError, match="receipt chain invalid"):
        c.append("score", {"x": 3})
    assert len(c) == 1


def test_concurrent_appends_preserve_order_and_integrity():
    from concurrent.futures import ThreadPoolExecutor
    c = ReceiptChain()
    with ThreadPoolExecutor(max_workers=8) as pool:
        receipts = list(pool.map(lambda i: c.append("score", {"i": i}), range(64)))
    assert sorted(r.index for r in receipts) == list(range(64))
    assert c.verify()


def test_nonfinite_tampering_is_invalid_instead_of_crashing():
    c = ReceiptChain()
    c.append("score", {"x": 1})
    c._items[0].payload["x"] = float("nan")
    assert c.verify() is False
