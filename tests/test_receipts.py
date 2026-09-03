from szl_calibration.receipts import ReceiptChain


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
