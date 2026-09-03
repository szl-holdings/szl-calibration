"""Generate binary safetensors fixtures for the CI weight gate (binaries stay out of git)."""
import json
import pathlib
import struct

import numpy as np

FX = pathlib.Path(__file__).parent / "fixtures"


def main() -> None:
    FX.mkdir(exist_ok=True)
    hdr = json.dumps({"w": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]}}).encode()
    data = np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4").tobytes()
    (FX / "tiny.safetensors").write_bytes(struct.pack("<Q", len(hdr)) + hdr + data)
    (FX / "truncated.safetensors").write_bytes(struct.pack("<Q", 500) + b"{}")
    print(f"fixtures written to {FX}")


if __name__ == "__main__":
    main()
