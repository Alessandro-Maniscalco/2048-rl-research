"""Import/export TDL2048+ weight files, keeping external provenance explicit.

Serial format is documented by weight::save/load in the author's MIT-licensed
2048.cpp. We read only float32 value tables and skip optional TC accumulators.
The pattern names identify the exact ordering of the learned lookup tables.
"""

import argparse
import json
from pathlib import Path
import struct

import numpy as np

from rl2048.agents.ntuple import LAYOUTS, NTupleAgent


def read_exact(file, count):
    data = file.read(count)
    if len(data) != count:
        raise ValueError("Truncated TDL checkpoint.")
    return data


def import_weights(source, output, provenance):
    tables = {}
    with Path(source).open("rb") as file:
        code, count = struct.unpack("<BI", read_exact(file, 5))
        if code != 0 or count > 64:
            raise ValueError("Unsupported TDL container.")
        for _ in range(count):
            serial = read_exact(file, 1)[0]
            if serial != 4:
                raise ValueError(f"Unsupported table serial {serial}")
            name_raw = read_exact(file, 8)
            if name_raw[6:] == b"\x00\x00":
                number, width, _ = struct.unpack("<IHH", name_raw)
                name = f"{number:0{width or 6}x}"
            else:
                name = name_raw.decode("ascii").strip()
            dtype_bytes, length = struct.unpack("<HQ", read_exact(file, 10))
            if dtype_bytes != 4 or length != 16 ** 6:
                raise ValueError(f"Expected float32 6-tuples; found {name}: {dtype_bytes}, {length}")
            tables[name] = np.fromfile(file, dtype="<f4", count=length)
            if len(tables[name]) != length:
                raise ValueError("Truncated weight array.")
            while True:
                block_size = struct.unpack("<H", read_exact(file, 2))[0]
                if block_size == 0:
                    break
                extra_length = struct.unpack("<Q", read_exact(file, 8))[0]
                file.seek(block_size * extra_length, 1)
    layout = next((name for name in ("4x6", "8x6") if set(LAYOUTS[name]) == set(tables)), None)
    if layout is None:
        raise ValueError(f"Unsupported pattern set {list(tables)}")
    agent = NTupleAgent(layout)
    for index, pattern in enumerate(LAYOUTS[layout]):
        agent.weights[index] = tables[pattern]
    if not np.isfinite(agent.weights).all():
        raise ValueError("Non-finite imported weights.")
    agent.metadata = {"provenance": provenance, "source_file": str(Path(source).resolve()),
                      "source_patterns": list(tables), "training_history_known": False}
    agent.save(output)
    return agent


def export_weights(agent, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as file:
        file.write(struct.pack("<BI", 0, len(agent.patterns)))
        for pattern, weights in zip(LAYOUTS[agent.layout], agent.weights, strict=True):
            file.write(struct.pack("<BIHHHQ", 4, int(pattern, 16), len(pattern), 0, 4, len(weights)))
            weights.astype("<f4", copy=False).tofile(file)
            file.write(struct.pack("<H", 0))
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--provenance", required=True)
    args = parser.parse_args()
    agent = import_weights(args.source, args.output, args.provenance)
    print(json.dumps({"layout": agent.layout, "metadata": agent.metadata}, indent=2))
