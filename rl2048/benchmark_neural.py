"""Measure full forward/backward/Adam updates, including CPU-to-device input."""
import json
from pathlib import Path
import time
import numpy as np
import torch
from rl2048.agents.neural import BoardNet, tensor_boards


def main():
    torch.set_num_threads(4)
    results = []
    rng = np.random.default_rng(0)
    for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
        for batch in (256, 1024, 4096, 16384):
            torch.manual_seed(0)
            model = BoardNet(5).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
            states = rng.integers(0, 12, (batch, 16), dtype=np.uint8)
            def update():
                x = tensor_boards(states, device)
                prediction = model(x)
                loss = prediction.square().mean()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            for _ in range(5):
                update()
            if device == 'mps':
                torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(30):
                update()
            if device == 'mps':
                torch.mps.synchronize()
            seconds = (time.perf_counter() - start) / 30
            row = {'device': device, 'batch': batch, 'seconds_per_update': seconds,
                   'examples_per_second': batch / seconds, 'cpu_threads': 4}
            results.append(row)
            print(json.dumps(row), flush=True)
    path = Path('runs/research/compute/neural.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2))

if __name__ == '__main__':
    main()
