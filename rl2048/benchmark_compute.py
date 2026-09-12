"""Measure this Mac on the actual n-tuple workload, including transfers.

Large batched GPU gather throughput is reported separately from single-board
latency. Online self-play has dependent decisions; the two are not interchangeable.
"""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import platform
import time

import numpy as np
from numba import njit
import torch

from rl2048.agents.ntuple import NTupleAgent, row_tables
from rl2048.fast2048 import train_batch, value


@njit(cache=True, nogil=True)
def values_batch(boards, weights, patterns):
    result = np.empty(len(boards), np.float32)
    for index in range(len(boards)):
        result[index] = value(boards[index], weights, patterns)
    return result


def benchmark(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    agent = NTupleAgent("4x6")
    rng = np.random.default_rng(12)
    agent.weights[:] = rng.uniform(-1, 1, size=agent.weights.shape).astype(np.float32)
    result = {"python": platform.python_version(), "platform": platform.platform(),
              "torch": torch.__version__, "mps_available": torch.backends.mps.is_available(),
              "cpu_vs_gpu": [], "thread_scaling": []}
    if torch.backends.mps.is_available():
        weights_gpu = torch.from_numpy(agent.weights.ravel()).to("mps")
        patterns_gpu = torch.from_numpy(agent.patterns.ravel()).to("mps")
        powers_gpu = (16 ** torch.arange(6, dtype=torch.int64)).to("mps")
        offsets_gpu = (torch.arange(4, dtype=torch.int64) * 16 ** 6).to("mps").view(1, 4, 1)

        def gpu_values(boards_gpu):
            encoded = boards_gpu[:, patterns_gpu].reshape(-1, 4, 8, 6)
            indices = (encoded * powers_gpu).sum(dim=3) + offsets_gpu
            return weights_gpu[indices].sum(dim=(1, 2))

        for batch in (1, 64, 1024, 8192):
            boards = rng.integers(0, 13, size=(batch, 16), dtype=np.uint8)
            boards[rng.random(boards.shape) < .35] = 0
            expected = values_batch(boards, agent.weights, agent.patterns)
            gpu_boards = torch.from_numpy(boards.astype(np.int64)).to("mps")
            actual = gpu_values(gpu_boards).cpu().numpy()
            np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-5)
            repeats = max(10, 5000 // batch)
            start = time.perf_counter()
            for _ in range(repeats):
                values_batch(boards, agent.weights, agent.patterns)
            cpu = (time.perf_counter() - start) / repeats
            torch.mps.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                gpu_values(gpu_boards)
            torch.mps.synchronize()
            gpu = (time.perf_counter() - start) / repeats
            # Sequential decision cost also waits for action values on the CPU.
            start = time.perf_counter()
            for _ in range(min(repeats, 50)):
                gpu_values(torch.from_numpy(boards.astype(np.int64)).to("mps")).cpu().numpy()
            transfer = (time.perf_counter() - start) / min(repeats, 50)
            result["cpu_vs_gpu"].append({"batch": batch, "cpu_seconds": cpu,
                                         "mps_resident_seconds": gpu, "mps_roundtrip_seconds": transfer})
        del weights_gpu, gpu_boards
        torch.mps.empty_cache()
    agent.weights.fill(0)
    rows, rewards = row_tables()
    dummy = np.zeros((1, 1), np.float32)
    train_batch(agent.weights, agent.patterns, rows, rewards, 1, 0, .1, dummy, dummy, False)
    for workers in (1, 2, 4, 8, 12, 18):
        agent.weights.fill(0)
        start = time.perf_counter()
        # Shared sparse weights, asynchronous Hogwild updates, as in the
        # parallel-training literature. No reproducibility claim for >1 worker.
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(train_batch, agent.weights, agent.patterns, rows, rewards,
                                   300, 900 + i, .1, dummy, dummy, False) for i in range(workers)]
            batches = [future.result() for future in futures]
        seconds = time.perf_counter() - start
        transitions = sum(int(batch[:, 1].sum()) for batch in batches)
        result["thread_scaling"].append({"workers": workers, "seconds": seconds,
                                         "transitions": transitions, "transitions_per_second": transitions / seconds})
        print(result["thread_scaling"][-1], flush=True)
    result["selected_workers"] = max(result["thread_scaling"], key=lambda r:r["transitions_per_second"])["workers"]
    (out / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    benchmark("runs/research/compute")
