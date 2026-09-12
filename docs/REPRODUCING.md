# Reproducing the project

Use Python 3.13 and the committed `uv.lock`. Run all commands from the repository root. The original experiments ran on an M5 Pro MacBook Pro with 24 GB RAM. GPU/MPS is useful only where measured neural training throughput benefits; native search and game simulation use CPUs.

## Runnable without research caches

```bash
uv sync --locked
uv run pytest -q tests/test_game.py tests/test_q_learning.py tests/test_experiments.py
uv run python -m scripts.verify_record
uv run python -m rl2048.train --steps 2000 --out runs/my_smoke
uv run python -m rl2048.evaluate --checkpoint runs/my_smoke/checkpoint.npz --out runs/my_eval
uv run jupyter lab
```

The record verifier reconstructs all random spawns from seed 8982012, checks every recorded action is legal, and checks final score, board, tile and natural termination. It proves the trajectory obeys the game's rules; reproducing the **policy's decisions** additionally requires the frozen native engines and tables. The public action file includes the SHA-256 of the original full local replay.

To watch the included game:

```bash
uv run python -m http.server 8849 --bind 127.0.0.1
```

Open `http://127.0.0.1:8849/docs/record/replay.html`. The replay initially displays the record's final board; click Restart and Play to watch from the beginning.

## Optional native research dependencies

```bash
uv run python -m scripts.setup_sources
```

This clones the pinned revisions in `vendor-sources.json` and applies the committed patches. It leaves existing directories untouched. No native executable or table cache is silently downloaded. Native dependencies require a C++ toolchain; Apple's Command Line Tools provide `clang++` on macOS.

Build entry points are `research/endgame_build/CMakeLists.txt`, `research/build_frozen_tablebase.py`, `research/build_full_rank_search.py`, and `research/build_cached_full_rank.py`; inspect each before use. See `research/README.md` for the independent TDL2048 route. The best hybrid requires its generated formation caches under `runs/research/tablebase_search/cache` and the compiled adapters expected by `rl2048/agents/frozen_tablebase.py` and `rl2048/agents/endgame.py`. These multi-gigabyte artifacts are not bundled. The source release is **not a turnkey reconstruction of all historical trained models**.

The record campaign, on an already provisioned research checkout, is:

```bash
uv run python -m research.record_campaign \
  --out runs/research/endgame_tablebase/my_record_campaign \
  --games 1000 --workers 16 --first-seed 9040000 --hours 12
```

Use a new output directory and fresh seed range for a new campaign. Do not remove an existing STOP marker unless intentionally resuming work. `status.json`, `games.json`, `protocol.json` and `verified.json` retain outcomes, budgets and integrity checks; incomplete games are not complete-game scores. Most research runners are historical experiments, not a universal orchestration framework.

## Evidence boundaries

Scores in `MODELS.md` summarize local experiment artifacts. The public record has a compact, independently replayable action log. This source publication does not include every historical dataset, checkpoint, private supervision journal, or full benchmark log. Do not treat the summary table as an independently reproduced benchmark suite.

Held-out evaluations use fixed game seeds separate from training; selection sets can be reused for development and must be labeled accordingly. A record hunt selects an extreme outcome across many games and must not be compared to a mean. Some expensive studies ended with partial games; those incomplete batches do not supply a valid complete-batch mean.

Unit tests cover game rules, legal-action masks, TD targets, replay, distributions and native adapters. Native integration tests may skip when their local build is absent. Some research integration checks additionally need historical artifacts. The three core test files above are the minimal clean-checkout suite.
