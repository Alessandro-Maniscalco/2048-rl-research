# Attribution and third-party code

Built by Alessandro Maniscalco for fun and research, with substantial Codex assistance in implementation, experiments and documentation.

The combined distribution is GPL-3.0. This does not replace the licenses or copyright notices of separately attributed files.

- **MacroXue / 2048-ai**, GPL-3.0: native search, formation lookup tables and derived adapters/patches. The record player builds on this work.
- **game-difficulty / 2048EndgameTablebase**, GPL-3.0: compact endgame search/tablebase code used by the hybrid.
- **moporgic / TDL2048**, MIT: `research/tdl2048_reference.cpp`; original notice in `research/TDL2048_LICENSE.md`.
- **tsangwpx / ml2048**, MIT: `research/pretrained_sources/ml2048/` source files and `rl2048/agents/ml2048_network.py`; licenses in that source directory and `rl2048/agents/ML2048_LICENSE.txt`. The pretrained model was external, not trained from scratch here.
- **wwk1397 / Code-for-Refining-Evaluation-Functions-for-Game-2048-by-Extended-Temporal-Difference-Learning**: optional research reference; see upstream terms. Its repository is not vendored here.

Pinned upstream revisions and local modifications are listed in `docs/vendor-sources.json` and `docs/vendor-patches/`. Nested upstream repositories, pretrained weight files and generated multi-gigabyte caches are excluded from this source release. Fetching upstream code does not grant additional rights beyond its license.
