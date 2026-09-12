# Research and native acceleration

The learning algorithms in `rl2048` are independently written, readable Python.
The native acceleration experiment also uses Hung Guei's MIT-licensed TDL2048+
backend: https://github.com/moporgic/TDL2048 . Pinned revision:
`a99f620aec0d30a75943a4c9646743f1f53b0197`.

Reproduce the local compiler build:

```bash
git clone https://github.com/moporgic/TDL2048 research/TDL2048
git -C research/TDL2048 checkout a99f620aec0d30a75943a4c9646743f1f53b0197
git -C research/TDL2048 apply ../tdl2048_mac.patch
clang++ -std=c++20 -O3 -DNDEBUG -pthread -Wno-invalid-specialization -Wno-bitwise-conditional-parentheses -Wno-string-plus-int research/TDL2048/2048.cpp -o research/TDL2048/2048-mac
```

Xcode supplies Apple's compiler and SDK. The compiler is now working. This
project does not require opening Xcode or developing an iPhone application.
The patch resolves Clang integer overloads and a non-constant constructor.

`tdl2048_reference.cpp` is the original source retained for paper/code reading;
its MIT license is in `TDL2048_LICENSE.md`. The working clone, executable, weights,
and datasets stay local and are ignored by Git.

The native parallel TC run collapsed; it is **not** a successful trained model.
A serial pilot did not show that immediate collapse. This is consistent with a
concurrency problem in TC's accumulator ratio, but the exact cause is not yet
verified. Use the tested Numba TC implementation for current experiments.

The published `8x6patt.w` download is **externally pretrained**, not learned on
this Mac. Its URL and SHA256 are saved in `runs/research/published/source.json`.
Imported directory checkpoints retain explicit provenance. Locally trained
checkpoints originate from our own zero/optimistic initialized models.

Scripts here run concrete research experiments; they are not part of the
introductory lesson. All neural algorithms, including AWR/IQL/CQL/SAC, are in
`rl2048/agents`, with their visible loops in corresponding training modules.
