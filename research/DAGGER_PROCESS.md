# Learning from teacher corrections on student boards

The completed plain REINFORCE experiment learned a better sampled policy, but
its final greedy mean was 5,956 versus 6,514 before RL. Baseline, smaller game
batches and lower temperature did not produce a large greedy improvement.
Replays contain confident choices with immediate-death risk even when a safer
action earns the same immediate points and is preferred by the teacher.
This motivates correcting the student on the states its own choices produce.

This is a separate imitation-learning experiment, based on
[DAgger, Ross, Gordon and Bagnell (2011), Algorithm 3.1](https://proceedings.mlr.press/v15/ross11a.html).
Its theoretical guarantees need assumptions that we do not establish for our
finite Transformer optimization. It is an empirical test, not a promise.

1. Load the original supervised actor. Copy its encoder and four policy rows;
   discard the scalar value head. Both comparison arms receive identical weights.
2. Start with the original fitting partition only. Entire validation games remain
   excluded. The four-logit model has no engineered strategy features.
3. For the adaptive arm, freeze the greedy student and play a batch of complete
   games. It selects every action itself (`beta=0`). CPU simulation alternates
   with GPU inference. Finished games wait; no incomplete game is assigned a return.
4. Label every visited current board with the unchanged published teacher:

   `Q_T(s,a) = (r(s,a) + V_T(f(s,a))) / 128`.

   `f(s,a)` is the deterministic board after sliding/merging and before the new
   random tile. `r` is the actual merge score. `V_T` sums all 64 table entries.
   There is no future search, reward shaping, value floor or tile downgrading.
5. Retain the complete new labelled block alongside the original and all previous
   blocks. Uniformly sample boards from the cumulative dataset. The fixed-data
   arm instead continues sampling only the original fitting partition.
6. Let `A*(s)` contain all legal actions tied for best teacher value. Minimize

   `L(theta) = -mean_s log(sum_{a in A*(s)} pi_theta(a | s))`.

   With a unique best action this is ordinary cross entropy. With a tie, reward
   probability on either equivalent move. Adam changes the student weights;
   the teacher and labels stay fixed. Both arms use the same update budget,
   minibatch, learning rate and gradient limit. No TD bootstrap, Monte Carlo
   return weighting or scalar value regression is used.
7. Repeat collection and fitting. Measure complete greedy games on the same 128
   monitoring seeds; also track agreement on 2,048 held-out teacher-labelled
   boards. Save the best monitor checkpoint, the last checkpoint, 100-game
   endpoint evaluations and both game replays. Reserved final test seeds stay unused.

The original dataset is shared by both arms, so adding student-visited boards is
the changing factor. Hard targets and removing the auxiliary value objective
are changes relative to the older supervised actor, shared by both new arms.
DAgger costs extra CPU collection, GPU inference and teacher queries; comparisons
report both optimizer updates and actual time. The dataset grows without a cap
within the bounded experiment; this is not a latest-policy replay-buffer test.

Files:

- `rl2048/agents/reinforce.py`: reusable four-output policy Transformer.
- `rl2048/agents/imitation.py`: teacher-best action set and differentiable loss.
- `research/reinforce_experiment.py`: complete-game collection, with a separate
  greedy option for imitation; normal REINFORCE still samples actions.
- `research/dagger_experiment.py`: teacher labels, aggregation, the full training
  loop, validation, provenance, checkpoints and replay creation.
- `research/report_dagger.py`: live comparison graph and evidence links.
- `tests/test_dagger.py`: hand loss/gradient checks, complete table/gain target,
  frozen greedy collection, fitting partition isolation, cumulative data,
  complete training/evaluation/reload/replay for both arms.

Each saved `round_*.npz` records boards, legal actions, teacher Q, accepted actions,
played student actions, and episode lengths/scores. `dataset_blocks.json` records
hashes, seeds, query counts and example corrections. `training.pt` saves weights,
Adam state and RNG state. An explicit `resume_dagger` continuation restores a
complete collection/fitting round, verifies the actor and optimizer saves agree,
checks the unchanged recipe, teacher/data hashes and every accumulated block,
and restores all boards and random state. Old blocks are copied into the new
run; fresh collection continues at the next round seed. Both new and lifetime
updates, moves and times are recorded. A CPU test verifies that split/resumed
training exactly matches uninterrupted weights, Adam state and collected data.
Partial fitting rounds are deliberately rejected by this continuation path.


Optional cost-sensitive branch: retain the teacher-action loss and add
`lambda * mean_s sum_a pi(a|s) * (max_b Q_T(s,b)-Q_T(s,a)) * 128 / 16384`.
Illegal actions and accepted numerical ties have zero cost. The new test uses
lambda=1 versus a matched lambda=0 control. Teacher values are detached; this
remains imitation. Changing the objective during continuation requires explicit
`transfer_dagger_objective=True`, preserving actor, Adam, RNG and all fitting
blocks but recording that the objective changed. See
`COST_SENSITIVE_IMITATION_NOTES.md` for rationale and verification.
