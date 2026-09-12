# Learn representations and train from fresh games

The latest request removes the 80 manually calculated features from new experiments. Older checkpoints retain their original architecture so they remain reproducible.

## Execution path

1. `VectorGame` runs 128 independent games. It supplies 16 tile exponents per board, legal masks, raw merge rewards and final boards before reset.
2. `MLPQNetwork` uses those exponents directly or learns a tile embedding. `TransformerQNetwork` embeds/projects each cell, adds a learned position vector, then applies two standard Transformer encoder blocks. Each block has attention AND a feed-forward MLP, residual connections and layer normalization. Both architectures output four action values.
3. Epsilon-greedy action selection chooses a legal move. Epsilon falls from 1 to 0.1 over 100,000 transitions. No planner selects actions.
4. `learning_rewards` uses score / 128, or the explicitly separate log-score experiment. Neither gives corner or snake bonuses.
5. `NStepReplay` constructs discounted one-step or three-step records separately for each environment. A terminal state removes bootstrapping; a time limit keeps the final-board bootstrap. Neither crosses a reset.
6. `DQN.update` samples 512 records, selects the legal next action with the online network, evaluates it with the slower target network, minimizes Huber TD loss, and updates weights with Adam. Three-step targets use gamma cubed, or the actual shorter horizon at a boundary.
7. Newly learned weights select subsequent actions, collecting new data while replay retains earlier experience. This is ONLINE, OFF-POLICY learning. No teacher or fixed demonstration dataset is used.
8. The experiment saves configuration, weights, optimizer/target state, raw episode scores, loss/TD-error curves, and evaluations. It reloads each saved policy and plays 100 complete held-out games. Replay data are not saved in the checkpoint; a continuation must refill replay.

The return target is sum(j=0..k-1) gamma^j r[t+j] + gamma^k Q_target(s[t+k], argmax_legal Q_online(s[t+k], a)). k is n, shortened at episode or collection boundaries. Natural termination removes the last term. This ordinary n-step DQN implementation does not correct exploratory intermediate actions to the greedy target policy.

Simple input encoding: empty=0, 2=1, 4=2, etc., divided by16. Embedding input: 18 learned categorical vectors, empty through exponent17 (larger ranks share the last category). The network can learn relationships; equality, neighbors, differences, corners and snakes are not supplied.

The logarithmic reward is log2(1 + raw merge points) / log2(129). The +1 handles zero, and the divisor makes a 128-point reward equal 1 for both reward variants. This nonlinear transformation changes preferences among sequences of merges. Input encoding and reward transforms need not match.

## Commands (from the project root)

```sh
.venv/bin/python -m pytest -q
.venv/bin/python -m research.inspect_student_moves
.venv/bin/python -m research.transformer_td_experiment --out runs/research/transformer_td_new --steps 131072
.venv/bin/python -m research.report_transformer_td
.venv/bin/python -m rl2048.view --checkpoint runs/research/transformer_td/transformer_q_embedding_n3_score/seed0/last --out runs/transformer_replay.html --seed 8660000
```

`research/transformer_td_experiment.py` is a readable end-to-end loop. `--seeds`, `--limit`, and `--steps` permit a smaller smoke run; all step budgets must be divisible by256. Reports use the completed `runs/research/transformer_td` study by default. Each output directory is new, preventing accidental checkpoint overwrite.

## Interpretation

The first screen uses three training seeds, 131,072 transitions each, and ten variants. The architectures have different parameter counts, so this is an equal-data/update-budget screen, not a parameter-matched architecture proof. Batch size512 and LR0.0001 are fixed controls, not optimized values. Measure CPU versus MPS with the actual inference and training batch sizes before selecting devices. Do not compare these new models as if they had the hundreds of millions of transitions used by previous checkpoints. Extend promising variants under larger equal budgets before drawing conclusions about ultimate performance.
