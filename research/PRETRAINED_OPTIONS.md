# Pretrained neural 2048 models — 7 September 2026

We need weights trained on structured 2048 boards. Similar layer shapes alone do not make a useful transfer: tile encoding, action order, reward units and task semantics must match.

| Candidate | Architecture and weights | Fit to this project |
|---|---|---|
| [tsangwpx/ml2048](https://github.com/tsangwpx/ml2048) | MIT PyTorch CNN encoder, MLP actor and critic; actual 5.8 MB checkpoint downloaded and strictly loaded | Selected for a working fine-tuning pilot. Current board in; four policy logits and one state value out. These logits are not Q-values. |
| [SergioIommi/DQN-2048](https://github.com/SergioIommi/DQN-2048) | Three dense hidden layers, four Q-values; public 23.6 MB HDF5 checkpoint | Closest literal MLP/Q candidate, but uses 20 simulated successor boards rather than only the current board. Legacy Keras/TensorFlow implementation needs weight conversion and matching preprocessing. Explicit license not verified. Not run. |
| [HAN-oQo/RL_for_2048](https://github.com/HAN-oQo/RL_for_2048) | PyTorch DQN/SAC, README links Google Drive weights | Potential alternative; Drive checkpoint download not verified. Author reports roughly 6,000 mean score for DQN and 2,000 for SAC. Not run. |
| [Improving DNN-based 2048 Players](https://ieee-cog.org/2022/assets/papers/paper_222.pdf), [2025 thesis](https://kutarr.kochi-tech.ac.jp/record/2000372/files/1258010.pdf) | FC-CNN / afterstate value learning, including fine-tuning | Architecturally relevant to a value model plus planning. No public downloadable checkpoint verified in this search. Not run. |

## Selected model: inputs, processing, outputs

The 16 current cell values become ranks: empty=0, 2=1, 4=2, and so on. Each rank is one-hot encoded into 16 channels. The imported model has 1,451,077 parameters.

`board -> one-hot channels -> row / column / whole-board convolution branches -> 1,024 features`

The features feed two separate MLPs, each with hidden widths 256 and 64:

- Actor: four logits z(s,a). Mask illegal moves; training samples from softmax(z), evaluation chooses the largest legal logit.
- Critic: scalar V(s), estimating discounted future learning rewards. It trains the actor through advantage estimates; it is not used for search here.

Upstream actions are left/right/up/down. Our adapter explicitly reorders them to up/right/down/left. Upstream has 16 rank categories, so our adapter aliases ranks above 15 to rank 15; game rules remain unchanged. None of these evaluation games reached that representation limit.

## Actual fine-tuning experiment

Question: can a short PPO continuation adapt the downloaded policy to our raw merge-score reward?

Retain the pretrained encoder and actor. Reset only the critic's final output layer, because upstream uses different reward shaping. Train all parameters using Adam at 1e-5, PPO clipping 0.2, gamma 0.99, GAE lambda 0.95, entropy coefficient 0.001, minibatch 1,024, two epochs per rollout. Each rollout contains 256 games x 64 moves = 16,384 transitions. Learning reward is raw merge score / 128, with no added corner reward.

The Apple GPU (PyTorch MPS) completed 1,032,192 environment transitions in 90.2 seconds. Evaluation time is separate. This is one fine-tuning seed, not a hyperparameter sweep.

PPO uses delta_t = r_t + gamma * (1-terminal_t) * V(s_next) - V(s_t), GAE to accumulate these TD residuals, and a clipped policy-probability ratio to limit each update. Resetting the critic changes initial advantages, so a short continuation can degrade a strong actor even with a small learning rate. That is a possible explanation to test, not an established cause.

On the same 100 validation game seeds (8400000–8400099), the original scored 87,348.8 on average and the fine-tuned policy scored 78,515.0. The paired difference is -8,833.8, with an approximate 95% interval [-21,344.7, 3,677.1]. There is no demonstrated improvement. Preserve the original as the preferred checkpoint; retain the fine-tuned checkpoint as an experiment.

Both checkpoints were frozen before a separate 100-game test split (8500000–8500099); its results are saved alongside this report. All evaluations use complete games, legal argmax moves and no planning. Game-seed uncertainty is not training-seed uncertainty.

| Frozen checkpoint | Test mean score | Reached 2048 | Reached 8192 |
|---|---:|---:|---:|
| Original pretrained CNN | 81,715.08 | 94% | 33% |
| After 90-second PPO fine-tuning | 80,027.56 | 93% | 31% |

The held-out paired difference is -1,687.52 points, approximate 95% interval [-13,528.91, 10,153.87]. This short experiment establishes that the checkpoint can be adapted and trained locally; it does not establish a performance gain or a statistically clear degradation. All 200 test games ended naturally without truncation.

The upstream author reports at least five billion training transitions for the project. The precise training budget of this particular epoch-2500 checkpoint was not independently established. Its performance cannot be used as an equal-budget architecture comparison against our locally trained MLPs or n-tuples.

## Files and reproduction

- `rl2048/agents/ml2048_network.py`: MIT-attributed network definitions; accompanying `ML2048_LICENSE.txt` preserves the license.
- `rl2048/agents/pretrained2048.py`: checkpoint import, tile ranks, action permutation and critic recalibration.
- `rl2048/agents/neural.py`: shared action masking and save/load support.
- `rl2048/ppo_train.py`: actual PPO rollout and update loop, now accepting `--pretrained-checkpoint`.
- `research/evaluate_pretrained.py`: identical seeded games for frozen original and fine-tuned policies; paired score comparison.
- `research/pretrained_sources/ml2048/provenance.json`: pinned upstream commit and downloaded checkpoint SHA-256.
- `runs/research/pretrained_cnn/`: preserved checkpoints, training logs, validation/test records and replay.

```bash
uv run python -m rl2048.ppo_train \
  --out runs/research/pretrained_cnn/new_run \
  --pretrained-checkpoint research/pretrained_sources/ml2048/ml2048_20240330_013340-epoch-2500.pt \
  --device mps --seconds 90 --lr 0.00001 --epochs 2 \
  --batch 1024 --entropy 0.001 --width 1024 --reward-mode score
uv run python research/evaluate_pretrained.py --split validation
```

The evaluation script targets the preserved original experiment, not `new_run`; edit its checkpoint paths deliberately for another comparison. Do not tune against the held-out test seeds.

For a strict four-Q-output project, the next practical transfer is to keep this pretrained CNN encoder and train a new four-Q head with TD targets (or distill it into our MLP). The actor logits must not simply be relabeled as Q-values. Neither head transfer nor distillation was performed in this pilot.

The rising training-score curve is not evidence of improvement: all parallel games start together, so short, low-scoring games finish first and long games appear later. Use the fixed-seed complete-game evaluations to judge performance.
