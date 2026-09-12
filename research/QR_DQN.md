# QR-DQN with a Transformer

Implementation: `rl2048/agents/qr_dqn.py`. Source: [Dabney et al., Distributional Reinforcement Learning with Quantile Regression](https://arxiv.org/abs/1710.10044).

## Inputs, process, outputs

The standard Transformer still reads sixteen cell values, projects each to128 features, adds learned positions, and uses two attention-plus-MLP blocks. Instead of four scalar output values, its final layer outputs **4 × 51 quantile locations**. Each location has equal probability mass1/51 in the approximation. There is no softmax over these return values and no fixed categorical support.

For quantile index i=0,...,50, the quantile level is tau_i=(i+0.5)/51. The acting interface returns the mean over those51 values for each action. Illegal actions are masked before argmax. A return distribution describes randomness in future rewards; it is not automatically a measure of uncertainty about the learned weights.

The width128 model has817,100 parameters, compared with407,300 for the scalar control. The Transformer torso is the same; the output head is larger. We initialize every action's quantiles to the corresponding scalar control prediction. With the same random seed, the torso and initial mean predictions match within numerical precision. This controlled initialization is our experiment choice. Different quantile-loss gradients separate the outputs during training.

## Learning equations

The game supplies fresh online experience to the existing one-million-record n-step replay buffer. No expert data or search is used. Sample512 records per update. For k actual steps (usually3):

`R = sum_j gamma**j * reward[t+j]`

`a_star = argmax_legal mean_i z_online(next_state, action, i)`

`target_j = R + gamma**k * (1-terminal) * z_target(next_state, a_star, j)`

For the action actually taken, form **all** predicted/target quantile pairs:

`delta_ij = target_j - z_online(state, action, i)`

`loss = mean_batch,i,j abs(tau_i - 1[delta_ij < 0]) * Huber_kappa(delta_ij) / kappa`

kappa=1. Targets and the sign indicator are detached. Our reduction averages over both quantile axes, so loss magnitude is normalized by the number of pairs. This is not the same numeric reduction as implementations that sum over predicted quantiles; learning-rate comparisons must account for it. The scalar Huber loss and quantile Huber loss should not be compared as if they measured the same error.

AdamW uses LR1e-4 and weight decay0.01; gradients are clipped to norm10. The target weights move0.005 toward the online weights each update. This is **QR-DQN with Double-Q selection and three-step replay**, not the paper's exact Atari configuration or full Rainbow. No prioritization, noisy layers or dueling head has been added.

## Files and execution

- `rl2048/agents/qr_dqn.py`: quantile Transformer, quantile targets, pairwise loss and optimizer update.
- `rl2048/agents/transformer_q.py`: unchanged shared attention/MLP encoder structure.
- `research/transformer_td_experiment.py`: selects the learner through `algorithm`, then collects games, trains, evaluates and checkpoints it.
- `rl2048/agents/neural.py`: saves architecture/quantile count and reloads policies for evaluation and replay.
- `tests/test_qr_dqn.py`: hand targets and gradients, masks, terminal handling, input modes, update behavior and complete checkpoint/replay path.

The existing worker reads the selected job from the study manifest:

```
.venv/bin/python -m research.scaled_transformer worker --job qr_compare_qr_dqn_score_seed0
```

The single queue starts that worker automatically. Do not execute it separately while the queue owns GPU training, or reuse an existing attempt directory.

## Controlled experiment

Four fresh runs each receive4,194,304 transitions, seed0, width128/depth2, batch512, same legal moves and same fixed evaluation suites:

| Algorithm | Reward |
|---|---|
| QR-DQN | points/128 |
| Double DQN | points/128 |
| QR-DQN | points/128 + corner/snake potential difference, strength2 |
| Double DQN | same strategic reward |

These isolate the algorithm within each reward setting and reward shaping within each algorithm. They do not match parameter counts. Initialization mean predictions are matched, but trajectories diverge as learning changes actions. Later comparisons need multiple training seeds and longer continuations; a4M budget ending is not evidence of a plateau.

The full51-quantile CPU smoke completed2,048 transitions,11 updates, checkpoint reload,100 complete evaluation games and an HTML replay. It is a plumbing check, not a policy-strength result. A separate MPS smoke is queued under the same GPU owner before the four full runs. Monitoring uses128 fixed games; the100-game suite is selection data. Reserved final-test seeds remain untouched.


## Whole-board MLP architecture control

`rl2048/agents/qr_mlp.py` adds a plain neural alternative to the Transformer. Sixteen tile exponents divided by 16 enter two fully connected ReLU layers. The screen uses width 800: 16 -> 800 -> 800 -> 204, then reshapes the outputs to [4 actions, 51 quantiles]. Acting still averages each action’s quantiles and takes the largest legal value. The same `QRDQN.update` computes the same n-step quantile targets and loss; replay and rewards are unchanged. `NeuralAgent.save/load` preserves this architecture as `mlp_qr`.

This MLP has 817,804 parameters, very close to the existing Transformer’s 817,100. We compare whole architectures at similar capacity, not attention alone: ReLU versus GELU, normalization and connectivity also differ. The first run keeps the 16 simple inputs. Learned embeddings are supported for a later controlled test; engineered relational inputs are rejected. The experiment is `qr_broad_mlp800_seed0`, with 4,194,304 transitions and the same ordinary-reward, three-step QR recipe. It is queued, not yet a completed result.


## Prioritized replay experiment

The optional replay sampler in `rl2048/prioritized_replay.py` keeps the same n-step records but revisits larger mean TD errors more often. For record i, let delta_i be the target mean minus the predicted mean. Then:

- p_i = abs(delta_i) + epsilon
- P(i) = p_i^alpha / sum_j p_j^alpha
- w_i = (N P(i))^(-beta) / max_j (N P(j))^(-beta)
- L = mean_i [w_i * quantile_Huber_loss_i]

The QR target and quantile loss stay the same; only sampling and per-record weighting change. Sum/min trees avoid rescanning a million records for each minibatch. New records receive the highest priority observed so far. Sampled priorities are refreshed after each update. These weights correct the replay sampling distribution; they do not correct the exploratory actions inside an n-step return. Replay is refilled on resume, as in the existing uniform-replay training.

The first full screen uses alpha0.6, beta0.4 increasing to1 over4,194,304 transitions, and epsilon0.001 in learning-reward units. The floor limits severe shrinkage of all normalized weights when some errors approach zero. Learning rate and the other QR settings are unchanged. The hypothesis is better use of rare informative transitions, motivated by repeated near-terminal value-ranking errors. Stochastic high-error records can also be unhelpful, so raw game score and compute cost decide the result. It is queued as `qr_broad_prioritized_seed0`; no strength result exists yet. [Original method and caveats](https://arxiv.org/html/1511.05952v4).
