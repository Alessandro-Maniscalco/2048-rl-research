> **Source correction, 2026-09-09:** The July 2025 2048/Horizon-DQN paper cited below was [withdrawn in v2](https://arxiv.org/abs/2507.05465). Its authors report incomplete and duplicated evaluation logs causing incorrect tables and figures. The historical v1 score claims below must not be used as verified evidence or an algorithm ranking. Our independently saved local experiments are unchanged.

# Q-learning options and restored strategic rewards

Reviewed 2026-09-08. There is no verified universally best Q-learning algorithm for our 2048 setup. A Transformer is the function approximator; Double DQN, QR-DQN, IQN and PQN are different learning recipes that can use a neural encoder.

## Implemented now: strategic reward controls

The two new width128 Transformer candidates resume the same 4M seed1 parent used by the completed unshaped continuation. They restore the existing corner-plus-snake potential at lambda2 and10; the network still sees only16 tile exponents. The training loop now reads the configured shaping strength explicitly.

`r = merge_points/128 + lambda * (gamma * Phi(next) - Phi(current))`

Terminal next potential is zero. Phi adds a bonus for a maximum tile at bottom-left to weighted exponent placement. The positional weights are:

```
 1  2  3  4
 8  7  6  5
 9 10 11 12
16 15 14 13
```

Larger ranks are favored near the start of the bottom-left snake. This is a soft heuristic, not a strict adjacent-order test, and upward moves remain legal. Potential differences telescope in the discounted episodic return; they provide learning guidance without a recurring positive bonus merely for staying in an arrangement. No clipping of shaped rewards is introduced. Approximate finite-data learning can still perform differently. On resume, value predictions initially reflect the old reward and must adapt.

Raw game points remain the evaluation metric. New reward candidates use the same batch512, n3, gamma0.99, LR1e-4, validation seeds and plateau rule as the unshaped continuation. No claims of improvement precede results.

## Research leads, not implemented algorithms

**Beyond The Rainbow (BTR), ICML2025.** A desktop-oriented Atari method combining IQN, Munchausen updates, normalization and vectorized training with Rainbow components such as prioritized replay, dueling and noisy exploration. It is strong evidence that plain DQN is not the endpoint. Its CNN and Atari performance do not transfer automatically to a Transformer, 2048 or Apple MPS.

- https://proceedings.mlr.press/v267/clark25a.html
- https://openreview.net/pdf?id=V3KXsUFw8D

**PQN, ICLR2025.** Parallelized Q-learning uses normalized networks and batches of fresh experience to avoid a large replay buffer and a separate target network. Its reported speedups concern its tested implementations and environments; they are not a measured Mac speedup. This is a worthwhile distinct algorithm comparison after the reward controls.

- https://proceedings.iclr.cc/paper_files/paper/2025/hash/c23f3852601f6dd7f0b39223d031806f-Abstract-Conference.html
- https://mttga.github.io/posts/pqn/

**Aftab, August2026 preprint.** Studies PQN encoder and value-head improvements on Atari and Procgen. Recent and relevant to efficient value learning, but no 2048 result establishes superiority here.

- https://arxiv.org/html/2608.07335v1

**2048 distributional study, July2025.** Compares DQN, PPO, QR-DQN and H-DQN. Its table reports average scores1,442.64,1,830.52,3,478.51 and5,693.67 at5,000 episodes; extended H-DQN averages6,536.43 and peaks at41,828. Maximum and average must not be confused. Several components and encoders differ, so this is not a clean distribution-only ablation. The text claims no shaping but also describes a monotonicity bonus; treat its protocol cautiously. Its results are not directly comparable with our seed suites and training budgets.

- https://arxiv.org/html/2507.05465v1

## QR-DQN implementation and next comparisons

QR-DQN is now implemented on the same Transformer torso, first isolating the distributional objective. It predicts N return quantiles for each of four actions; choose by their mean. Return variation represents stochastic outcomes, not automatically calibrated confidence about model error.

`Q(s,a) = mean_i z_i(s,a)`

`a_star = argmax_legal mean_i z_online_i(next,a)`

`y_j = n_step_reward + gamma**k * (1-terminal) * z_target_j(next,a_star)`

Fit all predicted/target quantile pairs with quantile Huber loss. The distribution can model varied future returns without fixing a categorical value range. Add prioritized replay, dueling or IQN only in controlled follow-ups; do not label a partial combination full Rainbow, BTR or H-DQN. QR-DQN has now been implemented and tested in a subsequent change; see QR_DQN.md. IQN, PQN, BTR and Aftab remain unimplemented research leads.
