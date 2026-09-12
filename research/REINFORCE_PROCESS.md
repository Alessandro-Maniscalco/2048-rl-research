# Plain REINFORCE in this project

The model chooses probabilities directly. It does not predict Q-values. This experiment tests whether actual full-game returns work better than the critic and bootstrapped targets in PPO for the current Transformer.

## Files and execution

1. `research/scaled_transformer.py` reads a job from the manifest and calls `research/reinforce_experiment.py:train_one`. The existing controller owns the single GPU worker.
2. `rl2048/agents/reinforce.py:PolicyTransformer` maps 16 tile categories to learned embeddings and positions, through four width-128 Transformer blocks with four attention heads and MLP sublayers, to four logits. Eight shared rotated/reflected views map actions back before averaging. The scalar critic head is absent.
3. `collect_complete_games` freezes this policy, samples only legal actions, and uses the compiled rules in `rl2048/vector_game.py` to finish 128 games. Games that finish early wait. Their first reset board is never played in the same batch.
4. `reward_to_go` works backward separately through each completed game. The game supplies the reward; no teacher label, prediction, or value network supplies it.
5. `policy_loss` differentiates negative log probability weighted by actual future rewards. All 128 games contribute before one Adam step. Boards are processed in chunks of 128 only to bound memory. Chunks do not cause separate optimizer steps.
6. `research/teacher_policy_experiment.py:evaluate` evaluates the frozen policy greedily on the same 128 monitoring games. Final and selected checkpoints each get 100 separate selection games. The reserved final test stays unused.
7. `rl2048/agents/neural.py` saves/reloads the actor. `rl2048/view.py` writes a replay. `research/report_reinforce.py` renders the score curves and protocol.

## Equations

Let i index a completed game, t a move in that game, and T_i its length. The network parameters are theta; pi_theta(a|s) is the legal-action softmax probability.

    r_(i,t) = actual merge points / 128
    G_(i,t) = sum from k=t to T_i-1 of r_(i,k)
    L(theta) = -(1/B) sum_i sum_t G_(i,t) log pi_theta(a_(i,t)|s_(i,t))

Here B=128 complete games and gamma=1. The gradient increases probabilities of actions in proportion to their observed remaining return. The expectation over fresh sampled games is what makes this a policy-gradient estimator. It does not identify one lucky move as causally responsible for every later point.

For a three-move episode with rewards [4, 0, 8], the unscaled returns are [12, 8, 8]. The second move gets credit for the later eight-point merge even though it earned zero immediately. With two equally probable legal actions and a chosen action whose return is 6, the loss gradient with respect to their logits is [-3, +3]; gradient descent raises the chosen action logit. Illegal logits receive zero gradient.

We use Adam with learning rate 0.0001 and a global gradient-norm cap of 0.5. There is no baseline, centering, standard deviation normalization, critic, replay, TD bootstrap, PPO clipping, or entropy bonus. The norm cap rescales a large gradient, and is different from PPO probability-ratio clipping.

## Data and evaluation

Each update discards the previous training batch and generates new complete games under the updated policy. A teacher-initialized arm copies the supervised actor once; its previous teacher pretraining is extra compute. A scratch arm starts fresh. Neither queries the teacher during REINFORCE.

A requested transition budget finishes its current 128-game batch, so exact counts can exceed the request. Count actual transitions and training time when comparing against PPO. A timeout or explicit stop discards the interrupted batch instead of assigning invented terminal returns.

The saved `first_training_game.json` follows every move of an actual first-batch game: exponent board, legal mask, sampled action and probability, merge points, and actual future return. `episodes.json` records all completed training games, `progress.json` every optimizer update, and `curve.json` fixed-game monitoring scores. Scores during sampling and greedy evaluation are different measurements. Loss values need not monotonically decrease, and neither loss nor a short score increase proves a better final policy.

The batch contains 128 complete games, often tens of thousands of boards. The GPU memory chunk contains 128 boards, expanded internally to 1,024 views. These two batch sizes have different purposes. The CPU simulates and the GPU predicts/updates in alternating phases; this implementation does not claim asynchronous collection or full hardware saturation.

## Published lab methods

- OpenAI InstructGPT used PPO, which uses a learned value baseline and clipped policy updates: https://openai.com/index/instruction-following/
- Cohere researchers studied RLOO: REINFORCE with a leave-one-out baseline formed from other responses to the same prompt: https://aclanthology.org/2024.acl-long.662/
- DeepSeek-R1 uses GRPO: compare rewards within groups of responses, omit the critic, and constrain updates with a clipped objective: https://arxiv.org/html/2501.12948v1#S2.SS2.SSS1
- REINFORCE++ adds normalization and stabilization to critic-free policy optimization: https://arxiv.org/abs/2501.03262

These are published examples, not a claim that all current labs use one algorithm. Language-model RL starts with a very strong pretrained model. Our much weaker 2048 actor, random tile spawns, and long games may make the variance and learning problem different.


## Separate baseline comparison after the plain runs

The plain control finished at 5,955.68 from the supervised actor (best selected 7,151.08), and 1,305.00 from scratch. The earlier one-update improvement did not establish a lasting gain.

The next named algorithm is `reinforce_loo`, not the original `reinforce`. For game i and move t, average actual remaining returns from the other B-1 completed games; use zero where an other game has already ended:

    b_(i,t) = sum_(j != i) G_(j,t) / (B-1)
    A_(i,t) = G_(i,t) - b_(i,t)
    L = -sum_i sum_t A_(i,t) log pi_theta(a_(i,t)|s_(i,t)) / B

No own-game return enters its own baseline. The baseline is a fixed number during differentiation. The expected score-function gradient before optimizer clipping remains unchanged because the baseline is independent of the current game's sampled action. Variance may improve, but the measured game results decide; lower centered-return standard deviation is not by itself proof of lower gradient variance.

This compares different independent games using a move-index baseline. It is inspired by leave-one-out baselines and is not a reproduction of language-model RLOO's same-prompt response groups. The model, initial weights, points/128 reward, full-game batch, memory chunks, optimizer and new-move budget remain fixed. There is still no learned critic, teacher query, normalization, replay, or clipped probability-ratio objective.


## Separate lower-temperature comparison

The 16-game batch screen made 198 updates instead of 26, but finished at 5,748 versus 6,842 and took 546 versus 400 training seconds. It is not promoted.

The next fixed-temperature experiment restores 128 complete games and uses T=0.25 instead of T=1:

    pi_T(a|s) = exp(z_a/T) / sum_legal_b exp(z_b/T)
    gradient log pi_T = (gradient z_a - sum_b pi_T(b|s) gradient z_b) / T

The SAME temperature is used when collecting actions and differentiating their log probabilities. Changing it only in collection would give the wrong on-policy gradient. Actual returns and the baseline are unchanged. Greedy argmax ignores positive temperature scaling, so a temperature change alone does not change the greedy actor.

The frozen original actor scored 3,028.68 with T=1 sampling and 5,772.00 with T=0.25, compared with 6,513.72 for greedy actions. This is a collection-policy control, not a learned improvement. The first T=0.25 training update gave mixed results; it is not promoted. The bounded full run tests whether better initial collection yields stronger learning, despite less exploration and more saturated action probabilities. New monitoring measures both the actual sampled training policy and greedy play, using 128 fixed game seeds and separate per-game action RNGs. Greedy score still chooses the saved best checkpoint, and both saved replays are greedy. Final sampled score is saved separately in sampled_evaluation.json.
