# Models, equations and lessons

This is a concise research notebook, not a claim of a controlled comparison across every row. Different families used different budgets, architectures, data and search. Rounded scores in the README come from the experiment artifacts summarized here. Small screens and selection sets are explicitly distinguished from held-out evaluations. [Selected original summary JSONs](results/) preserve sample sizes and configurations; local logs/checkpoints are not all redistributed.

Notation: s is the board before a move, a a legal direction, r the merge points, s′ the board after the random spawn, γ the discount, θ model parameters, and θ⁻ a delayed target network. Every maximum is over legal actions. Terminal targets have no bootstrap. Evaluation always reports **untransformed game points**.

## 1. Random, tabular and feature Q-learning

The random baseline uniformly samples legal moves. Tabular Q-learning uses the complete board as a dictionary key; feature Q-learning shares weights across hand-designed board properties.

$$Q(s,a)\leftarrow Q(s,a)+\alpha\,[r+\gamma\max_b Q(s',b)-Q(s,a)]$$

$$Q_w(s,a)=w_a^\top\phi(s)$$

After 100,000 transitions, tabular learning had 98,121 distinct updated boards and only 1,879 repeated visits. Held-out means were **1,105.32** for Q-learning and **1,125.84** for random. The update was correct; state coverage was poor. Engineered features generalize but impose assumptions about good play.

## 2. MLP DQN, Double DQN and dueling networks

We tested 16 tile inputs → hidden layers → four Q-values, including raw values, exponents, relative encodings, different widths/depths, learning rates and training budgets. Double DQN separates choosing the next action from evaluating it; dueling separates state value from action advantage.

$$y=r+\gamma Q_{\theta^-}(s',\arg\max_b Q_\theta(s',b))$$

$$Q(s,a)=V(s)+A(s,a)-\tfrac14\sum_b A(s,b)$$

Exponent-input Double DQN reached **3,317.76** on 100 held-out games; a longer selected DQN reached **4,851.60**. Dueling repetitions varied and did not establish a universal improvement. More parameters did not fix scarce useful experience or inaccurate bootstrapped targets. Equal initial Q-values should be compared with controlled initialization, not just matching labels such as “DQN.”

## 3. N-tuple afterstate values

An afterstate x is the board **after sliding/merging but before the random tile appears**. The tile is not removed from the actual game. Small position patterns index learned tables, summed across rotations/reflections.

$$V(x)=\sum_{g\in D_4}\sum_p w_p[\operatorname{pattern}_p(gx)]$$

$$\delta_t=r_{t+1}+\gamma V(x_{t+1})-V(x_t)$$

The reward in this afterstate update belongs to the **next** move. To choose the current move, score its immediate reward plus its afterstate value. A selected locally trained player with planning averaged **353,812.36** over 100 held-out games. Lookup tables offer fast, direct credit assignment to reusable configurations. Temporal coherence, optimistic initialization and symmetry were explored. Unsafe parallel updates were not equivalent to stable serial training.

This is not proof that tables inherently beat neural networks. In an early, more closely matched 40,448-transition pilot, neural/table depth-one means were **5,056.76 / 3,562.08**; depth-two means over only 16 games were **10,224 / 10,362.25**. Maturity, parameter counts, data and planning confound broad comparisons.

## 4. Transformers, normalization and strategic rewards

The model sees 16 tile tokens, using numeric projections or learned tile embeddings plus positions. Self-attention mixes information across tiles; each transformer block also has an MLP. We tested removal of the 80 engineered features, capacity, symmetry, replay-buffer sizes, relative/log encodings and corner/snake rewards.

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(QK^\top/\sqrt d)V$$

$$r'=r+\beta[\gamma\Phi(s')-\Phi(s)]$$

The second equation is potential-based shaping, with terminal potential zero; it is the principled way to encourage a strategic configuration without simply paying forever for occupying it. An arbitrary snake bonus need not preserve the original objective. Rotating both the board and action labels supports any corner; a fixed bottom-left bonus must rotate consistently or it conflicts with that symmetry.

A capacity screen compared **0.81M parameters / 3,108.56 mean** against **3.19M / 3,220.36**, with an inconclusive score difference and 2.48× runtime. Bigger was not clearly better. A 4–4–2 pattern resembles 8–8–4, but their futures are not identical: newly spawned tiles are still 2 or 4. Relative normalization can hide important absolute scale. Log or state-relative rewards change what is optimized; log inputs do not require log rewards.

## 5. QR-DQN and 1-, 3-, 5-step TD

QR-DQN predicts return quantiles for each action; the policy chooses the legal action with the largest **mean** quantile value. For target quantile j and prediction i, δij = yj − zi; Hκ is the Huber loss.

$$L=\frac1{N^2}\sum_{i,j}|\tau_i-\mathbf1[\delta_{ij}<0]|\,H_\kappa(\delta_{ij})$$

$$y^{(n)}=\sum_{k=0}^{n-1}\gamma^k r_{t+k+1}+\gamma^n Q_{\theta^-}(s_{t+n},\arg\max_b Q_\theta(s_{t+n},b))$$

Three training seeds, each with 4,194,304 transitions, gave **7,584.56** for QR versus **3,924.81** for scalar Q. QR had roughly twice the parameters, so this was not parameter-matched. The same horizon study gave **5,419.04 / 7,584.56 / 6,647.36** for n = 1 / 3 / 5. Three steps helped this setting, but seed variation was large. Longer targets transmit rewards faster and also include more behavior-policy choices. Prioritized replay and quantile-risk variants were explored without establishing a universal additional gain.

## 6. PPO and REINFORCE

Policy methods output action probabilities. REINFORCE uses sampled returns G and a baseline b; PPO uses probability ratios ρ between new and old policies and an advantage estimate A.

$$L_{\rm REINFORCE}=-\mathbb E[(G-b)\log\pi_\theta(a\mid s)]$$

$$J_{\rm PPO}=\mathbb E[\min(\rho A,\operatorname{clip}(\rho,1-\epsilon,1+\epsilon)A)]$$

An early PPO MLP averaged **8,724.84** on 100 held-out games; its CNN variant averaged **8,607.32**. Later initialized-transformer REINFORCE finished at **5,955.68**, below its **6,513.72** starting policy; an intermediate **7,151.08** checkpoint did not persist. Scratch REINFORCE reached only **1,305** in that experiment. Full-game returns are noisy, and improving a policy is harder than taking more gradient steps.

## 7. Discrete SAC

This implementation uses a stochastic actor, twin Q critics, an explicit value network and a replay buffer. α controls entropy; Qmin is the smaller critic.

$$V(s)=\sum_a\pi(a\mid s)[Q_{\min}(s,a)-\alpha\log\pi(a\mid s)]$$

$$L_\pi=\mathbb E_{s,a\sim\pi}[\alpha\log\pi(a\mid s)-Q_{\min}(s,a)]$$

A short selection experiment averaged **3,670.44**. It did not establish an advantage over the simpler alternatives at our budgets. This is discrete SAC with a value network, not an assertion that continuous-action SAC can be applied unchanged.

## 8. Offline BC, AWR, IQL and CQL

An early offline dataset contained **1,228,462 transitions**, from 200 local n-tuple expert games and 1,000 random games. Offline learning holds that dataset fixed; online replay learning continuously adds new transitions. Uniform transition sampling overrepresents long, late-game trajectories.

**Behavior cloning (BC)** copies dataset actions:

$$L_{\rm BC}=-\mathbb E_D\log\pi(a\mid s)$$

**AWR** prefers actions with high estimated advantage A:

$$L_{\rm AWR}=-\mathbb E_D[\operatorname{clip}(e^{A/\beta},0,w_{\max})\log\pi(a\mid s)]$$

**IQL** fits an expectile value below good observed actions, then uses advantage-weighted policy fitting:

$$L_V=\mathbb E_D[|\tau-\mathbf1[u<0]|u^2],\qquad u=Q(s,a)-V(s)$$

**CQL** discourages optimistic values for actions unsupported by data:

$$L_{\rm CQL}=L_{\rm TD}+\alpha\,\mathbb E_D[\log\sum_a e^{Q(s,a)}-Q(s,a_D)]$$

Held-out means: BC **8,974.28**, AWR **5,923.96**, IQL **3,277.32**. CQL's **3,124.88** was a separate short selection run. More complicated offline objectives did not outperform cloning in these experiments. The dataset, return estimation and policy's own visited states matter as much as the objective.

## 9. Pretrained CNN, value distillation and DAgger

We adapted an explicitly credited pretrained 2048 CNN and separately trained students on teacher values/actions. DAgger adds teacher labels on states the student itself visits. A KL term anchors the student to an already useful policy.

$$L=\lambda_V(V_\theta-V_T)^2-\lambda_a\log\pi_\theta(a_T\mid s)+\lambda_{\rm KL}D_{\rm KL}(\pi_0\Vert\pi_\theta)$$

The original CNN averaged **81,622.96** on 100 selection games. An aggressive fine-tuning diagnostic reduced teacher loss while the 128-game monitor collapsed from **85,014 to 1,354**. Lower learning rate plus KL anchoring preserved performance; its **90,371.92** selection mean was not an established improvement over the original because the paired interval included zero. Copying teacher values is supervised distillation, not behavior cloning unless action imitation is also used. Good regression loss does not guarantee correct action rankings on the student's states.

## 10. Planning, endgame tables and record hunting

Search enumerates legal moves and random spawns, then evaluates leaves. For a reward-value search:

$$Q_d(s,a)=r(s,a)+\gamma\sum_z P(z\mid x)V_{d-1}(x+z)$$

Endgame tables instead solve a formation-specific goal probability:

$$P_{\rm goal}(s)=\max_a\sum_z P(z\mid x)P_{\rm goal}(x+z)$$

Goal states have probability one and failures zero. These probabilities are not raw-score Q-values. Our strongest hybrid uses native heuristic expectimax as fallback; its native heuristic is not a neural value with raw rewards added to every backup.

For the **same frozen neural Q model**, increasing decision-time depth from two to three raised the 100-game confirmation mean from **41,031.36 to 56,191.04**. This is not three-step TD. A depth-four eight-game screen was promising but too small to establish superiority.

An earlier timed hybrid averaged **636,750.44** over 100 fresh games. A separate 16-pair fixed-work table ablation averaged **715,240.25 with tables / 613,404.50 without**, but the score-difference interval included zero; tables were 2.68× faster. A four-million-work screen cost 3.25× as much without establishing a gain, so the record attempts used the economical one-million-work player. Its verified best game is **1,357,916**. Cached full-rank search accelerated queries without proving a better full-game policy. Longer rollout choices sometimes reversed under fresh samples: Monte Carlo noise is substantial.

The CPU-heavy record experiments tested independent games without training updates. A record can improve while mean policy strength remains unchanged.

## References and attribution

- [2048 reinforcement-learning survey](https://arxiv.org/abs/2212.11087).
- [Distributional RL with quantile regression](https://arxiv.org/abs/1710.10044).
- [PPO](https://arxiv.org/abs/1707.06347), [SAC](https://arxiv.org/abs/1801.01290), [AWR](https://arxiv.org/abs/1910.00177), [IQL](https://arxiv.org/abs/2110.06169), [CQL](https://arxiv.org/abs/2006.04779).
- [MacroXue search and lookup tables](https://github.com/macroxue/2048-ai), [game-difficulty endgame tables](https://github.com/game-difficulty/2048EndgameTablebase), [moporgic TDL2048](https://github.com/moporgic/TDL2048), [tsangwpx pretrained CNN](https://github.com/tsangwpx/ml2048).
