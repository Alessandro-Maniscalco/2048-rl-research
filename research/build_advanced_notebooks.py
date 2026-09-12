"""Create small, executable teaching notebooks around the real implementations."""
from pathlib import Path
import nbformat as n

ROOT=Path(__file__).resolve().parents[1]
def notebook(name,cells):
    book=n.v4.new_notebook(cells=[n.v4.new_markdown_cell(text.replace(chr(92)*2, chr(92))) if kind=='m' else n.v4.new_code_cell(text) for kind,text in cells])
    book.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    n.write(book,ROOT/'lessons'/name)

setup="""from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / 'rl2048').exists() else Path.cwd().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
"""
notebook('02_afterstates_and_search.ipynb',[
('m',r'''# Afterstates, shared patterns, and planning

The original Q-table memorized entire boards. An n-tuple model instead learns values of local tile patterns and sums them. Eight rotations/reflections share weights. This notebook does small calculations; saved long experiments are read from disk.

Execution path: `game.py` defines the rules → `fast2048.py` compiles the same rules and updates → `agents/ntuple.py` supplies patterns and weights → `ntuple_train.py` collects games → `evaluate.py` / `view.py` evaluate and replay.

```mermaid
flowchart LR
 S[Current board] --> M[Choose legal move]
 M --> R[Merge reward]
 M --> X[Afterstate before spawn]
 X --> N[Random tile]
 N --> S2[Next board]
 X --> V[N-tuple value]
```

Sources: [Guei dissertation](https://arxiv.org/abs/2212.11087), [optimistic TD](https://arxiv.org/abs/2111.11090), [Jaskowski](https://arxiv.org/abs/1604.05085).'''),
('c',setup+'''from rl2048.agents.ntuple import NTupleAgent, encode, row_tables
from rl2048.fast2048 import value, update_value
from rl2048.game import move, legal_actions
board = np.array([[2,2,4,0],[0,4,0,0],[0,0,0,0],[0,0,0,0]])
after, reward, _ = move(board, 3)
print('Before:\\n',board,'\\nAfter left, before spawn:\\n',after,'\\nCurrent reward:',reward)
'''),
('m',r'''## Follow one update

Action selection uses $r_t+V(x_t)$. But $V(x_t)$ begins **after** the current reward, so its TD error is

$$\delta_t=r_{t+1}+\gamma V(x_{t+1})-V(x_t).$$

Suppose $r_{t+1}=8$, the next afterstate has value 20, the current value is 0, and $\gamma=1$. The target is 28. With learning rate 0.1 and 32 active features, each feature occurrence receives $0.1\times28/32=0.0875$.

A feature can occur more than once because a board has repeated patterns. The implementation accumulates each occurrence; therefore total predicted-value change can exceed $0.1\delta$ when there are duplicates.'''),
('c', '''agent = NTupleAgent('4x4')  # tiny demonstration tables, same update kernel
state = encode(after)
before = value(state, agent.weights, agent.patterns)
delta = 8 + 20 - before
active_features = agent.patterns.shape[0] * agent.patterns.shape[1]
dummy = np.zeros((1,1),np.float32)
update_value(state, 28., agent.weights, agent.patterns, .1, dummy, dummy, False)
print({'old_value':before,'target':28,'TD_error':delta,'update_per_occurrence':.1*delta/active_features,
       'new_value':value(state,agent.weights,agent.patterns)})'''),
('m',r'''## Why planning helps

At a chance node, enumerate every empty square and both possible tiles. Each square gets $0.9/n$ probability for a 2 and $0.1/n$ for a 4. At a decision node, choose the legal action with the largest immediate reward plus continuation value. Search depth 1 uses the learned value immediately; depth 2 simulates one more random spawn and decision.

Discount 1 targets total game score. Randomness does not require a smaller or larger discount: it requires an expectation. Deeper search costs more computation and still depends on the leaf value estimates.'''),
('c', '''import json
for label, path in [('local model',ROOT/'runs/research/native_otd/validation.json'),
                    ('published reference',ROOT/'runs/research/published/checkpoint/validation.json')]:
    if path.exists():
        print(label)
        for depth, result in json.loads(path.read_text()).items():
            s=result['summary']
            print('depth',depth,'mean score',round(s['mean_score']), 'games',s['games'],'seconds',round(s['seconds'],2))'''),
('m', '''## What the short experiment taught us

The zero-initialized 4×6 model beat the larger optimistic candidates in our first short-budget screen. That is a budget-specific finding, not a contradiction of optimistic TD's long-run results. Temporal coherence improved greedy play but did not automatically improve shallow search.

Published pretrained results use weights learned elsewhere. Local checkpoints and download provenance remain separate. The graph below may change as experiments finish.'''),
('c', '''from IPython.display import Image, display, HTML
plot = ROOT/'runs/research/training_overview.png'
if plot.exists(): display(Image(filename=str(plot)))
display(HTML('<a href="http://127.0.0.1:8848/research/local_best_replay.html" target="_blank">Replay our locally trained agent</a>'))''')])
notebook('03_ppo_sac_and_offline_rl.ipynb',[
('m',r'''# PPO, SAC, and learning from a fixed dataset

All these neural agents see only the current board. Their policy outputs four logits, one for each move. Invalid actions are masked. `agents/neural.py` contains the shared network; each algorithm's math is in its own agent file.

```mermaid
flowchart LR
 G[vector_game.py] --> P[ppo_train.py fresh rollout]
 G --> R[offline_data.py replay buffer]
 R --> S[sac_train.py]
 D[Saved complete games] --> O[offline_train.py]
 O --> A[agents/awr.py]
 O --> I[agents/iql.py]
 O --> C[agents/cql.py]
 P --> E[evaluate.py raw score]
 S --> E
 A --> E
 I --> E
 C --> E
```

A rollout batch and a gradient minibatch are different. Default PPO collects 256 games × 64 moves = 16,384 transitions. Four epochs with minibatches of 4,096 make 16 gradient updates. Increasing to 16,384 gives four updates per rollout.'''),
('c',setup+'''import torch
from rl2048.agents.ppo import advantages, clipped_policy_loss
from rl2048.agents.sac import soft_value
from rl2048.agents.iql import expectile_loss
from rl2048.agents.cql import conservative_penalty
from rl2048.agents.neural import masked_log_probs
'''),
('m',r'''## PPO: reuse a rollout carefully

$\delta_t=r_t+\gamma(1-\mathrm{terminated}_t)V(s_{t+1})-V(s_t)$.

$A_t=\delta_t+\gamma\lambda(1-\mathrm{done}_t)A_{t+1}$.

The probability ratio is $\rho=\pi_{new}(a|s)/\pi_{old}(a|s)$. Maximize $\min(\rho A,\mathrm{clip}(\rho,0.8,1.2)A)$. The code returns the **negative** because the optimizer minimizes.

Timeouts bootstrap their final next state but cut the GAE recurrence before an auto-reset. Natural terminal states never bootstrap.'''),
('c', '''new_logp=torch.tensor([1.5,1.5]).log()
old_logp=torch.zeros(2)
advantage=torch.tensor([2.,-2.])
print('PPO loss:',clipped_policy_loss(new_logp,old_logp,advantage).item())
# First sample: min(3,2.4)=2.4. Second: min(-3,-2.4)=-3.
# Negative average = 0.3.
rewards=np.array([[1.,1.],[2.,2.]],np.float32)
values=np.array([[3.,3.],[4.,4.]],np.float32)
next_values=np.array([[4.,4.],[99.,5.]],np.float32)
term=np.array([[False,False],[True,False]])
trunc=np.array([[False,False],[False,True]])
print('GAE:',advantages(rewards,values,next_values,term,trunc,gamma=.5,lam=1)[0])'''),
('m',r'''## SAC: actor, critic, value, and buffer

Our explicit-value variant has two critics $Q_1,Q_2$, actor $\pi$, state value $V$, and slowly moving target $\bar V$.

- Critic target: $y=r+\gamma(1-\mathrm{terminated})\bar V(s')$.
- Value target: $\sum_a\pi(a|s)[\min(Q_1,Q_2)(s,a)-\alpha\log\pi(a|s)]$.
- Actor minimizes: $\sum_a\pi(a|s)[\alpha\log\pi(a|s)-\min(Q_1,Q_2)(s,a)]$.

The buffer stores the actual final next state, never the reset board. Modern discrete SAC can compute the soft value directly, but this version keeps the separate V you asked to examine.'''),
('c', '''mask=torch.tensor([[True,True,False,False]])
logs=masked_log_probs(torch.zeros(1,4),mask)
q=torch.tensor([[2.,4.,999.,999.]])
print('Soft V:',soft_value(logs,q,.1).item(), '= 3 + 0.1 log(2)')'''),
('m',r'''## Offline RL: no new game interactions during fitting

The fixed dataset has full games from our local expert and a random policy. All offline methods sample the same transitions.

**BC:** minimize $-\log\pi(a_D|s)$.

**AWR:** regress $V(s)$ to complete discounted return $G$. Set $A=G-V(s)$; minimize $-\min(\exp(A/\beta),20)\log\pi(a_D|s)$. This version uses Monte Carlo returns.

**IQL:** fit $V$ with expectile loss $|\tau-\mathbf{1}[u<0]|u^2$, where $u=\min(\bar Q_1,\bar Q_2)(s,a_D)-V(s)$. Fit critics toward $r+\gamma V(s')$. Extract the actor with weights $\min(\exp(\beta(Q-V)),100)$. The beta convention is inverse temperature here, unlike AWR's denominator.

**CQL:** add $\alpha_{CQL}[\log\sum_{a\;legal}\exp Q(s,a)-Q(s,a_D)]$ to the Bellman loss. This discourages assigning large values to unsupported actions. Our discrete version uses a Double-DQN target and a greedy critic policy.

An accurate imitation loss is not sufficient to guarantee long-game success: small action errors can lead to boards that the dataset rarely contains.'''),
('c', '''print('Expectile loss for errors [-2,+2], tau=.7:',expectile_loss(torch.tensor([-2.,2.]),.7).item())
q=torch.tensor([[0.,999.,0.,999.]])
masks=torch.tensor([[True,False,True,False]])
print('CQL penalty:',conservative_penalty(q,torch.tensor([0]),masks).item(),'= log(2)')
# AWR: return12, value10, temperature2 -> exp(1)=2.718 weight.
print('AWR example weight:',np.exp((12-10)/2))'''),
('m',r'''## What to graph

Primary measures are held-out raw score, maximum-tile distribution, tile-reaching rates, and game length. Put environment transitions on the x axis for online learning; offline learning has zero new environment transitions, so use optimizer updates or sampled training examples. Also record elapsed time.

Loss is a debugging signal. Different algorithms have different losses and reward scales; a lower number does not mean a stronger player.

Experiments change one factor: gamma .99/.999/1, minibatch 4,096/16,384, bottom-left/top-right potential shaping, constant survival reward, or symmetry augmentation. Final conclusions need multiple training seeds.'''),
('c', '''import json
for path in sorted((ROOT/'runs/research').glob('*/validation.json')):
    data=json.loads(path.read_text())
    if 'summary' in data:
        print(path.parent.name,round(data['summary']['mean_score']),data['summary']['episodes'],'games')'''),
('m', '''Sources: [PPO](https://arxiv.org/abs/1707.06347), [SAC with V](https://arxiv.org/abs/1801.01290), [discrete SAC](https://arxiv.org/abs/1910.07207), [AWR](https://arxiv.org/abs/1910.00177), [IQL](https://arxiv.org/abs/2110.06169), [CQL](https://arxiv.org/abs/2006.04779).

Read the implementation in this order: `agents/neural.py` → one algorithm file → its training loop → `evaluate.py`. Change one parameter, save to a new output directory, and compare scores on the same evaluation seeds.''')])
