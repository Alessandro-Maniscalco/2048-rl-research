from pathlib import Path
import nbformat as n
from nbclient import NotebookClient
root=Path(__file__).resolve().parents[1]
cells=[]
def md(text):cells.append(n.v4.new_markdown_cell(text))
def code(text):cells.append(n.v4.new_code_cell(text))
md('''# Sixteen inputs, four Q-values, one move

This is the architecture you proposed. The input is the current 4×4 board flattened into 16 numbers. The outputs estimate the future return for up, right, down, and left. Evaluation chooses the largest **legal** Q-value.

`agents/dqn.py` defines the network and TD loss. `dqn_train.py` collects games, samples replay minibatches, and calls the update. `rewards.py` defines the separate reward experiments. `evaluate.py` always reports raw game score.''')
code('''from pathlib import Path
import sys
ROOT=Path.cwd() if (Path.cwd()/'rl2048').exists() else Path.cwd().parent
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from rl2048.agents.dqn import QNetwork, bellman_target
from rl2048.agents.ntuple import encode
from rl2048.agents.neural import NeuralAgent
from rl2048.game import Game2048,ACTION_NAMES
from IPython.display import SVG,display

display(SVG('''+repr('''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 850 100"><defs><marker id="a" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto"><path d="M0 0L6 3L0 6" fill="#52796f"/></marker></defs><g fill="#e4efec" stroke="#52796f"><rect x="5" y="20" width="170" height="60" rx="8"/><rect x="235" y="20" width="215" height="60" rx="8"/><rect x="510" y="20" width="160" height="60" rx="8"/><rect x="725" y="20" width="120" height="60" rx="8"/></g><g stroke="#52796f" marker-end="url(#a)"><path d="M175 50H230"/><path d="M450 50H505"/><path d="M670 50H720"/></g><g font-family="sans-serif" font-size="16" text-anchor="middle" fill="#263f36"><text x="90" y="55">16 cell values</text><text x="342" y="55">256 → 256 hidden units</text><text x="590" y="55">4 Q-values</text><text x="785" y="55">legal argmax</text></g></svg>''')+'''))
print(QNetwork())''')
md('''## Change the meaning of each input

The first layer still has exactly 16 inputs in all three cases.

- **Exponents:** empty → 0; tile 2 → 1/16; 4 → 2/16; 8 → 3/16. This compresses the dynamic range.
- **Tile values:** empty → 0; tile 2 → 2/65,536; 4 → 4/65,536. The fixed scaling keeps large tiles manageable, but early tiles are tiny numbers.
- **Relative:** divide each tile by the board's largest tile. This loses absolute scale: differently sized tiles can produce the same input pattern, although spawned tiles are always 2 or 4.

Changing input encoding changes what distinctions the network can learn easily. It does not change the game rules.''')
code('''board=np.array([[0,2,4,8],[16,32,64,128],[0,0,256,512],[1024,0,0,2048]])
stored=torch.tensor(encode(board)[None])
for mode in ('exponents','raw','relative'):
    network=QNetwork(input_encoding=mode)
    print(mode, network.encode_inputs(stored).numpy().round(6), '\\n')''')
md('''## Trace a real current-state decision and TD target

The current Q network selects the next action. The target network evaluates that action. That is Double DQN. Ordinary DQN instead takes the maximum from the target network itself.

The target is `learning_reward + gamma × target_Q(next_state, chosen_next_action)` for a nonterminal transition. At natural game over, it is just the learning reward. Illegal next actions never enter the maximization.

The cell below reads a saved policy if present and calculates one target without training or changing it. For this isolated trace, the target network is a copy of the current network; during training it moves slowly toward the current weights.''')
code('''path=ROOT/'runs/research/dqn16_exponents/last'
network=NeuralAgent.load(path).policy if (path/'metadata.json').exists() else QNetwork()
env=Game2048();board,info=env.reset(seed=123)
state=torch.tensor(encode(board)[None])
with torch.no_grad(): q=network(state)[0]
action=int(q.masked_fill(~torch.tensor(info['action_mask']),-1e9).argmax())
next_board,reward,terminated,truncated,next_info=env.step(action)
next_state=torch.tensor(encode(next_board)[None])
with torch.no_grad():
    next_q=network(next_state)
    target=bellman_target(torch.tensor([reward/128]),torch.tensor([terminated]),next_q,next_q,
                          torch.tensor(next_info['action_mask'][None]),.99,True)
print('Board:\\n',board)
print('Q-values:',dict(zip(ACTION_NAMES,q.tolist())))
print('Chosen move:',ACTION_NAMES[action], 'raw reward:',reward)
print('TD target:',target.item(),'old Q:',q[action].item(),'TD error:',target.item()-q[action].item())''')
md('''## Reward experiments

`score` is merge reward/128. `corner` rewards progress toward keeping the largest tile at bottom-left; `snake` rewards progress toward a descending positional snake; `corner_snake` combines them.

These three use the potential difference `scale × (gamma × Phi(next) − Phi(current))`, with terminal potential zero. This avoids paying the agent forever merely for leaving a tile in the corner. `dense_corner` deliberately pays a corner bonus each step as a separate comparison; that changes the objective.

The experiment first compares encodings with score reward. It then holds the best encoding fixed while varying reward. Each first-pass run uses the same transition budget and validation seeds. We repeat the selected combination with additional training seeds.''')
code('''from rl2048.rewards import learning_rewards
s=encode(board)[None]; ns=encode(next_board)[None]
for mode in ('score','corner','snake','corner_snake','dense_corner'):
    value=learning_rewards(np.array([reward]),s,ns,np.array([terminated]),.99,mode,2.)
    print(mode, float(value[0]))''')
code('''import json
path=ROOT/'runs/research/dqn16_comparison.json'
if path.exists():
    for result in json.loads(path.read_text()):
        print(result['name'], 'mean raw score:',round(result['mean_score']),
              'encoding:',result['encoding'],'reward:',result['reward'])
else:
    print('Experiments are still running; rerun this cell later.')''')
md('''## Read the learning curve correctly

A low TD loss can coexist with a weak policy. Look at frozen-policy raw score versus training transitions, and maximum-tile reaching rates. Exploration makes training-game scores different from evaluation scores.

The default minibatch has 4,096 sampled transitions. One update changes network weights using all of them; it is not a game with 4,096 steps. The replay buffer holds up to one million old transitions. Epsilon falls from 1 to .05 over the first million environment transitions.

Source equations: [DQN](https://arxiv.org/abs/1312.5602), [Double DQN](https://arxiv.org/abs/1509.06461).''')
md("""## Inspect the network while playing

The panel shows the 16 inputs and four Q-values for the current board. Click a direction yourself or **Agent move** to take the legal argmax. The model stays frozen; every new random tile produces a new input and a new set of Q-values.""")
code("""from rl2048.view import q_inspector
checkpoint=ROOT/'runs/research/dqn16_corner_snake/last'
if checkpoint.exists():
    display(q_inspector(checkpoint,seed=0))""")
md('''## Keep the network fixed; change how we choose a move

Direct action selection reads **one current board → four Q-values → legal argmax**.
Planning adds the known game rules before choosing. For each legal move, we merge
the board, enumerate every possible 2/4 spawn, and apply the same Q-network to
those next boards. We then average their targets using the true spawn probabilities:

`Search(s,a) = sum_over_spawns p(spawn) * [learning_reward + gamma * max_legal Q(next_board)]`.

`agents/q_planning.py` contains this expectation. `outcomes` generates boards and
probabilities, `planned_values` evaluates and averages them, and `act` chooses
the largest legal result. Depth two repeats that calculation before consulting
the network. It costs more computation per move and performs no extra training.

The learning reward must match the checkpoint, including corner/snake shaping.
Natural terminal boards contribute zero continuation. An untrained-network and
zero-leaf control help distinguish learned value information from game-rule search.''')
code('''from rl2048.agents.q_planning import PlanningQAgent
from rl2048.game import legal_actions
path=ROOT/'runs/research/dqn16_long/last'
if path.exists():
    agent=NeuralAgent.load(path)
    planner=PlanningQAgent(agent,reward_mode='corner_snake')
    board,info=Game2048().reset(seed=42)
    with torch.no_grad(): q=agent.policy(torch.tensor(encode(board)[None]))[0].numpy()
    search=planner.action_values(board)
    print('Same board:\\n',board)
    for action,name in enumerate(ACTION_NAMES):
        print(name,'legal:',info['action_mask'][action], 'direct Q:',round(float(q[action]),3),
              'one-move search:',round(float(search[action]),3))
    print('Direct move:',ACTION_NAMES[agent.act(board,info['action_mask'])])
    print('Planning move:',ACTION_NAMES[planner.act(board,info['action_mask'])])
results=ROOT/'runs/research/q_planning_comparison.json'
if results.exists():
    for name,result in json.loads(results.read_text()).items():
        print(name,'direct mean:',round(result['greedy']['mean_score']),
              'planning mean:',round(result['planning']['mean_score']))''')
book=n.v4.new_notebook(cells=cells)
book.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
NotebookClient(book,timeout=120,kernel_name='python3',resources={'metadata':{'path':str(root)}}).execute()
n.write(book,root/'lessons/04_sixteen_inputs_four_q_values.ipynb')
print('Executed DQN notebook',len(cells),'cells')
