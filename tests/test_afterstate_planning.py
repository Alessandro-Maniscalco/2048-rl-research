import numpy as np
import torch
import pytest
from argparse import Namespace
import json

from rl2048.agents.afterstate_mlp import AfterstateLearner, AfterstateMLPAgent
from rl2048.agents.afterstate_planning import AfterstateLookahead
from rl2048.agents.ntuple import encode
from rl2048.game import move, legal_actions
from rl2048.view import save_replay
from research.afterstate_search_experiment import run


def test_two_moves_matches_independent_raw_board_enumeration():
    torch.set_num_threads(1)
    learner = AfterstateLearner(width=8, gamma=.5)
    with torch.no_grad(): learner.policy.layers[-1].bias.fill_(6.)
    agent = AfterstateMLPAgent(learner)
    # Almost full, asymmetric board: different terminal probabilities by move.
    board = np.array([[2,4,8,16],[4,8,16,32],[8,16,32,64],[16,32,2,2]])
    terminal = np.array([[2,4,8,16],[4,8,16,32],[8,16,32,64],[16,32,64,128]])
    planned = AfterstateLookahead(agent, spawn_batch=1).planned_values(np.stack([encode(board), encode(terminal)]))
    manual = np.full(4, -np.inf)
    for a in np.flatnonzero(legal_actions(board)):
        after, gain, _ = move(board, int(a)); empties=np.argwhere(after==0)
        expected = 0.
        for row,col in empties:
            for tile,mass in [(2,.9),(4,.1)]:
                state=after.copy();state[row,col]=tile
                legal=legal_actions(state)
                best=0. if not legal.any() else max(move(state,int(b))[1]/128+.5*6 for b in np.flatnonzero(legal))
                expected += mass/len(empties)*best
        manual[a]=gain/128+.5*expected
    np.testing.assert_allclose(planned[0],manual,rtol=1e-6,atol=1e-6)
    assert np.isneginf(planned[1]).all()
    # Batching and chunk size must not change probabilities or action values.
    other=AfterstateLookahead(agent, spawn_batch=256).planned_values(encode(board)[None])
    np.testing.assert_allclose(planned[:1],other,rtol=1e-6,atol=1e-6)
    # Independently recurse using raw boards and public game rules. The third
    # move needs two correctly weighted chance layers and three reward terms.
    def exact(state,depth):
        values=np.full(4,-np.inf)
        for a in np.flatnonzero(legal_actions(state)):
            after,gain,_=move(state,int(a))
            if depth==1: future=6.
            else:
                empty=np.argwhere(after==0);future=0.
                for row,col in empty:
                    for tile,mass in [(2,.9),(4,.1)]:
                        nxt=after.copy();nxt[row,col]=tile
                        future+=mass/len(empty)*(exact(nxt,depth-1).max() if legal_actions(nxt).any() else 0.)
            values[a]=gain/128+.5*future
        return values
    third=AfterstateLookahead(agent,depth=3).planned_values(encode(board)[None])[0]
    np.testing.assert_allclose(third,exact(board,3),atol=1e-6,rtol=1e-6)


def test_budget_resume_preserves_game_seed_and_score(tmp_path):
    agent=AfterstateMLPAgent(AfterstateLearner(width=8))
    checkpoint=tmp_path/'checkpoint';agent.save(checkpoint,dict(width=8))
    args=Namespace(checkpoint=checkpoint,out=tmp_path/'study',games=1,seed_start=8940000,
                   spawn_batch=256,seconds=0,stop_file=tmp_path/'STOP',resume=False)
    run(args)
    partial=json.loads((args.out/'evaluation.json').read_text())
    assert not partial['complete'] and partial['mean_score'] is None
    # Older CPU evaluations have neither of these optional protocol fields.
    config=json.loads((args.out/'config.json').read_text())
    config.pop('device');config.pop('nonnegative_leaf')
    (args.out/'config.json').write_text(json.dumps(config))
    args.resume=True;args.seconds=60;run(args)
    completed=json.loads((args.out/'evaluation.json').read_text())
    reference=save_replay(AfterstateLookahead(agent),tmp_path/'reference.html',seed=args.seed_start)
    assert completed['complete'] and not completed['episodes'][0]['truncated']
    assert completed['episodes'][0]['score']==reference['score']
    assert completed['episodes'][0]['length']==reference['length']
    args.nonnegative_leaf=True
    with pytest.raises(ValueError,match='unchanged checkpoint'):
        run(args)


def test_nonnegative_leaf_preserves_known_rewards_and_terminal_mask():
    torch.set_num_threads(1)
    learner=AfterstateLearner(width=8,gamma=.5)
    with torch.no_grad(): learner.policy.layers[-1].bias.fill_(-6.)
    agent=AfterstateMLPAgent(learner)
    board=np.array([[2,4,8,16],[4,8,16,32],[8,16,32,64],[16,32,2,2]])
    terminal=np.array([[2,4,8,16],[4,8,16,32],[8,16,32,64],[16,32,64,128]])
    def exact(state,depth):
        values=np.full(4,-np.inf)
        for a in np.flatnonzero(legal_actions(state)):
            after,gain,_=move(state,int(a));future=0.
            if depth>1:
                empty=np.argwhere(after==0)
                for row,col in empty:
                    for tile,mass in [(2,.9),(4,.1)]:
                        nxt=after.copy();nxt[row,col]=tile
                        future+=mass/len(empty)*(exact(nxt,depth-1).max() if legal_actions(nxt).any() else 0.)
            values[a]=gain/128+.5*future
        return values
    for depth in [1,2,3]:
        values=AfterstateLookahead(agent,depth=depth,nonnegative_leaf=True).planned_values(
            np.stack([encode(board),encode(terminal)]))
        np.testing.assert_allclose(values[0],exact(board,depth),rtol=1e-6,atol=1e-6)
        assert np.isneginf(values[1]).all()
    # The option is inference-only; it does not change the network's weights.
    assert learner.policy.layers[-1].bias.item()==-6.
    with torch.no_grad(): learner.policy.layers[-1].bias.fill_(6.)
    np.testing.assert_allclose(
        AfterstateLookahead(agent,nonnegative_leaf=True).planned_values(encode(board)[None]),
        AfterstateLookahead(agent).planned_values(encode(board)[None]))
