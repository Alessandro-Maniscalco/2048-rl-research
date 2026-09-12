"""One paired experiment: same transitions, afterstate target, symmetry, budget.

The collection policy alternates neural/table preferences across environments
and time, with identical epsilon exploration. Each learner fits every collected
afterstate. Optimizers differ: Adam/MSE versus collision-normalized table SGD.
This isolates more factors than comparing pretrained tables with direct DQN,
but is not equal parameter count or fully tuned optimal learning for either.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from rl2048.afterstate_compare import NeuralValue,TupleValue,greedy_values,td_targets,evaluate_batch
from rl2048.vector_game import VectorGame

def run(args):
    args.out.mkdir(parents=True,exist_ok=False)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    (args.out/'config.json').write_text(json.dumps(config,indent=2))
    torch.set_num_threads(2);torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    neural=NeuralValue(args.width,args.encoding,args.device,args.lr)
    table=TupleValue(args.layout,args.alpha)
    env=VectorGame(args.envs,args.seed);steps=updates=0;start=time.time()
    end=min(args.deadline,start+args.seconds);next_log=0;next_eval=args.eval_every
    timings=dict(neural_seconds=0.,table_seconds=0.,environment_seconds=0.)
    history=[]
    while steps<args.steps and time.time()<end:
        t=time.perf_counter();qn,after,legal=greedy_values(env.boards,neural);timings['neural_seconds']+=time.perf_counter()-t
        t=time.perf_counter();qt,_,_=greedy_values(env.boards,table);timings['table_seconds']+=time.perf_counter()-t
        t=time.perf_counter()
        actions=np.where((np.arange(args.envs)+updates)%2==0,qn.argmax(1),qt.argmax(1))
        random_actions=np.where(legal,rng.random(legal.shape),-np.inf).argmax(1)
        epsilon=max(.01,1-steps/args.epsilon_steps*.99)
        actions=np.where(rng.random(args.envs)<epsilon,random_actions,actions)
        selected=after[np.arange(args.envs),actions].copy()
        _,next_states,term,trunc,completed=env.step(actions)
        timings['environment_seconds']+=time.perf_counter()-t
        t=time.perf_counter();yn=td_targets(next_states,neural);ln=neural.fit(selected,yn);timings['neural_seconds']+=time.perf_counter()-t
        t=time.perf_counter();yt=td_targets(next_states,table);lt=table.fit(selected,yt);timings['table_seconds']+=time.perf_counter()-t
        steps+=args.envs;updates+=1
        if updates%args.target_every==0:neural.sync();table.sync()
        if time.time()-start>=next_log:
            row=dict(transitions=steps,updates=updates,seconds=time.time()-start,epsilon=epsilon,
                     neural_mse=ln,table_mse=lt,**timings)
            history.append(row);(args.out/'progress.json').write_text(json.dumps(history,indent=2))
            print(json.dumps(row),flush=True);next_log+=15
        if steps>=next_eval:
            for label,model in [('neural',neural),('table',table)]:
                result=evaluate_batch(model,range(8200000,8200016),deadline=min(args.deadline,time.time()+90))
                result['training_transitions']=steps
                (args.out/f'validation_{label}_{steps}.json').write_text(json.dumps(result,indent=2))
            neural.save(args.out/'neural');table.save(args.out/'table');next_eval+=args.eval_every
    neural.save(args.out/'neural');table.save(args.out/'table')
    meta=dict(transitions=steps,updates=updates,seconds=time.time()-start,complete_budget=steps>=args.steps,
        neural_parameters=sum(p.numel() for p in neural.model.parameters()),table_parameters=int(table.weights.size),
        **timings,protocol='Fresh zero-output models; identical online transitions, epsilon and target-copy cadence. Gamma1, raw rewards/128, no shaping. D4 mean neural/D4 shared tables. No replay; one update per256 transitions. Optimizers differ.')
    (args.out/'training.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=1_000_000);p.add_argument('--seconds',type=float,default=600)
    p.add_argument('--deadline',type=float,default=float('inf'));p.add_argument('--seed',type=int,default=0)
    p.add_argument('--width',type=int,default=512);p.add_argument('--encoding',default='relational')
    p.add_argument('--device',default='mps');p.add_argument('--layout',default='4x6')
    p.add_argument('--alpha',type=float,default=.03);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--envs',type=int,default=256);p.add_argument('--epsilon-steps',type=int,default=500_000)
    p.add_argument('--target-every',type=int,default=64);p.add_argument('--eval-every',type=int,default=250_000)
    run(p.parse_args())
