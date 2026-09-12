"""Readable PPO loop: collect -> GAE -> shuffled minibatches -> evaluate.

Example: python -m rl2048.ppo_train --out runs/ppo --seconds 300 --device mps
All reported episode scores are raw game scores; learning rewards divide by128.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import torch
from rl2048.agents.neural import sample_actions, BoardNet, NeuralAgent, chosen, masked_log_probs, optimize, tensor_boards
from rl2048.agents.ppo import advantages, clipped_policy_loss
from rl2048.vector_game import VectorGame
from rl2048.evaluate import evaluate
from rl2048.rewards import learning_rewards


def train(args):
    args.out.mkdir(parents=True, exist_ok=False)
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    env = VectorGame(args.envs, args.seed)
    if getattr(args,'pretrained_checkpoint',None):
        from rl2048.agents.pretrained2048 import Pretrained2048
        model=Pretrained2048.from_external(args.pretrained_checkpoint,reset_critic=True).to(args.device)
        config['transfer']='Pretrained actor/encoder retained; critic output reset for raw-score/128 reward.'
        config['architecture']=model.architecture
        config['width']=model.width
        (args.out / 'config.json').write_text(json.dumps(config, indent=2))
    else:
        model = BoardNet(5, args.width, args.architecture).to(args.device)  # four logits + state value
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Compile environment operations before starting the experiment timer.
    env.step(np.array([np.flatnonzero(m)[0] for m in env.masks()]))
    env = VectorGame(args.envs, args.seed)
    start = time.perf_counter()
    steps, iteration = 0, 0
    episode_rows, progress = [], []
    while time.perf_counter() - start < args.seconds:
        collected = {key: [] for key in ('states','masks','actions','logp','rewards','values','next_values','terminated','truncated')}
        for _ in range(args.horizon):
            states, masks = env.boards.copy(), env.masks()
            with torch.no_grad():
                output = model(tensor_boards(states, args.device))
                logp = masked_log_probs(output[:, :4], torch.as_tensor(masks, device=args.device))
                # Sampling on CPU is deterministic under our recorded NumPy seed.
                probs = logp.exp().cpu().numpy()
                actions = sample_actions(probs, rng)
                old_logp = logp.cpu().numpy()[np.arange(args.envs), actions]
                values = output[:, 4].cpu().numpy()
            reward, final_states, term, trunc, episodes = env.step(actions)
            with torch.no_grad():
                next_values = model(tensor_boards(final_states, args.device))[:, 4].cpu().numpy()
            steps += args.envs
            for score, length, tile in episodes[episodes[:, 1] > 0]:
                episode_rows.append({'transitions': steps, 'score': int(score), 'length': int(length), 'max_tile': int(tile)})
            learn_reward = learning_rewards(reward, states, final_states, term, args.gamma, args.reward_mode, args.shaping_scale)
            data = (states,masks,actions,old_logp,learn_reward,values,next_values,term,trunc)
            for key, item in zip(collected, data, strict=True):
                collected[key].append(item)
        arrays = {key: np.asarray(value) for key, value in collected.items()}
        adv, returns = advantages(arrays['rewards'], arrays['values'], arrays['next_values'],
                                  arrays['terminated'], arrays['truncated'], args.gamma, args.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        size = args.envs * args.horizon
        states = tensor_boards(arrays['states'].reshape(size,16), args.device)
        masks = torch.as_tensor(arrays['masks'].reshape(size,4), device=args.device)
        actions = torch.as_tensor(arrays['actions'].reshape(size), device=args.device)
        old_logp = torch.as_tensor(arrays['logp'].reshape(size), device=args.device)
        adv = torch.as_tensor(adv.reshape(size), device=args.device)
        returns = torch.as_tensor(returns.reshape(size), device=args.device)
        losses = []
        for _ in range(args.epochs):
            for indices in np.array_split(rng.permutation(size), max(1, int(np.ceil(size / args.batch)))):
                index = torch.as_tensor(indices, device=args.device)
                output = model(states[index])
                logs = masked_log_probs(output[:, :4], masks[index])
                policy_loss = clipped_policy_loss(chosen(logs, actions[index]), old_logp[index], adv[index])
                value_loss = (output[:, 4] - returns[index]).square().mean()
                entropy = -(logs.exp() * logs).sum(-1).mean()
                loss = policy_loss + .5 * value_loss - args.entropy * entropy
                optimize(optimizer, loss, model.parameters(), .5)
                losses.append([float(policy_loss.detach().cpu()), float(value_loss.detach().cpu()), float(entropy.detach().cpu())])
        iteration += 1
        recent = episode_rows[-100:]
        row = {'iteration': iteration, 'transitions': steps, 'seconds': time.perf_counter()-start,
               'mean_score_100': float(np.mean([r['score'] for r in recent])) if recent else 0.,
               'policy_loss': float(np.mean(losses,axis=0)[0]), 'value_loss': float(np.mean(losses,axis=0)[1]),
               'entropy': float(np.mean(losses,axis=0)[2])}
        progress.append(row)
        print(json.dumps(row), flush=True)
        NeuralAgent(model, 'ppo', args.device, config['width']).save(args.out/'last', config | {'transitions': steps})
        for name, rows in [('progress',progress),('episodes',episode_rows)]:
            if rows:
                with (args.out/f'{name}.csv').open('w', newline='') as file:
                    writer=csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    torch.save({'model': model.state_dict(), 'optimizer':optimizer.state_dict(), 'transitions':steps,
                'torch_rng':torch.get_rng_state(), 'numpy_rng':rng.bit_generator.state}, args.out/'training.pt')
    # Inference on CPU avoids a GPU dispatch for each individual game move.
    agent = NeuralAgent(model.cpu(), 'ppo', width=config['width'])
    summary, episodes = evaluate(agent, seeds=range(6_100_000,6_100_020), max_steps=40_000)
    (args.out/'validation.json').write_text(json.dumps({'summary':summary,'episodes':episodes}, indent=2))
    print('VALIDATION', json.dumps(summary), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True); p.add_argument('--seconds',type=float,default=300)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu'); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--envs',type=int,default=256); p.add_argument('--horizon',type=int,default=64)
    p.add_argument('--batch',type=int,default=4096); p.add_argument('--epochs',type=int,default=4)
    p.add_argument('--width',type=int,default=256); p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--gamma',type=float,default=.99); p.add_argument('--lam',type=float,default=.95)
    p.add_argument('--entropy',type=float,default=.01)
    p.add_argument('--reward-mode',choices=['score','constant','bottom_left','top_right'],default='score')
    p.add_argument('--shaping-scale',type=float,default=10.)
    p.add_argument('--architecture',choices=['mlp','cnn'],default='mlp')
    p.add_argument('--pretrained-checkpoint',type=Path,help='Fine-tune the supported external ml2048 CNN with critic recalibration')
    train(p.parse_args())
