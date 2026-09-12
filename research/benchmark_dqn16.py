import json,time
from pathlib import Path
import numpy as np
import torch
from rl2048.agents.dqn import DQN
from rl2048.offline_train import tensor_batch

torch.set_num_threads(4)
rng=np.random.default_rng(0);results=[]
for device in ('cpu','mps'):
    for size in (256,1024,4096,16384):
        learner=DQN(device=device)
        batch={'states':rng.integers(0,12,(size,16),dtype=np.uint8),'next_states':rng.integers(0,12,(size,16),dtype=np.uint8),
               'actions':rng.integers(0,4,size),'rewards':rng.random(size).astype(np.float32),
               'masks':np.ones((size,4),bool),'next_masks':np.ones((size,4),bool),'terminated':np.zeros(size,bool)}
        for _ in range(5):learner.update(tensor_batch(batch,device))
        if device=='mps':torch.mps.synchronize()
        start=time.perf_counter()
        for _ in range(20):learner.update(tensor_batch(batch,device))
        if device=='mps':torch.mps.synchronize()
        seconds=(time.perf_counter()-start)/20
        result={'device':device,'batch':size,'seconds_per_update':seconds,'examples_per_second':size/seconds}
        results.append(result);print(json.dumps(result),flush=True)
Path('runs/research/compute/dqn16.json').write_text(json.dumps(results,indent=2))
