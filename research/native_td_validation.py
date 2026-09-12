"""Import the native TD return-length comparison and evaluate fixed settings."""
import argparse,json
from pathlib import Path
from rl2048.import_tdl import import_weights
from rl2048.ntuple_train import evaluate_fast

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['control','lambda'],required=True)
    args=p.parse_args();root=Path('runs/research');name='native_td_'+args.variant
    provenance=('Locally trained from local_selected; 1,002,000 additional native backward-TD episodes, '
                'alpha .03; '+('5-step truncated lambda return, lambda .5' if args.variant=='lambda' else 'one-step TD'))
    agent=import_weights(root/'local_selected'/f'{name}.w',root/name,provenance)
    agent.training_games=2307600
    agent.metadata.update({'parent_checkpoint':'local_selected','additional_training_games':1002000,
                           'training_transitions':'Not recovered for native phases; do not infer from episode counts',
                           'algorithm_backend':'Pinned MIT TDL2048+ native executable'})
    agent.save(root/name)
    results={}
    for depth,threshold in [(1,0),(2,0),(3,16384)]:
        agent.depth=depth;agent.downgrade_threshold=threshold
        summary,episodes=evaluate_fast(agent,range(6000000,6000100),workers=6)
        results[f'depth{depth}_downgrade{threshold}']={'summary':summary,'episodes':episodes}
        (root/name/'validation.json').write_text(json.dumps(results,indent=2))
        print(name,depth,threshold,summary['mean_score'],summary['max_score'],flush=True)
