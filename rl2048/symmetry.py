"""Square symmetries transform states, action labels, and legal masks together."""
import numpy as np


def transform_batch(batch,rotation=0,reflect=False):
    result=dict(batch)
    mapping=np.array([0,3,2,1]) if reflect else np.arange(4)
    mapping=(mapping-rotation)%4
    for key in ('states','next_states'):
        if key in batch:
            board=batch[key].reshape(-1,4,4)
            if reflect:
                board=board[:,:,::-1]
            result[key]=np.rot90(board,rotation,axes=(1,2)).reshape(-1,16).copy()
    result['actions']=mapping[batch['actions']]
    for key in ('masks','next_masks'):
        if key in batch:
            mask=np.empty_like(batch[key]); mask[:,mapping]=batch[key]
            result[key]=mask
    return result
