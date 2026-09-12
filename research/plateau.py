"""A practical stopping rule for noisy fixed-game policy scores, not a proof."""
import numpy as np


def plateau_status(curve, *, min_transitions, patience=8, window=3, min_gain=.02):
    if patience < 1 or window < 1 or min_transitions < 0 or min_gain < 0:
        raise ValueError('Invalid plateau settings')
    anchor=None
    last_improvement=0
    rolling=None
    for i in range(window-1,len(curve)):
        rolling=float(np.median([r['mean_score'] for r in curve[i-window+1:i+1]]))
        if anchor is None or rolling > anchor + max(1.,abs(anchor)*min_gain):
            anchor=rolling
            last_improvement=i
    stale=max(0,len(curve)-1-last_improvement)
    reached=bool(anchor is not None and curve[-1]['transitions']>=min_transitions and stale>=patience)
    return dict(reached=reached,rolling_median=rolling,significant_best=anchor,
                evaluations_without_improvement=stale,min_relative_gain=min_gain,
                window=window,patience=patience,min_transitions=min_transitions)
