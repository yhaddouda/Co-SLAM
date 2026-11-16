"""Rotating historical pose/ray cohorts with permanent recent-frame anchors.

The phase is deterministic. Probabilities describe the fixed-population average
across a complete phase cycle (or a uniformly randomized seed phase), not fresh
independent random sampling conditional on each evolving optimizer state.
"""
import numpy as np


def choose_pose_cohort(num_keyframes, frame, map_every, count, recent, seed,
                       transition_keyframes=0, transition_period=2):
    """Select recent anchors, an optional maturation band, and older cohorts.

    A history index g has keyframe age num_keyframes-g. The first ``recent``
    ages are always retained; the next ``transition_keyframes`` ages are
    retained every two phases; older observations retain the native count-call
    period. The original seed phase is unchanged. A fixed two-phase transition
    requires an even mature period >=4 when enabled.
    """
    values=dict(num_keyframes=num_keyframes,frame=frame,map_every=map_every,
                count=count,recent=recent,seed=seed,
                transition_keyframes=transition_keyframes,transition_period=transition_period)
    for name,value in values.items():
        if isinstance(value,(bool,np.bool_)) or int(value)!=value:
            raise ValueError(f'{name} must be an integer')
    if num_keyframes<0 or frame<0 or recent<0 or map_every<1 or count<1 or transition_keyframes<0:
        raise ValueError('Invalid cohort population, frame, schedule or anchor budget')
    if transition_period != 2:
        raise ValueError('The maturation transition period is fixed at two')
    if transition_keyframes and (count<4 or count%transition_period):
        raise ValueError('A maturation transition requires an even cohort count >= 4')
    num_keyframes,frame,map_every,count,recent,seed=map(int,
        (num_keyframes,frame,map_every,count,recent,seed))
    phase=(frame//map_every+seed)%count
    indices=np.arange(num_keyframes,dtype=np.int64)
    anchors=(indices==0)|(indices>=max(1,num_keyframes-recent))
    probabilities=np.full(num_keyframes,1./count,dtype=np.float32)
    probabilities[anchors]=1.
    selected=anchors|(indices%count==phase)
    if transition_keyframes:
        ages=num_keyframes-indices
        transition=(~anchors)&(ages<=recent+transition_keyframes)
        probabilities[transition]=1./transition_period
        selected=np.where(transition,indices%transition_period==phase%transition_period,selected)
    active=indices[selected]
    return active,probabilities,phase
