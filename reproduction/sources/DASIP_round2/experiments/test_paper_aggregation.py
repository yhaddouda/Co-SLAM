"""Numerical checks for equal-scene weighting and three-seed aggregate SD."""
import math
from refresh_paper_results import aggregate


def main():
    scenes=[dict(dataset='test',scene=s,hash_size=16) for s in ('short','long')]
    runs=[]
    for offset,scene,frames in [(0,'short',100),(2,'long',10000)]:
        for seed in (0,1,2):
            for c,scale in [('BASE',1),('MC',2)]:
                runs.append(dict(dataset='test',scene=scene,hash_size=16,seed=seed,condition=c,frames=frames,ate_cm=scale*(seed+1+offset)))
    # Extra incomplete scenes are not part of the paired-comparison set.
    runs.append(dict(dataset='test',scene='partial',hash_size=16,seed=0,condition='MC',ate_cm=10000))
    result=aggregate(scenes,runs,'test','MC',16,fields=['ate_cm'])
    assert result['BASE_ate_cm']==3 and result['MC_ate_cm']==6
    assert result['BASE_ate_cm_sd']==1 and result['MC_ate_cm_sd']==2
    assert result['ate_cm_change_pct']==100 and result['ate_cm_reduction_pct']==-100
    assert result['scene_count']==2
    try:
        aggregate(scenes,runs+[runs[0]],'test','MC',16,fields=['ate_cm'])
    except AssertionError:
        pass
    else:
        raise AssertionError('Duplicate run was accepted')
    try:
        aggregate(scenes,runs[1:],'test','MC',16,fields=['ate_cm'])
    except KeyError:
        pass
    else:
        raise AssertionError('Missing paired seed was accepted')
    assert math.isfinite(result['BASE_ate_cm_sd'])
    print('PAPER_AGGREGATION_CHECKS_PASSED')


if __name__=='__main__':main()
