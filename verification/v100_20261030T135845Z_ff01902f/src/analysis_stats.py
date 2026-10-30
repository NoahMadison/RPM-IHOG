"""Token-weighted paired bootstrap over contiguous evaluation blocks."""
import numpy as np
def weighted_block_delta(reference,converted,seed=919,replications=2000):
    assert len(reference)==len(converted)
    counts=np.asarray([r['count'] for r in reference],float)
    assert all(r['count']==c['count'] and r['block']==c['block'] for r,c in zip(reference,converted))
    sums=np.asarray([c['nll_sum']-r['nll_sum'] for r,c in zip(reference,converted)],float)
    rng=np.random.default_rng(seed);values=[]
    for _ in range(replications):
        idx=rng.integers(0,len(counts),len(counts));values.append(float(sums[idx].sum()/counts[idx].sum()))
    return {'mean':float(sums.sum()/counts.sum()),'ci95':np.quantile(values,[.025,.975]).tolist(),'unit':'contiguous512-token context block; token-weighted ratio of sums','blocks':len(counts),'bootstrap_replications':replications,'seed':seed}
