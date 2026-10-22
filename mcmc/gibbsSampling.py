"""Fixed-K sampler support, adapted from Min Xu (2020-2021) and project updates."""
import numpy as np

class InsufficientSamplesError(RuntimeError):
    """Sampling budget exhausted below the required valid-sample count."""

    def __init__(self, valid, total, required):
        self.valid, self.total = (valid, total)
        super().__init__(f'{valid}/{total} anchor-valid samples; require {required}')

def _sample_is_valid(sample, node_to_component, n_anchors):
    """Require exactly one root in every anchor set."""
    roots = sample['roots']
    if any((r not in node_to_component for r in roots)):
        return False
    comps = [node_to_component[r] for r in roots]
    return len(comps) == n_anchors and len(set(comps)) == n_anchors

def gibbsFullCT_multichain_until_valid(graf, anchors, K, alpha, beta, target_valid, burn, samples_per_chain, max_chains, min_valid=None, size_thresh=0.1, birth_thresh=0.8, display=False, seed_mode="two_node"):
    """Pool independent chains up to the target or the sampling budget."""
    if min_valid is None:
        min_valid = target_valid
    node_to_component = {}
    for k, comp in enumerate(anchors):
        for node in comp:
            node_to_component[node] = k
    n_anchors = len(anchors)
    all_samples = []
    n_valid = 0
    n_chains_run = 0
    while n_valid < target_valid and n_chains_run < max_chains:
        res = gibbsFullCT(graf.copy(), seed_mode=seed_mode, Burn=burn, M=samples_per_chain, gap=1, alpha=alpha, beta=beta, K=K, display=display, size_thresh=size_thresh, birth_thresh=birth_thresh)
        n_chains_run += 1
        all_samples.extend(res['samples'])
        n_valid += sum((_sample_is_valid(s, node_to_component, n_anchors) for s in res['samples']))
    if n_valid < min_valid:
        raise InsufficientSamplesError(n_valid, len(all_samples), min_valid)
    return {'samples': all_samples, 'node_tree_coo': np.zeros((len(graf.vs), 0))}

def gibbsFullCT(graf, Burn=40, M=50, gap=1, alpha=0, beta=1, K=1, display=True, size_thresh=0.01, birth_thresh=0.8, **options):
    """Run the paper's fixed-K two-founder sampler."""
    seed_mode = options.pop("seed_mode", "two_node")
    if seed_mode != "two_node":
        raise ValueError("The paper package supports seed_mode='two_node' only")
    from .two_founder import gibbs_two_founder
    return gibbs_two_founder(graf, Burn=Burn, M=M, gap=gap,
                            alpha=alpha, beta=beta, K=K, display=display,
                            size_thresh=size_thresh, birth_thresh=birth_thresh,
                            **options)
