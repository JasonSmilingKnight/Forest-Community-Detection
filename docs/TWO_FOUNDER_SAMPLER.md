# Two-founder Monte Carlo sampler

The paper uses `mcmc/two_founder.py`, called through `mcmc/gibbsSampling.py::gibbsFullCT`.

Each of K components starts from an ordinary edge joining two founders. Later vertices have one earlier parent. With homogeneous ER noise and a uniform compatible-order prior, the joint target is proportional to `product_v product_{j=1}^{degree_F(v)-1}(beta*j+alpha)`.

- `sample_parents()` fixes the founder edges during the sweep. Remove the old parent edge of each later vertex and choose an earlier observed neighbor w with weight `beta*degree_Fminus(w)+alpha`.
- `sample_ordering()` selects roots by history counts, draws local compatible histories by subtree-size-weighted frontier sampling, permutes the components, puts founder pairs first, and interleaves the remaining histories by their remaining lengths.
- `gibbs_two_founder()` alternates these updates and stores roots, assignments, sizes, and founder pairs after burn-in.
- `gibbsFullCT_multichain_until_valid()` collects independently initialized chains until the valid-sample target or budget is reached. `Leaf_stripping_postCT()` requires exactly one root in each anchor set, aligns labels, and averages assignments.

Simulation defaults: burn-in 30, 200 draws per chain, at most five chains, target 200 valid samples, minimum 50. Fewer than 50 valid samples or a posterior hard classification with fewer than K communities triggers the explicitly recorded distance fallback when enabled. Other runtime errors are recorded as failed trials.

The parent update is a Gibbs sweep preserving the conditional distribution, not an independent draw. The ordering draw is uniform over compatible paired-founder histories. No imaginary loops are used. General ergodicity is not established; when n=2K, the chain cannot change its initial matching. The large paper examples are outside that special case.

Run the exact conditional/ordering checks with `python -m unittest discover -s tests -p test_two_founder.py -v`.
