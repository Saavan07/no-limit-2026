**Broader scout validation after the human-coalition refits**

The exact coalition study used outside scouts of 0,1,3. Its four reference best
responses are now known and saved in post-definition-reference-optima.json. Freeze
those responses, the human vectors/range representatives, and original primary
machine vectors before constructing this stress test. Preserve all prior outputs.

The user's narrative also allows limpers with up to 24 troops. Draw nonpriority
bids independently and uniformly from 0 through 24, then allocate the remaining
budget on the priority coalition with a one-soldier minimum on each target.
Use either uniform positive compositions or value-weighted multinomial allocation.
Uniform compositions are sampled through a Dirichlet-multinomial with all parameters
one. For five-heavy cases, reject complete proposal rows until castle 5 has at least
50. This conditions the full joint proposal and does not silently rebalance budgets.

Use the same five priority sets, two laws, open variants for all sets, and separate
five-heavy variants for the three sets containing castle 5: 16 variants, 100,000
independent accepted draws each. Save every legal draw and seed. Create four
reference averages matching the original exact study's all/favorite and open/heavy
conditions. All means are conditional on the saved sample tables; report a simultaneous
DKW score bound conditional on these generator definitions. There is no applicant
validation or inferred prevalence claim.

Evaluate all previously frozen candidates. Compute new reference DP optima only as
post-evaluation benchmarks, not additions to the frozen candidate comparison set.
Report mean points and match-win credit separately. Do not retune this study after
seeing the broader-scout results. Model any further repair as a subsequent study.
