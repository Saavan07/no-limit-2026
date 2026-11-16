**Game 2 population-family and complete C2/C4 split challenge**

Preserve every previous recommendation, candidate freeze, source snapshot and result. The new candidate freeze contains the same 760 legal vectors as the external replay, including all 101 allocations `(0,b,0,100-b,0)` for integer b from zero through one hundred. Candidate selection does not admit any allocation after new results are observed.

The study has three distinct parts. It does not count deliberate counterpopulations as equally likely applicant forecasts.

**Freshly checked population models**

Use the supplied forty-family catalog as the base. Build twelve predefined variants: unchanged; target-family scouts fixed at zero; target-family scouts uniform from zero through twenty-four; target-family scouts from five through sixteen; uniform active target splits; splits concentrated near equal bids; splits concentrated near value-proportional bids; asymmetric target splits; sixfold within-family mass on castle-4 bids from sixty-one through seventy-two; an explicit castle-2/4 family with castle-4 bids sixty through seventy-five and scouts zero through three; and one- or five-troop transfer perturbations of all thirty-five non-atom families. Keep the five named exact point families intact. Every generated vector is a legal full joint allocation.

Fit each catalog using a sampled cubic payoff polynomial. The coefficient table conditions on every unordered triple of family identities. Own action samples are stratified by family, with independent sampled opponent allocations inside each conditional triple. Expand the symmetric coefficients to compute exact analytic derivatives of that sampled surrogate. Validate those derivatives against finite differences and a direct polynomial on a small independent example.

The fitting loss assigns half its CDF weight to the original twenty-six thresholds and half to a 181-point grid. It uses multiple initializations and fixed bandwidths, without pretending to be a likelihood or posterior. For the base catalog, also fit every one of the forty omissions with that family's mass fixed at zero. Save every optimizer termination status. A failure to find an acceptable field does not prove no acceptable field exists.

Freeze the selected fitted weights before independent CDF validation. Include the seventeen supplied fields as separately rechecked baselines. Validate with 256 own draws per family and 131,072 fresh opponent trios. Retain a model only if the full CDF maximum gap is at most 0.085, its 181-grid mean gap and original-threshold mean gap are both at most 0.035, and modeled incumbent mass above the public maximum plus 0.2 is at most 0.015. These are diagnostic tolerances with numerical and public-score noise, not confidence regions for hidden allocations.

For every retained field, draw 1,048,576 fresh legal opponent trios. Score every one of the 101 exact C2/C4 splits on all groups. Score the complete frozen 760-vector comparison pool on the first 262,144 groups. Keep both rankings separate because their numerical precision differs. The primary rule maximizes the lowest expected placement payout across retained fields; exact ties use lexicographic allocation order. Do not average models into an invented applicant-frequency distribution.

Save all integer sums and the complete 101-by-101 matrix of payout cross-products. This supports paired comparisons of every split on the same opponents. For split means and all unordered split pairs, use two-sided, multiplicity-adjusted empirical Bernstein bounds derived from [Maurer and Pontil, Theorem 4](https://www.cs.mcgill.ca/~colt2009/papers/012.pdf). The union covers 101 mean events and 5,050 pair events per retained field at joint alpha 0.05. The range is nine for payouts and eighteen for payout differences. The sample variance uses denominator N minus one. The intervals cover sampling in the specified frozen fields only. They do not certify an optimum in the hidden applicant population.

**Deliberate family stresses**

Evaluate all 760 candidates against every distinct pure-family law in all twelve catalogs, deduplicating byte-identical normalized laws. Keep the full mapping from every catalog/family slot to its law. These are diagnostic opponents even when they do not match the public score distribution.

For each of the seventeen supplied fields and every original family, draw the conditional cases with one or two opponents from that replacement family. Use the base population for zero replacements and its pure-family law for three. This covers all forty single-family replacement directions, every exact C2/C4 split, and fixed outside-family reference candidates. Opponents retain complete legal allocations; fixed opponent slots are valid because identities are exchangeable.

Mix the four conditional scores using their binomial probabilities. Minimize the resulting cubic on each declared replacement-share interval, including interior stationary points. This covers every share in that interval for the estimated conditional polynomial. Opponent sampling and model error still remain. One-family replacement directions do not exhaust all simultaneous mixtures of several new families.

**Independent verification and reporting**

Regenerate every catalog and new opponent bank. Reconstruct all model constraints, retention decisions and selections. Check all candidate scorers on declared independent subsets and focal/nearby splits on full opponent banks. Reconstruct confidence calculations from integer first and second moments. Check all fitted CDF knots independently. Verify every pure-family law and conditional replacement bank on a declared scoring subset. Confirm that original freezes and recommendation records remain unchanged.

Report both the winning point estimates and uncertainty among nearby splits, which catalog changes move the result, every family-omission outcome, and the full one-hundred-one-split curves. Retain failed model fits and deliberately severe stresses as separate evidence. No application is submitted by this study.

**Supplement: the outside-family result and equal-precision refinement**

The full 760-vector comparison identified a C3/C4 allocation outside the initially emphasized C2/C4 family. All 101 exact C3/C4 splits were already in the pre-evaluation candidate freeze. Score them on the same full 1,048,576 saved groups per retained field, keeping the original C2/C4 and prefix results unchanged. This is a precision refinement of frozen actions, not a fresh population test or new candidate admission.

Allocate the final joint alpha of 0.05 in three equal parts: all 760 prefix means; the predeclared 101 C2/C4 full means and 5,050 unordered pairs; and the potential full-group means and unordered pairs of all 760 frozen actions. The third class uses B times (760 plus 760 choose 2) comparisons, where B is the retained field count. It protects the choice to investigate C3/C4 after looking at the prefix comparison, even though only the needed C3/C4 subset of those potential full-group statistics is computed. Use the larger full-pool count for its empirical Bernstein bounds. The initially narrower selected-family confidence calculation was widened before the final refinement audit; source and scope history are preserved with the correction record.

Compare a refined C3/C4 lower bound with the largest prefix upper bound among frozen actions outside C3/C4. Separation supports the candidate under the declared finite population laws and candidate set. It does not identify the hidden population, distinguish every nearby C3/C4 split, or establish a full legal-action optimum.
