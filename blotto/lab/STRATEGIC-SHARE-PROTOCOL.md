**Complete strategic-share maps, version 1**

Use the four declared behavior references and strategic reference from the completed
contamination study. These are known hypotheses. No new holdout or prevalence claim
is made. Human strategy input remains reserved.

For each reference, solve the pure best response to (1-s)*behavior+s*strategic for
every real s in [0,1]. The objective is linear in s for each fixed allocation, so
the optimum is a convex piecewise-linear envelope of finitely many lines. Use exact
rational DP oracles to discover every interval and boundary. At each endpoint, choose
the appropriate extreme slope among primary optima to avoid losing a phase through
lexicographic tie choice. Count every primary optimum at each true boundary and
inside each open interval. Do not interpret s as an estimated applicant proportion.

Independently verify each segment's chosen allocation is globally optimal at both
ends using ordinary exact best-response DP. Its difference from any other fixed
allocation is linear, so endpoint optimality proves the full segment. Recheck all
boundary and interior tie counts, complete coverage, and consistency with the prior
five sampled share settings. Save exact fractions, queries, source, and artifact
hashes. Distinguish mathematical precision conditional on the reference from
uncertainty about those references. This study does not choose a final submission.
