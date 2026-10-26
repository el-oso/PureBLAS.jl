# Probes live HERE, not in /tmp

Throwaway measurement scripts go in this directory, **not** in a scratch dir outside the repo.

Not tidiness — enforcement. `test/estimator_lint.jl` scans `bench/` for unapproved timing reductions
(`minimum`/`mean` over timings instead of the approved `median`). A probe written in `/tmp` is invisible
to that lint, and that is exactly where the 2026-08-03/04 failure happened: the shipped kernels obeyed
the rule, the probes did not, and the probes are what drove the decisions.

Contents are gitignored — the scripts are disposable, the linting is not.

Rules for anything in here:
- Reduce timings with `Measure.tstat` from `bench/measure.jl` (median). Never `minimum`, never `mean`.
- When reporting a number, report its estimator and sample count with it (`Measure.report`).
- There is no annotation that exempts a reduction — `test/estimator_lint.jl` removed both of its
  escape hatches because each was talked around. To show a distribution, print quantiles (`q1`/`q3`,
  or the samples in sorted order) beside the median rather than `minimum`/`maximum`.
