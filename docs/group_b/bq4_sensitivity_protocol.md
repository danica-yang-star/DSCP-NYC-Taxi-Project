# B-Q4 sensitivity and stability protocol

## Purpose

B-Q4 tests whether the main B-Q2 conclusions depend on a small number of analysis choices. It does not replace the frozen primary model and does not select a new model from favorable results.

## Frozen sensitivity scenarios

1. Primary log-normal M1 model.
2. Exclude tips strictly above the pooled 99.9th percentile.
3. Exclude records with `tip_rate > 1`.
4. Fare-only continuous specification.
5. Distance-only continuous specification.
6. Gamma GLM with log link and the primary predictor set.
7. `log(tip_rate)` sensitivity outcome without fare as a predictor.

The tip-rate model answers a different relative-tipping question and is therefore not treated as a competing primary model.

## Temporal stability

The primary model is refit four times, omitting 2022, 2023, 2024, and 2025 in turn. These fits retain at least 36 year-month clusters. Stability is judged from direction, magnitude, and interval overlap rather than p-values alone.

The execution quality gate checks completeness, finite estimates, row reconciliation, and expected cluster counts. A sign change or materially different magnitude is reported as a substantive sensitivity finding; it does not make a successfully completed run fail.

## Reported effects

- percentage difference for a 10% fare increase;
- average percentage difference after adding one trip mile;
- percentage difference for one calendar year;
- Lyft-versus-Uber percentage difference.

All results remain conditional associations among trips with recorded positive tips. Tail exclusions and alternative outcomes are diagnostic checks, not evidence that excluded records are erroneous.
