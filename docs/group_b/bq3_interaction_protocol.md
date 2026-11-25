# B-Q3 prespecified interaction protocol

## Questions

1. Does the adjusted Lyft-versus-Uber difference in positive tip amount vary across pickup boroughs?
2. Does the adjusted Lyft-versus-Uber difference change along the 2022–2025 calendar-year trend?

These are conditional observational comparisons among trips with recorded positive tips. They are not causal platform effects.

## Models

`M1_primary_main_effects` is the accepted B-Q2 model.

`M2_prespecified_interactions` adds only:

- platform × pickup borough;
- platform × centered calendar year.

No additional interaction is selected from the data. The 29 `Other/Unknown` borough observations remain pooled with the Manhattan adjustment and do not receive an interaction coefficient.

## Inference

Both models use month-clustered robust covariance with 48 year-month clusters and the finite-cluster covariance correction. Two joint Wald tests are reported: a four-degree-of-freedom platform-by-borough test and a one-degree-of-freedom platform-by-year test.

Because the sample is large, practical heterogeneity is evaluated from borough-year Lyft-versus-Uber percentage contrasts and confidence intervals, not from p-values alone.

## Reporting

The primary output contains 20 contrasts: five boroughs across four calendar years. A second output provides empirical-standardized predicted positive-tip means for both platforms in every borough-year combination.

The analysis does not claim that platform causes higher tips. Differences may reflect passenger composition, fare structure, app interfaces, service mix, or other unmeasured factors.
