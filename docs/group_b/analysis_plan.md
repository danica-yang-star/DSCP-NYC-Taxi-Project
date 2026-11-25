# Group B analysis plan: positive tip amount

## Research question

Among valid NYC HVFHV trips with a recorded positive tip, what factors are associated with the amount tipped?

This is an observational, conditional analysis. Results describe associations among trips with `tips > 0`; they do not identify causal effects and do not describe the decision to leave a tip.

## Analysis population

The Group B population is frozen before modeling:

```text
valid_for_primary_model == True
tips > 0
```

Group B uses the same shared cleaning contract as Group A and then applies the positive-tip condition. The modeling artifact is the deterministic 0.5% monthly sample produced by the approved 48-task Slurm array.

## Candidate outcome models

Two prespecified candidates will be compared:

1. OLS for `log(tips)`, interpreted on a multiplicative scale and retransformed for dollar-scale validation.
2. Gamma GLM for `tips` with a log link.

The candidates will be developed on 2022–2024 data and compared on 2025 data using dollar-scale MAE, RMSE, calibration, residual diagnostics, and interpretability. Because the candidates use different response scales and likelihoods, AIC values will not be used as the sole cross-family selection rule. After choosing the response distribution, the selected specification will be refit on 2022–2025 data.

`tip_rate = tips / base_passenger_fare` is a sensitivity outcome, not the default primary outcome. A ratio outcome can introduce denominator-driven behavior and will therefore be interpreted cautiously.

## Predictors and transformations

Continuous predictors:

- `log(base_passenger_fare)`
- `log1p(trip_miles)`

Categorical predictors:

- pickup hour
- pickup weekday
- pickup borough
- platform
- year
- month

The fare–distance relationship will be audited before modeling. Both terms remain prespecified unless diagnostics show that their joint inclusion makes the model numerically unstable.

## Prespecified interactions

Only two interaction families are planned for the main comparison:

- platform × pickup borough
- platform × year

They represent plausible platform differences across geography and calendar time. Additional interactions require a written amendment before fitting.

## Tail and missing-data policy

- The primary analysis retains every eligible sampled trip.
- No observation is deleted solely because it is statistically extreme.
- Sensitivity analysis 1 excludes tips above the pooled 99.9th percentile.
- Sensitivity analysis 2 separately examines `tip_rate > 1` records.
- Missing or invalid values are governed by the approved shared cleaning contract.

## Inference and reporting

- Use cluster-robust standard errors by year-month.
- Report estimates, confidence intervals, and practical effect sizes.
- Do not select terms solely from p-values.
- Retain null, unfavorable, and heterogeneous findings.
- Interpret results as conditional associations, never as causal platform or location effects.

## Reproducibility boundary

The raw TLC files are not committed to Git. Code, configuration, aggregate audits, tests, figures, and model outputs may be committed. Generated modeling samples remain in the versioned HPC/Google Drive data package.
