# B-Q2 main-effects protocol

## Question

Among valid trips with a recorded positive tip, how are fare, distance, pickup time, pickup borough, and platform associated with tip amount after mutual adjustment?

This is a conditional observational analysis. It does not address whether a passenger tips and does not identify causal platform, borough, fare, or distance effects.

## Frozen response model

B2 selected OLS for `log(tips)` with Duan smearing for dollar-scale predictions. The choice was frozen after a 2025 temporal validation on the complete 903,434-row Group B sample. B-Q2 does not reopen the response-family comparison.

## Nested models

`M0_trip_time` contains transformed fare and distance, pickup hour, weekday, month, and a centered linear year trend.

`M1_primary_main_effects` adds pickup borough and platform. M1 is the primary B-Q2 model. The change in R-squared is descriptive evidence of incremental fit; terms are not retained or removed from p-values.

## Inference

Standard errors are clustered by year-month, producing 48 clusters in the full analysis. The finite-cluster correction is enabled. With a very large row count, interpretation emphasizes effect sizes and confidence intervals instead of statistical significance alone.

## Effect reporting

- Fare: percentage change in conditional tip associated with a 10% fare increase.
- Distance: average change in retransformed predicted tip after adding one mile to every trip while holding other recorded fields fixed.
- Year: percentage change associated with a one-year increase in the fitted time trend.
- Platform: Lyft versus Uber percentage difference and standardized predicted dollar means.
- Borough: percentage differences relative to Manhattan and standardized predicted dollar means.

The 29 `Other/Unknown` borough observations remain in the population but are pooled with the Manhattan adjustment, as frozen before candidate fitting.

## Boundaries

Platform-by-borough and platform-by-year interactions are not included in B-Q2. They are reserved for B-Q3. Tail exclusions, tip-rate outcomes, fare/distance alternative specifications, and temporal stability checks are reserved for B-Q4.
