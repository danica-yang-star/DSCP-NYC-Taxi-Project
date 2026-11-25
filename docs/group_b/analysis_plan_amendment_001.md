# Analysis plan amendment 001

## Timing

This amendment is recorded after the B1 data audit and before fitting any candidate model.

## Reason for the amendment

The frozen B1 plan proposed a categorical year term while also reserving all of 2025 for validation. A model developed only on 2022–2024 cannot estimate a separate 2025 category coefficient. Using an unestimated 2025 indicator would make the temporal validation incoherent.

## Resolution

For candidate development and 2025 validation:

- replace categorical `year` with `year_centered = year - 2022`;
- replace platform × categorical year with platform × `year_centered`;
- retain categorical month to adjust for recurring seasonality;
- after selecting a response family, refit the same trend specification on 2022–2025.

This is a prospective implementation correction, not a response to model results.

## Sparse borough handling

The B1 audit found 29 `Other/Unknown` observations among 903,434 Group B rows. These observations remain in the analysis population, but for model adjustment they are pooled with the Manhattan reference level and are not assigned a separate geographic interpretation. Platform × borough interactions are created only for Bronx, Brooklyn, Queens, and Staten Island.

## Fare–distance collinearity

The B1 Pearson correlation between fare and distance was 0.862, and the correlation after the planned log transformations was 0.866. The primary candidate formula retains both because they represent distinct recorded trip attributes. B2 reports the two-variable VIF and later sensitivity fits will compare both, fare-only, and distance-only specifications. No term will be removed solely because its p-value is large.
