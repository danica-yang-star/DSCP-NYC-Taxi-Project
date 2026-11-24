# Proposed Shared Cleaning Rules

> **Status: PROPOSED FOR FULL-GROUP REVIEW.**
>
> These rules are not final and must not be implemented as the authoritative
> shared cleaning pipeline until all group members have reviewed the evidence.

## Purpose

This document proposes one shared cleaning contract for both project groups.
Group A and Group B must begin from the same validated trip population and use
the same variable definitions, location mapping, platform mapping, and data-quality rules.

Group-specific filtering occurs only after the shared processing step:

- Group A uses all shared model-eligible trips and models `tip_positive`.
- Group B filters the shared model-eligible trips to `tips > 0` and models tip amount.

## Pilot dataset

- Source: NYC TLC HVFHV January 2024 Parquet file
- File size: 472,757,547 bytes
- Raw rows: 19,663,930
- Positive-tip rows: 3,728,929
- Positive-tip rate: 18.9633%
- Platforms observed: HV0003 and HV0005
- Pickup dates outside January 2024: 0
- Pickup LocationIDs outside 1-265: 0
- Missing values in the six proposal variables: 0

The pilot is evidence for the proposed rules, but all-month results may reveal
additional missingness, platforms, schema changes, or data-quality conditions.

## Pilot data-quality evidence

| Condition | All trips | Positive-tip trips |
|---|---:|---:|
| Negative fare | 191 | 21 |
| Zero fare | 7,143 | 21 |
| Negative miles | 0 | 0 |
| Zero miles | 3,286 | 214 |
| Fare or miles nonpositive | Not yet expressed as a union for all trips | 256 |
| Fare and miles both nonpositive | Not yet expressed for all trips | 0 |
| Missing proposal variables | 0 | 0 |
| Invalid pickup LocationID | 0 | 0 |
| Pickup outside expected month | 0 | 0 |
| Unsupported platform | 0 | 0 |

After requiring positive fare and positive miles, the pilot Group B cohort
contains 3,728,673 rows, retaining 99.9931% of positive-tip trips.

## Pilot location evidence

The original lookup read incorrectly treated the literal `N/A` value as a
missing value. Reading the official lookup with `keep_default_na=False` fixed
the problem:

- True unmatched zone rows: 0
- `N/A` borough rows among model-eligible positive-tip trips: 63
- EWR rows among model-eligible positive-tip trips: 1

These valid special labels must not be silently deleted as failed joins.

## Pilot tail evidence

| Measure | Count among positive-tip trips |
|---|---:|
| Tip amount above 50 dollars | 2,094 |
| Tip amount above 100 dollars | 55 |
| Tip rate above 100% | 6,429 |
| Tip rate above 200% | 770 |

| Variable | Median | 99th percentile | 99.9th percentile | Maximum |
|---|---:|---:|---:|---:|
| Tip amount | 4.19 | 25.75 | 47.97 | 196.80 |
| Base passenger fare | 20.74 | 125.88 | 222.69 | 907.59 |
| Trip miles | 3.101 | 30.74 | 58.17 | 285.12 |
| Tip rate | 20.10% | 60.92% | 119.55% | 1111.11% |

The tail is substantial but rare. These observations are not automatically
classified as errors.

## Proposed shared variable contract

### Raw variables required by the proposal

- `tips`
- `base_passenger_fare`
- `trip_miles`
- `pickup_datetime`
- `PULocationID`
- `hvfhs_license_num`

### Variables allowed for validation only

- `dropoff_datetime`
- `trip_time`

These validation variables are not automatically added to the primary model.

### Shared derived variables

- `year`
- `month`
- `pickup_hour`
- `pickup_weekday`
- `pickup_borough_raw`
- `pickup_borough_model`
- `platform`
- `tip_positive`
- `fare_zero_flag`
- `miles_zero_flag`
- `valid_for_primary_model`

## Proposed decisions for group approval

| ID | Question | Recommended decision | Status |
|---|---|---|---|
| D1 | Use the six proposal variables? | Yes. Use them as the shared primary fields. | Pending |
| D2 | Add common derived variables? | Yes. Derive time, borough, platform, and `tip_positive` once in the shared worker. | Pending |
| D3 | Handle missing core values? | Report counts, do not impute, and exclude missing core values from the primary modeling cohort. Revisit if any month exceeds 1% missingness. | Pending |
| D4 | Handle negative and zero tips? | Exclude negative tips as invalid. Retain zero tips because Group A requires them. Group B filters to positive tips later. | Pending |
| D5 | Handle negative and zero fare? | Exclude negative fare from the analysis cohort. Retain zero fare in the audit with a flag, but exclude it from the primary model. | Pending |
| D6 | Handle negative and zero miles? | Exclude negative miles from the analysis cohort. Retain zero miles in the audit with a flag, but exclude it from the primary model. | Pending |
| D7 | Handle special or unmatched borough labels? | Reject invalid LocationIDs. Preserve valid `EWR`, `N/A`, and `Unknown` labels in the raw mapping and combine them as `Other/Unknown` for modeling. | Pending |
| D8 | Remove exact duplicates? | Report duplicate candidates but do not automatically delete them without a defensible unique trip identifier. | Pending |
| D9 | Handle extreme values? | Retain them in the primary analysis. Pre-specify a sensitivity analysis excluding the most extreme 0.1% after pooled thresholds are determined. Do not winsorize the primary response. | Pending |
| D10 | What should each monthly job output? | Produce an audit, a small grouped summary, and a deterministic uniform modeling sample. Do not keep a second full-size cleaned copy by default. | Pending |

## Proposed shared record states

### Invalid for shared analysis

- Missing core response or predictor fields
- Nonfinite core numeric fields
- Negative tips
- Negative fare
- Negative miles
- Pickup datetime outside the file month
- Pickup LocationID outside the official range
- Platform not covered by the group decision

### Retained but not eligible for the primary model

- Zero fare
- Zero miles

These rows remain visible in monthly audits and receive explicit flags.

### Eligible for the primary model

- Required values present and finite
- `tips >= 0`
- `base_passenger_fare > 0`
- `trip_miles > 0`
- Valid pickup month and LocationID
- Approved platform mapping

## Proposed monthly outputs

### 1. Audit JSON

One small file containing schema, row reconciliation, missingness, invalid counts,
zero counts, platform counts, zone counts, runtime, memory, disk, and final status.

### 2. Monthly summary

Small grouped results by year, month, platform, modeled borough, weekday, and hour,
including trip count, positive-tip count, positive-tip rate, and positive-tip
amount summaries.

### 3. Deterministic uniform modeling sample

Sample the shared model-eligible population using a fixed probability and a
reproducible year-month seed. Group A and Group B must use the same sample.

The sample probability remains pending until a single-month and HPC resource
benchmark shows what is computationally practical. A candidate range is 0.2%-0.5%.

## Group-specific boundaries

### Group A

- Uses all rows in the shared model sample.
- Response: `tip_positive`.
- Method: logistic regression or equivalent Binomial GLM.

### Group B

- Starts from the same shared model sample.
- Applies `tips > 0` only after shared validation and sampling.
- Response: positive tip amount.
- Compares a justified positive-response model with a limited sensitivity model.
- May examine tip rate as a sensitivity measure, not as a replacement for the
  proposal response without full-group approval.

## Items still requiring evidence

- Verify all 48 remote files and their total size.
- Compare schema across 2022-2025.
- Audit all-month missingness and platform values.
- Report duplicate candidates without automatically deleting them.
- Benchmark the candidate sample probabilities.
- Confirm HPC storage and job resource requests.
- Freeze the final primary and sensitivity model specifications.

## Approval condition

Implementation of the authoritative shared cleaner begins only after:

- all group members have access to this document and the pilot audits;
- the decisions above are confirmed or revised;
- disagreements and revisions are recorded in the pull request;
- the accepted version is merged into `main`.
