# Group B handoff: positive tip amount analysis

## 1. Scope and current status

Group B studies the following observational question:

> Among valid NYC HVFHV trips with a recorded positive tip, what trip, time,
> location, and platform characteristics are associated with the tip amount?

This is a conditional association analysis. It does not estimate causal effects.

Youming Li completed the shared 48-month cleaning pipeline and the frozen Group B
modeling work through sensitivity analysis. The remaining work is diagnostics,
technical reporting, and integration with Group A. Do not add more candidate models
unless the group first records a specific, justified amendment.

Authoritative Group B population and run:

- source period: 2022-01 through 2025-12;
- source inputs: 48 monthly NYC TLC HVFHV Parquet files, 21.414 GiB total;
- raw trips audited: 927,966,235;
- deterministic shared 0.5% sample: 4,639,013 rows;
- primary-valid sample rows: 4,635,942;
- Group B positive-tip rows: 903,434;
- final B-Q4 Slurm job: `114280`;
- jobs `114278` and `114279` are superseded and must not be cited.

The shared cleaning rules were subsequently accepted by the group. Historical v1
audit files retain `PROVISIONAL_PENDING_GROUP_APPROVAL` because the immutable
artifacts were not rewritten after approval.

## 2. Code and model access

Repository:

```text
https://github.com/danica-yang-star/DSCP-NYC-Taxi-Project
```

Current Group B branch:

```text
youming/group-b-positive-tip-modeling
```

For an existing clone:

```bash
git fetch origin
git switch youming/group-b-positive-tip-modeling || git switch --track origin/youming/group-b-positive-tip-modeling
git pull --ff-only origin youming/group-b-positive-tip-modeling
```

Do not commit directly to Youming's branch. Create a personal branch from the
reviewed Group B branch:

```bash
git switch -c YOUR_GITHUB_USERNAME/group-b-diagnostics-report
```

Create the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-group-b.txt
python -m pytest -q tests/group_b
```

The repository intentionally does not contain serialized model objects. The frozen
JSON configurations, Python model code, committed result tables, figures, audits,
and Slurm scripts reproduce the analysis more transparently than a version-sensitive
pickle file.

## 3. Shared data access

### Statistics HPC

The read-only v1 package is available at:

```text
/workspace/yli2955/dscp_nyc_taxi_shared/v1
```

Validate it before use:

```bash
SHARED_V1=/workspace/yli2955/dscp_nyc_taxi_shared/v1
cd "$SHARED_V1"
sha256sum -c checksums.sha256
find samples -type f -name 'model_sample_*.parquet' | wc -l
find summaries -type f -name 'monthly_summary_*.csv' | wc -l
find audits -type f -name 'cleaning_audit_*.json' | wc -l
```

Expected counts are 48 sample files, 48 summaries, and 48 audits.

Group B filter:

```python
import pyarrow.dataset as ds

dataset = ds.dataset(
    "/workspace/yli2955/dscp_nyc_taxi_shared/v1/samples",
    format="parquet",
)
group_b = dataset.to_table(
    filter=(
        ds.field("valid_for_primary_model").isin([True])
        & (ds.field("tips") > 0)
    )
)
```

The sample was selected at a constant 0.5% rate, so `sample_weight` is 200 for
every retained row. The Group B models use the full 903,434-row positive-tip sample.

### Drive fallback

The portable archive is named:

```text
nyc_hvfhv_shared_v1_2022_2025.tar.gz
```

Its size is approximately 77 MiB compressed. The Google Drive URL should be shared
privately with the group rather than committed to this public repository. After
download, validate it with the accompanying file:

```bash
shasum -a 256 -c nyc_hvfhv_shared_v1_2022_2025.tar.gz.sha256
tar -xzf nyc_hvfhv_shared_v1_2022_2025.tar.gz
cd v1
shasum -a 256 -c checksums.sha256
```

Do not commit raw TLC files, the shared archive, extracted sample Parquet files,
credentials, or local absolute paths to Git.

## 4. Completed analysis and authoritative conclusions

### B1: population and data audit

- Reconciled all 48 monthly partitions.
- Confirmed 903,434 Group B rows across all 48 year-month clusters.
- Audited tails, boroughs, platforms, schema, row counts, and sample weights.

### B2: response-family decision

The frozen 2022-2024 development / 2025 validation comparison evaluated:

- log-normal OLS with Duan smearing;
- Gamma GLM with log link.

Log-normal OLS was retained as the primary working model because it had lower 2025
RMSE and MAE and better mean calibration. Gamma remained a prespecified sensitivity
model. Both candidates underpredict the highest validation decile.

### B-Q2: main effects

The primary model is log-normal OLS with month-clustered inference. It includes
fare, distance, pickup hour, weekday, year, calendar month, pickup borough, and
platform.

Key full-sample conditional associations:

- 10% higher base fare: 8.243% higher mean positive tip;
- adding one mile: 0.276% higher standardized mean positive tip;
- Lyft versus Uber: 13.391% higher mean positive tip;
- one calendar year: 0.677% lower mean positive tip.

These are associations among positive-tip trips, not causal effects.

### B-Q3: prespecified interactions

Only `platform x borough` and `platform x year` were evaluated. Both joint tests
were statistically detectable, but the incremental R-squared was only about
0.000118. The interaction model is therefore descriptive heterogeneity analysis,
not a replacement for the primary main-effects model.

The adjusted Lyft-versus-Uber difference was positive in every borough and year and
generally increased over calendar time.

### B-Q4: sensitivity and temporal stability

- Removing tips above pooled p99.9 produced almost unchanged results.
- Removing `tip_rate > 1` produced almost unchanged results.
- Gamma retained the main directions but changed some magnitudes.
- Four leave-one-year-out fits retained the direction of all four core effects.
- Dropping fare inflated the distance effect to about 12.0% and reversed the year
  trend. This is evidence that fare cannot be omitted when interpreting distance.
- The log tip-rate analysis answers a different relative-tipping question and must
  not be described as a replacement primary outcome.

The final B-Q4 quality gate verifies that the primary B-Q4 effects reproduce the
frozen B-Q2 full-run effects within numerical tolerance.

## 5. File map

| Location | Purpose |
|---|---|
| `configs/group_b/` | Frozen analysis plans and model specifications |
| `docs/group_b/` | Analysis plans, amendments, and B2/B-Q2/B-Q3/B-Q4 protocols |
| `src/group_b/audit_sample.py` | B1 sample and population audit |
| `src/group_b/model_candidates.py` | B2 candidate-family comparison |
| `src/group_b/main_effects.py` | B-Q2 nested main-effects analysis |
| `src/group_b/interactions.py` | B-Q3 prespecified interactions |
| `src/group_b/sensitivity.py` | B-Q4 sensitivity and leave-one-year-out analysis |
| `scripts/group_b/` | Independent acceptance scripts |
| `slurm/group_b_*.sbatch` | Reproducible full HPC runs |
| `reports/group_b/audits/` | B1 audit evidence |
| `reports/group_b/b2/` | Candidate comparison and frozen model decision |
| `reports/group_b/bq2/` | Main-effects tables and HPC evidence |
| `reports/group_b/bq3/` | Interaction contrasts and HPC evidence |
| `reports/group_b/bq4/` | Sensitivity, stability, incident, and HPC evidence |
| `reports/group_b/figures/` | Pilot and full-run figures |
| `tests/group_b/` | Deterministic unit and acceptance-support tests |

Authoritative outputs use `_full` in their names. Files containing `_pilot` are
development checks and should not be quoted as final results.

## 6. Remaining work for teammates

### Workstream A: final diagnostics

Use the frozen B-Q2 primary model. Do not select another model from diagnostic
results.

Required diagnostics:

1. residual-versus-fitted plot on the log scale;
2. Q-Q plot and a concise assessment of tail departure;
3. scale-location or binned residual-spread assessment;
4. influence review appropriate for a very large sample;
5. calibration summary on the original dollar scale, explicitly noting the highest
   validation-decile underprediction;
6. confirmation that inference remains clustered by the 48 year-month partitions.

For plots that cannot reasonably display 903,434 points, use a documented,
deterministic plotting sample. Do not refit the primary model on a visually selected
subset.

Deliverables:

- diagnostic script under `src/group_b/`;
- independent audit under `scripts/group_b/`;
- tests under `tests/group_b/`;
- diagnostic tables under `reports/group_b/diagnostics/`;
- final figures under `reports/group_b/figures/diagnostics/`;
- a short diagnostic conclusion that distinguishes model limitations from execution
  failures.

### Workstream B: Group B technical narrative

Write a concise, client-readable section with this order:

1. population: positive-tip trips only;
2. reason for the log-normal primary model;
3. primary adjusted associations with 95% month-clustered confidence intervals;
4. platform and borough standardized predictions;
5. prespecified platform heterogeneity;
6. sensitivity and temporal-stability findings;
7. limitations and observational boundary.

Do not use causal language such as “caused,” “increased tips,” or “platform effect.”
Use “was associated with,” “adjusted difference,” and “conditional positive tip.”

### Workstream C: combine Group A and Group B

The final two-part explanation must keep the estimands separate:

- Group A: probability that a valid trip has `tips > 0`;
- Group B: tip amount conditional on `tips > 0`.

Do not multiply the two stages or present an unconditional expected tip unless the
group first specifies and validates that additional estimand. Compare directions and
patterns narratively, and make clear that the two models use different populations.

### Workstream D: reproducibility and final report integration

- Add method and results text to the group report.
- Select final figures; keep detailed diagnostics in an appendix if space is limited.
- Report Slurm resource use and the successful 48-month array design.
- Explain that raw monthly files were downloaded into worker-local temporary storage
  and removed after each successful job.
- Preserve the incident record: jobs `114278` and `114279` are superseded by final
  job `114280`.
- Update the root README only after the branch is reviewed and merged.

## 7. Required quality gate before merge

Run from the repository root:

```bash
source .venv/bin/activate
python -m ruff check src/group_b scripts/group_b tests/group_b
python -m ruff format --check src/group_b scripts/group_b tests/group_b
python -m pytest -q tests/group_b
bash -n slurm/group_b_candidates.sbatch
bash -n slurm/group_b_bq2_main_effects.sbatch
bash -n slurm/group_b_bq3_interactions.sbatch
bash -n slurm/group_b_bq4_sensitivity.sbatch
git diff --check
git status --short --branch
```

Current baseline: 16 Group B tests pass.

Merge readiness requires:

- all tests and independent audits pass;
- final result text matches committed `_full` evidence;
- no raw or extracted data are staged;
- no local absolute paths, credentials, or personal Drive URLs are committed;
- no frozen configuration is changed without a dated amendment;
- Git worktree is clean and the contributor branch is synchronized with its remote;
- another group member reviews the changes before merge.

## 8. Attribution

Youming Li completed the shared monthly cleaning implementation, 48-task Slurm array
validation, immutable shared-data package, Group B analysis plan and sample audit,
candidate-family comparison, primary main-effects analysis, prespecified interaction
analysis, sensitivity and leave-one-year-out analysis, testing, HPC execution, and
resource/provenance records through B-Q4.

Subsequent contributors should list their own diagnostics, writing, figure, and
integration contributions in the final report and commit history. Do not describe
group-authored narrative or Group A work as Youming's individual contribution.
