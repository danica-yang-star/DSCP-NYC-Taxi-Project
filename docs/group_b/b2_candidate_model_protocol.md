# B2 candidate-model protocol

## Purpose

B2 compares two prespecified positive-response models without declaring a final model automatically.

## Runs

1. Local pilot: deterministic capped samples from the 2022–2024 development period and 2025 validation period. This validates code, formula construction, outputs, and approximate resource needs.
2. HPC full run: the same program with no row caps, using every eligible Group B row in the shared 0.5% artifact.

The local pilot is not the final statistical result.

## Candidates

- Log-normal working model: OLS on `log(tips)` with Duan's smearing estimator for dollar-scale predictions.
- Gamma GLM: original-dollar `tips` with a log link.

Models use the same predictors and interactions. Candidate comparison is performed on 2025 dollar-scale predictions. A metric leader is reported, but the final choice requires review of calibration and residual diagnostics.

## Collinearity

B2 reports the Pearson correlation and two-predictor VIF for transformed fare and distance. The VIF is diagnostic evidence rather than an automatic deletion threshold.

## Outputs

- candidate comparison JSON;
- calibration-by-decile CSV;
- distribution and monthly-median figures;
- independent acceptance result.

No fitted binary model is committed during the pilot. Final coefficients and cluster-robust inference are deferred until the response family is selected and the full run passes.
