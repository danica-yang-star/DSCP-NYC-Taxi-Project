"""Fit and compare the prespecified Group B positive-tip candidate models."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")


SOURCE_COLUMNS = (
    "year",
    "month",
    "pickup_hour",
    "pickup_weekday",
    "pickup_borough_model",
    "platform",
    "base_passenger_fare",
    "trip_miles",
    "tips",
    "valid_for_primary_model",
)

WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

BOROUGHS = ("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island")
PLATFORMS = ("Uber", "Lyft")


FORMULA_TERMS = (
    "log_fare",
    "log1p_miles",
    "C(pickup_hour, Treatment(reference=0))",
    'C(pickup_weekday, Treatment(reference="Monday"))',
    'C(borough_for_model, Treatment(reference="Manhattan"))',
    'C(platform, Treatment(reference="Uber"))',
    "year_centered",
    "C(month, Treatment(reference=1))",
    "lyft_x_year",
    "lyft_x_bronx",
    "lyft_x_brooklyn",
    "lyft_x_queens",
    "lyft_x_staten_island",
)


def model_formula(outcome: str) -> str:
    """Build the frozen common formula for a specified outcome."""
    return f"{outcome} ~ " + " + ".join(FORMULA_TERMS)


def prepare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic model fields without dropping eligible rows."""
    data = frame.copy()
    data["log_tips"] = np.log(data["tips"])
    data["log_fare"] = np.log(data["base_passenger_fare"])
    data["log1p_miles"] = np.log1p(data["trip_miles"])
    data["year_centered"] = data["year"].astype(int) - 2022

    known_borough = data["pickup_borough_model"].isin(BOROUGHS)
    data["borough_for_model"] = data["pickup_borough_model"].where(known_borough, "Manhattan")

    data["pickup_weekday"] = pd.Categorical(data["pickup_weekday"], categories=WEEKDAYS)
    data["borough_for_model"] = pd.Categorical(data["borough_for_model"], categories=BOROUGHS)
    data["platform"] = pd.Categorical(data["platform"], categories=PLATFORMS)

    is_lyft = (data["platform"] == "Lyft").astype(float)
    data["lyft_x_year"] = is_lyft * data["year_centered"]
    for borough in BOROUGHS[1:]:
        field = f"lyft_x_{borough.lower().replace(' ', '_')}"
        data[field] = is_lyft * (data["borough_for_model"] == borough).astype(float)

    return data


def deterministic_cap(frame: pd.DataFrame, maximum_rows: int | None, seed: int) -> pd.DataFrame:
    """Return all rows or a reproducible row-position sample."""
    if maximum_rows is None or len(frame) <= maximum_rows:
        return frame.reset_index(drop=True)
    return frame.sample(n=maximum_rows, random_state=seed).reset_index(drop=True)


def prediction_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate original-dollar validation metrics."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    residual = actual_values - predicted_values
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_error_actual_minus_predicted": float(np.mean(residual)),
        "median_absolute_error": float(np.median(np.abs(residual))),
        "actual_mean": float(np.mean(actual_values)),
        "predicted_mean": float(np.mean(predicted_values)),
    }


def calibration_table(actual: pd.Series, predicted: np.ndarray, model_name: str) -> pd.DataFrame:
    """Aggregate actual and predicted tips by predicted-value decile."""
    frame = pd.DataFrame(
        {
            "actual_tip": actual.to_numpy(dtype=float),
            "predicted_tip": np.asarray(predicted, dtype=float),
        }
    )
    frame["prediction_decile"] = (
        pd.qcut(frame["predicted_tip"], q=10, labels=False, duplicates="drop") + 1
    )
    result = (
        frame.groupby("prediction_decile", observed=True)
        .agg(
            rows=("actual_tip", "size"),
            actual_mean_tip=("actual_tip", "mean"),
            predicted_mean_tip=("predicted_tip", "mean"),
            actual_median_tip=("actual_tip", "median"),
        )
        .reset_index()
    )
    result.insert(0, "model", model_name)
    return result


def _finite_report(report: dict[str, float]) -> bool:
    return all(math.isfinite(value) for value in report.values())


def fit_candidates(
    development: pd.DataFrame, validation: pd.DataFrame
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit both candidates and return diagnostics and calibration."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    log_formula = model_formula("log_tips")
    gamma_formula = model_formula("tips")

    ols_result = smf.ols(log_formula, data=development).fit()
    smearing_factor = float(np.mean(np.exp(ols_result.resid)))
    ols_log_prediction = np.asarray(ols_result.predict(validation), dtype=float)
    ols_prediction = np.exp(ols_log_prediction) * smearing_factor

    gamma_result = smf.glm(
        gamma_formula,
        data=development,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit(maxiter=100, disp=0)
    gamma_prediction = np.asarray(gamma_result.predict(validation), dtype=float)

    actual = validation["tips"].to_numpy(dtype=float)
    ols_metrics = prediction_metrics(actual, ols_prediction)
    gamma_metrics = prediction_metrics(actual, gamma_prediction)
    if not _finite_report(ols_metrics) or not _finite_report(gamma_metrics):
        raise ValueError("Candidate metrics contain nonfinite values")

    transformed_correlation = float(development["log_fare"].corr(development["log1p_miles"]))
    two_predictor_vif = float(1.0 / (1.0 - transformed_correlation**2))

    metric_leader = min(
        ("lognormal_ols_duan_smearing", "gamma_glm_log_link"),
        key=lambda name: (ols_metrics if name == "lognormal_ols_duan_smearing" else gamma_metrics)[
            "rmse"
        ],
    )

    comparison: dict[str, Any] = {
        "candidate_selection_status": "AWAITING_DIAGNOSTIC_REVIEW",
        "metric_leader_by_rmse_only": metric_leader,
        "formulas": {
            "lognormal_ols_duan_smearing": log_formula,
            "gamma_glm_log_link": gamma_formula,
        },
        "collinearity": {
            "log_fare_vs_log1p_miles_pearson": transformed_correlation,
            "two_predictor_vif": two_predictor_vif,
            "ols_design_condition_number": float(ols_result.condition_number),
        },
        "models": {
            "lognormal_ols_duan_smearing": {
                "metrics_2025": ols_metrics,
                "smearing_factor": smearing_factor,
                "parameter_count": int(len(ols_result.params)),
                "log_residual_quantiles": {
                    str(probability): float(np.quantile(ols_result.resid, probability))
                    for probability in (0.01, 0.5, 0.99)
                },
            },
            "gamma_glm_log_link": {
                "metrics_2025": gamma_metrics,
                "converged": bool(gamma_result.converged),
                "iterations": int(gamma_result.fit_history.get("iteration", -1)),
                "parameter_count": int(len(gamma_result.params)),
                "pearson_chi2_per_residual_df": float(
                    gamma_result.pearson_chi2 / gamma_result.df_resid
                ),
            },
        },
    }

    calibration = pd.concat(
        [
            calibration_table(
                validation["tips"],
                ols_prediction,
                "lognormal_ols_duan_smearing",
            ),
            calibration_table(validation["tips"], gamma_prediction, "gamma_glm_log_link"),
        ],
        ignore_index=True,
    )
    return comparison, calibration


def create_figures(frame: pd.DataFrame, figure_dir: Path) -> list[str]:
    """Write deterministic descriptive figures and return their filenames."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_names: list[str] = []

    distribution_path = figure_dir / "b2_tip_distribution.png"
    cap = float(frame["tips"].quantile(0.999))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].hist(frame["tips"].clip(upper=cap), bins=60, color="#35618f")
    axes[0].set_title("Positive tips are strongly right-skewed")
    axes[0].set_xlabel("Tip amount, capped at pooled p99.9 ($)")
    axes[0].set_ylabel("Trips")
    axes[1].hist(np.log(frame["tips"]), bins=60, color="#b45f06")
    axes[1].set_title("Log transform reduces but does not remove right skew")
    axes[1].set_xlabel("log(tip amount)")
    axes[1].set_ylabel("Trips")
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=160)
    plt.close(fig)
    figure_names.append(distribution_path.name)

    monthly_path = figure_dir / "b2_monthly_median_tip.png"
    monthly = (
        frame.groupby(["year", "month"], observed=True)["tips"]
        .median()
        .rename("median_tip")
        .reset_index()
    )
    monthly["date"] = pd.to_datetime({"year": monthly["year"], "month": monthly["month"], "day": 1})
    fig, axis = plt.subplots(figsize=(10.5, 4.5))
    axis.plot(monthly["date"], monthly["median_tip"], marker="o", markersize=3)
    axis.set_title("Median positive tip varies across the 48 monthly partitions")
    axis.set_xlabel("Pickup month")
    axis.set_ylabel("Median positive tip ($)")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(monthly_path, dpi=160)
    plt.close(fig)
    figure_names.append(monthly_path.name)
    return figure_names


def create_calibration_figure(calibration: pd.DataFrame, figure_dir: Path) -> str:
    """Plot validation calibration for both candidate models."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    output = figure_dir / "b2_validation_calibration.png"
    display_names = {
        "lognormal_ols_duan_smearing": "Log-normal OLS + smearing",
        "gamma_glm_log_link": "Gamma GLM with log link",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for axis, model_name in zip(axes, display_names, strict=True):
        subset = calibration.loc[calibration["model"] == model_name].copy()
        lower = float(min(subset["predicted_mean_tip"].min(), subset["actual_mean_tip"].min()))
        upper = float(max(subset["predicted_mean_tip"].max(), subset["actual_mean_tip"].max()))
        axis.plot([lower, upper], [lower, upper], linestyle="--", color="gray")
        axis.plot(
            subset["predicted_mean_tip"],
            subset["actual_mean_tip"],
            marker="o",
            color="#35618f",
        )
        for row in subset.itertuples(index=False):
            axis.annotate(
                str(int(row.prediction_decile)),
                (row.predicted_mean_tip, row.actual_mean_tip),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
        axis.set_title(display_names[model_name])
        axis.set_xlabel("Mean predicted tip by validation decile ($)")
        axis.set_ylabel("Mean observed tip by validation decile ($)")
        axis.grid(alpha=0.25)
    fig.suptitle("Both candidates underpredict the highest-tip validation decile")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output.name


def run_candidates(
    sample_dir: Path,
    config_path: Path,
    comparison_output: Path,
    calibration_output: Path,
    figure_dir: Path,
    maximum_development_rows: int | None,
    maximum_validation_rows: int | None,
) -> dict[str, Any]:
    """Load data, execute the candidate comparison, and write evidence."""
    import pyarrow.dataset as ds

    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_CANDIDATE_FIT":
        raise ValueError("Model specification was not frozen before fitting")

    dataset = ds.dataset(sample_dir, format="parquet")
    missing = sorted(set(SOURCE_COLUMNS) - set(dataset.schema.names))
    if missing:
        raise ValueError(f"Shared sample is missing columns: {missing}")

    population_filter = (
        (ds.field("valid_for_primary_model") == True)  # noqa: E712
        & (ds.field("tips") > 0)
    )
    frame = dataset.to_table(columns=list(SOURCE_COLUMNS), filter=population_filter).to_pandas()
    frame = prepare_frame(frame)

    development_full = frame.loc[frame["year"].isin(config["development_years"])]
    validation_full = frame.loc[frame["year"].isin(config["validation_years"])]
    seed = int(config["random_seed"])
    development = deterministic_cap(development_full, maximum_development_rows, seed)
    validation = deterministic_cap(validation_full, maximum_validation_rows, seed + 1)

    comparison, calibration = fit_candidates(development, validation)
    figure_names = create_figures(frame, figure_dir)
    figure_names.append(create_calibration_figure(calibration, figure_dir))
    run_mode = (
        "FULL" if maximum_development_rows is None and maximum_validation_rows is None else "PILOT"
    )
    passed = (
        len(development) > 0
        and len(validation) > 0
        and comparison["models"]["gamma_glm_log_link"]["converged"]
        and len(calibration) >= 10
    )

    report: dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "model_spec_id": config["model_spec_id"],
        "model_spec_status": config["status"],
        "run_mode": run_mode,
        "sample_directory_name": sample_dir.name,
        "group_b_rows": int(len(frame)),
        "development_rows_available": int(len(development_full)),
        "validation_rows_available": int(len(validation_full)),
        "development_rows_used": int(len(development)),
        "validation_rows_used": int(len(validation)),
        "maximum_development_rows": maximum_development_rows,
        "maximum_validation_rows": maximum_validation_rows,
        "figures": figure_names,
        **comparison,
        "calibration_output": str(calibration_output),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    comparison_output.parent.mkdir(parents=True, exist_ok=True)
    calibration_output.parent.mkdir(parents=True, exist_ok=True)
    comparison_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    calibration.to_csv(calibration_output, index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--comparison-output", type=Path, required=True)
    parser.add_argument("--calibration-output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--max-development-rows", type=int)
    parser.add_argument("--max-validation-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_candidates(
        sample_dir=args.sample_dir,
        config_path=args.config,
        comparison_output=args.comparison_output,
        calibration_output=args.calibration_output,
        figure_dir=args.figure_dir,
        maximum_development_rows=args.max_development_rows,
        maximum_validation_rows=args.max_validation_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
