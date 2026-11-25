"""Fit Group B nested main-effects models with month-clustered inference."""

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

from src.group_b.model_candidates import deterministic_cap, prepare_frame

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

M0_TERMS = (
    "log_fare",
    "log1p_miles",
    "C(pickup_hour, Treatment(reference=0))",
    'C(pickup_weekday, Treatment(reference="Monday"))',
    "year_centered",
    "C(month, Treatment(reference=1))",
)

M1_TERMS = M0_TERMS + (
    'C(borough_for_model, Treatment(reference="Manhattan"))',
    'C(platform, Treatment(reference="Uber"))',
)

PLATFORM_TERM = 'C(platform, Treatment(reference="Uber"))[T.Lyft]'
BOROUGH_PREFIX = 'C(borough_for_model, Treatment(reference="Manhattan"))[T.'
BOROUGHS = ("Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island")


def main_effect_formula(model_name: str) -> str:
    """Return a frozen B-Q2 formula."""
    if model_name == "M0_trip_time":
        terms = M0_TERMS
    elif model_name == "M1_primary_main_effects":
        terms = M1_TERMS
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return "log_tips ~ " + " + ".join(terms)


def percent_change(coefficient: float) -> float:
    """Transform a log-scale coefficient to a percentage difference."""
    return 100.0 * math.expm1(coefficient)


def _coefficient_interval(result: Any, term: str) -> tuple[float, float, float]:
    estimate = float(result.params[term])
    names = list(result.params.index)
    index = names.index(term)
    interval = np.asarray(result.conf_int(), dtype=float)[index]
    return estimate, float(interval[0]), float(interval[1])


def _model_metrics(result: Any) -> dict[str, float | int]:
    return {
        "nobs": int(result.nobs),
        "parameter_count": int(len(result.params)),
        "r_squared": float(result.rsquared),
        "adjusted_r_squared": float(result.rsquared_adj),
        "aic": float(result.aic),
        "bic": float(result.bic),
        "residual_standard_deviation_log_scale": float(np.sqrt(result.ssr / result.df_resid)),
    }


def parameter_effect_table(result: Any) -> pd.DataFrame:
    """Create a robust coefficient and transformed-effect table."""
    intervals = np.asarray(result.conf_int(), dtype=float)
    rows: list[dict[str, Any]] = []
    for index, term in enumerate(result.params.index):
        estimate = float(result.params.iloc[index])
        low = float(intervals[index, 0])
        high = float(intervals[index, 1])
        if term == "Intercept":
            effect_kind = "intercept"
            transformed = transformed_low = transformed_high = np.nan
        elif term == "log_fare":
            effect_kind = "elasticity_fare"
            transformed, transformed_low, transformed_high = estimate, low, high
        elif term == "log1p_miles":
            effect_kind = "elasticity_log1p_miles"
            transformed, transformed_low, transformed_high = estimate, low, high
        else:
            effect_kind = "percent_difference"
            transformed = percent_change(estimate)
            transformed_low = percent_change(low)
            transformed_high = percent_change(high)
        rows.append(
            {
                "term": term,
                "effect_kind": effect_kind,
                "estimate_log_scale": estimate,
                "cluster_robust_se": float(result.bse.iloc[index]),
                "ci_low_log_scale": low,
                "ci_high_log_scale": high,
                "p_value": float(result.pvalues.iloc[index]),
                "reported_effect": transformed,
                "reported_ci_low": transformed_low,
                "reported_ci_high": transformed_high,
            }
        )
    return pd.DataFrame(rows)


def primary_effects(
    result: Any, frame: pd.DataFrame, smearing_factor: float
) -> dict[str, dict[str, float | str]]:
    """Calculate prespecified interpretable B-Q2 effects."""
    fare_estimate, fare_low, fare_high = _coefficient_interval(result, "log_fare")
    miles_estimate, miles_low, miles_high = _coefficient_interval(result, "log1p_miles")
    year_estimate, year_low, year_high = _coefficient_interval(result, "year_centered")
    platform_estimate, platform_low, platform_high = _coefficient_interval(result, PLATFORM_TERM)

    fare_log_change = math.log(1.1)
    fare_effect = {
        "label": "10% increase in base passenger fare",
        "percent_change": percent_change(fare_estimate * fare_log_change),
        "ci_low": percent_change(fare_low * fare_log_change),
        "ci_high": percent_change(fare_high * fare_log_change),
    }

    fitted_log = np.asarray(result.predict(frame), dtype=float)
    fitted_dollars = np.exp(fitted_log) * smearing_factor
    distance_delta = np.log1p(frame["trip_miles"].to_numpy() + 1.0) - np.log1p(
        frame["trip_miles"].to_numpy()
    )

    def distance_summary(coefficient: float) -> tuple[float, float]:
        counterfactual = fitted_dollars * np.exp(coefficient * distance_delta)
        dollar_change = float(np.mean(counterfactual - fitted_dollars))
        percentage = 100.0 * dollar_change / float(np.mean(fitted_dollars))
        return dollar_change, percentage

    distance_dollars, distance_percent = distance_summary(miles_estimate)
    distance_low_dollars, distance_low_percent = distance_summary(miles_low)
    distance_high_dollars, distance_high_percent = distance_summary(miles_high)

    return {
        "fare_10_percent": fare_effect,
        "distance_plus_one_mile": {
            "label": "add one trip mile",
            "average_dollar_change": distance_dollars,
            "average_dollar_ci_low": distance_low_dollars,
            "average_dollar_ci_high": distance_high_dollars,
            "average_percent_change": distance_percent,
            "average_percent_ci_low": distance_low_percent,
            "average_percent_ci_high": distance_high_percent,
        },
        "one_year_trend": {
            "label": "one-calendar-year increase",
            "percent_change": percent_change(year_estimate),
            "ci_low": percent_change(year_low),
            "ci_high": percent_change(year_high),
        },
        "lyft_vs_uber": {
            "label": "Lyft versus Uber",
            "percent_change": percent_change(platform_estimate),
            "ci_low": percent_change(platform_low),
            "ci_high": percent_change(platform_high),
        },
    }


def adjusted_predictions(result: Any, frame: pd.DataFrame, smearing_factor: float) -> pd.DataFrame:
    """Calculate empirical-standardized platform and borough predictions."""
    fitted_log = np.asarray(result.predict(frame), dtype=float)

    platform_coefficient = float(result.params[PLATFORM_TERM])
    platform_contribution = np.where(
        frame["platform"].astype("string").to_numpy() == "Lyft",
        platform_coefficient,
        0.0,
    )
    uber_log = fitted_log - platform_contribution
    uber_mean = float(np.mean(np.exp(uber_log) * smearing_factor))

    rows: list[dict[str, Any]] = [
        {
            "family": "platform",
            "level": "Uber",
            "reference": "Uber",
            "adjusted_mean_tip": uber_mean,
            "ratio_to_reference": 1.0,
        },
        {
            "family": "platform",
            "level": "Lyft",
            "reference": "Uber",
            "adjusted_mean_tip": uber_mean * math.exp(platform_coefficient),
            "ratio_to_reference": math.exp(platform_coefficient),
        },
    ]

    borough_coefficients = {"Manhattan": 0.0}
    for borough in BOROUGHS[1:]:
        borough_coefficients[borough] = float(result.params[f"{BOROUGH_PREFIX}{borough}]"])
    current_borough = frame["borough_for_model"].astype("string")
    current_contribution = current_borough.map(borough_coefficients).to_numpy(dtype=float)
    manhattan_log = fitted_log - current_contribution
    manhattan_mean = float(np.mean(np.exp(manhattan_log) * smearing_factor))
    for borough in BOROUGHS:
        coefficient = borough_coefficients[borough]
        rows.append(
            {
                "family": "borough",
                "level": borough,
                "reference": "Manhattan",
                "adjusted_mean_tip": manhattan_mean * math.exp(coefficient),
                "ratio_to_reference": math.exp(coefficient),
            }
        )
    return pd.DataFrame(rows)


def create_figures(
    effects: dict[str, dict[str, float | str]],
    predictions: pd.DataFrame,
    figure_dir: Path,
    suffix: str,
) -> list[str]:
    """Create primary-effect and adjusted-prediction figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    effect_path = figure_dir / f"bq2_primary_effects_{suffix}.png"
    prediction_path = figure_dir / f"bq2_adjusted_predictions_{suffix}.png"

    labels = ("Fare +10%", "Distance +1 mile", "Calendar year +1", "Lyft vs Uber")
    values = [
        float(effects["fare_10_percent"]["percent_change"]),
        float(effects["distance_plus_one_mile"]["average_percent_change"]),
        float(effects["one_year_trend"]["percent_change"]),
        float(effects["lyft_vs_uber"]["percent_change"]),
    ]
    lows = [
        float(effects["fare_10_percent"]["ci_low"]),
        float(effects["distance_plus_one_mile"]["average_percent_ci_low"]),
        float(effects["one_year_trend"]["ci_low"]),
        float(effects["lyft_vs_uber"]["ci_low"]),
    ]
    highs = [
        float(effects["fare_10_percent"]["ci_high"]),
        float(effects["distance_plus_one_mile"]["average_percent_ci_high"]),
        float(effects["one_year_trend"]["ci_high"]),
        float(effects["lyft_vs_uber"]["ci_high"]),
    ]
    positions = np.arange(len(labels))
    errors = np.vstack((np.array(values) - np.array(lows), np.array(highs) - np.array(values)))
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    axis.errorbar(values, positions, xerr=errors, fmt="o", capsize=4, color="#35618f")
    axis.axvline(0, linestyle="--", color="gray")
    axis.set_yticks(positions, labels)
    axis.set_xlabel("Associated percentage difference in conditional positive tip")
    axis.set_title("Adjusted Group B main effects with month-clustered 95% CIs")
    axis.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(effect_path, dpi=160)
    plt.close(fig)

    platform = predictions.loc[predictions["family"] == "platform"]
    borough = predictions.loc[predictions["family"] == "borough"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(platform["level"], platform["adjusted_mean_tip"], color="#35618f")
    axes[0].set_title("Platform-standardized positive tips")
    axes[0].set_ylabel("Adjusted mean positive tip ($)")
    axes[1].bar(borough["level"], borough["adjusted_mean_tip"], color="#b45f06")
    axes[1].tick_params(axis="x", rotation=30)
    axes[1].set_title("Borough-standardized positive tips")
    axes[1].set_ylabel("Adjusted mean positive tip ($)")
    fig.suptitle("Predictions standardize over the observed covariate distribution")
    fig.tight_layout()
    fig.savefig(prediction_path, dpi=160)
    plt.close(fig)
    return [effect_path.name, prediction_path.name]


def run_main_effects(
    sample_dir: Path,
    config_path: Path,
    decision_path: Path,
    report_output: Path,
    effects_output: Path,
    predictions_output: Path,
    figure_dir: Path,
    maximum_rows: int | None,
) -> dict[str, Any]:
    """Run B-Q2 and write auditable outputs."""
    import pyarrow.dataset as ds
    import statsmodels.formula.api as smf

    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_MAIN_EFFECTS_FIT":
        raise ValueError("B-Q2 configuration was not frozen before fitting")
    if decision.get("selected_primary_model") != "lognormal_ols_duan_smearing":
        raise ValueError("The frozen B2 decision does not select log-normal OLS")

    dataset = ds.dataset(sample_dir, format="parquet")
    population_filter = (
        (ds.field("valid_for_primary_model") == True)  # noqa: E712
        & (ds.field("tips") > 0)
    )
    frame = dataset.to_table(columns=list(SOURCE_COLUMNS), filter=population_filter).to_pandas()
    available_rows = len(frame)
    frame = prepare_frame(frame)
    frame = deterministic_cap(frame, maximum_rows, int(config["random_seed"]))
    frame["year_month"] = frame["year"].astype(int) * 100 + frame["month"].astype(int)
    cluster_count = int(frame["year_month"].nunique())

    results: dict[str, Any] = {}
    fitted: dict[str, Any] = {}
    for model_name in ("M0_trip_time", "M1_primary_main_effects"):
        formula = main_effect_formula(model_name)
        result = smf.ols(formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": frame["year_month"],
                "use_correction": True,
            },
        )
        fitted[model_name] = result
        results[model_name] = {
            "formula": formula,
            **_model_metrics(result),
        }

    primary = fitted["M1_primary_main_effects"]
    smearing_factor = float(np.mean(np.exp(primary.resid)))
    effect_table = parameter_effect_table(primary)
    effects = primary_effects(primary, frame, smearing_factor)
    predictions = adjusted_predictions(primary, frame, smearing_factor)
    run_mode = "FULL" if maximum_rows is None else "PILOT"
    suffix = run_mode.lower()
    figures = create_figures(effects, predictions, figure_dir, suffix)

    report = {
        "status": "PASS",
        "analysis_id": config["analysis_id"],
        "analysis_status": config["status"],
        "run_mode": run_mode,
        "observational_boundary": (
            "Conditional associations among valid trips with recorded positive "
            "tips; no causal interpretation."
        ),
        "sample_directory_name": sample_dir.name,
        "group_b_rows_available": int(available_rows),
        "rows_used": int(len(frame)),
        "cluster_variable": "year_month",
        "cluster_count": cluster_count,
        "smearing_factor": smearing_factor,
        "models": results,
        "incremental_fit": {
            "m1_minus_m0_r_squared": float(primary.rsquared - fitted["M0_trip_time"].rsquared),
            "m1_minus_m0_adjusted_r_squared": float(
                primary.rsquared_adj - fitted["M0_trip_time"].rsquared_adj
            ),
        },
        "primary_effects": effects,
        "effect_table_output": str(effects_output),
        "adjusted_predictions_output": str(predictions_output),
        "figures": figures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    passed = (
        available_rows == 903434
        and len(frame) > 0
        and cluster_count == 48
        and primary.rsquared >= fitted["M0_trip_time"].rsquared
        and math.isfinite(smearing_factor)
    )
    report["status"] = "PASS" if passed else "FAIL"

    report_output.parent.mkdir(parents=True, exist_ok=True)
    effects_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    effect_table.to_csv(effects_output, index=False)
    predictions.to_csv(predictions_output, index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--effects-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_main_effects(
        sample_dir=args.sample_dir,
        config_path=args.config,
        decision_path=args.decision,
        report_output=args.report_output,
        effects_output=args.effects_output,
        predictions_output=args.predictions_output,
        figure_dir=args.figure_dir,
        maximum_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
