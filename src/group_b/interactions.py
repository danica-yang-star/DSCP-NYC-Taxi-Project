"""Fit and interpret the two prespecified Group B interaction families."""

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

from src.group_b.main_effects import (
    BOROUGH_PREFIX,
    BOROUGHS,
    PLATFORM_TERM,
    main_effect_formula,
)
from src.group_b.model_candidates import (
    deterministic_cap,
    model_formula,
    prepare_frame,
)

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

INTERACTION_TERMS = (
    "lyft_x_bronx",
    "lyft_x_brooklyn",
    "lyft_x_queens",
    "lyft_x_staten_island",
    "lyft_x_year",
)


def interaction_formula() -> str:
    """Return the frozen M2 interaction formula."""
    return model_formula("log_tips")


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


def linear_contrast(result: Any, weights: dict[str, float]) -> dict[str, float]:
    """Calculate one cluster-robust linear contrast and transformed CI."""
    names = list(result.params.index)
    vector = np.zeros(len(names), dtype=float)
    for term, weight in weights.items():
        vector[names.index(term)] = weight
    estimate = float(vector @ np.asarray(result.params, dtype=float))
    covariance = np.asarray(result.cov_params(), dtype=float)
    standard_error = float(np.sqrt(vector @ covariance @ vector))
    low = estimate - 1.96 * standard_error
    high = estimate + 1.96 * standard_error
    return {
        "estimate_log_scale": estimate,
        "cluster_robust_se": standard_error,
        "ci_low_log_scale": low,
        "ci_high_log_scale": high,
        "percent_difference": 100.0 * math.expm1(estimate),
        "percent_ci_low": 100.0 * math.expm1(low),
        "percent_ci_high": 100.0 * math.expm1(high),
    }


def interaction_contrasts(result: Any) -> pd.DataFrame:
    """Return Lyft-versus-Uber contrasts for each borough and year."""
    rows: list[dict[str, Any]] = []
    borough_interaction = {
        "Manhattan": None,
        "Bronx": "lyft_x_bronx",
        "Brooklyn": "lyft_x_brooklyn",
        "Queens": "lyft_x_queens",
        "Staten Island": "lyft_x_staten_island",
    }
    for year in (2022, 2023, 2024, 2025):
        centered = year - 2022
        for borough in BOROUGHS:
            weights = {
                PLATFORM_TERM: 1.0,
                "lyft_x_year": float(centered),
            }
            interaction = borough_interaction[borough]
            if interaction is not None:
                weights[interaction] = 1.0
            contrast = linear_contrast(result, weights)
            rows.append(
                {
                    "year": year,
                    "borough": borough,
                    "comparison": "Lyft versus Uber",
                    **contrast,
                }
            )
    return pd.DataFrame(rows)


def standardized_predictions(
    result: Any, frame: pd.DataFrame, smearing_factor: float
) -> pd.DataFrame:
    """Calculate platform-by-borough-by-year standardized predictions."""
    fitted_log = np.asarray(result.predict(frame), dtype=float)
    params = result.params
    year_coefficient = float(params["year_centered"])
    platform_coefficient = float(params[PLATFORM_TERM])

    borough_coefficients = {"Manhattan": 0.0}
    interaction_coefficients = {"Manhattan": 0.0}
    for borough in BOROUGHS[1:]:
        borough_coefficients[borough] = float(params[f"{BOROUGH_PREFIX}{borough}]"])
        interaction_coefficients[borough] = float(
            params[f"lyft_x_{borough.lower().replace(' ', '_')}"]
        )

    current_borough = frame["borough_for_model"].astype("string")
    current_borough_main = current_borough.map(borough_coefficients).to_numpy(dtype=float)
    current_borough_interaction = current_borough.map(interaction_coefficients).to_numpy(
        dtype=float
    )
    current_year = frame["year_centered"].to_numpy(dtype=float)
    current_lyft = (frame["platform"].astype("string").to_numpy() == "Lyft").astype(float)
    current_platform_total = current_lyft * (
        platform_coefficient
        + float(params["lyft_x_year"]) * current_year
        + current_borough_interaction
    )
    base_log = (
        fitted_log - year_coefficient * current_year - current_borough_main - current_platform_total
    )

    rows: list[dict[str, Any]] = []
    for year in (2022, 2023, 2024, 2025):
        centered = year - 2022
        for borough in BOROUGHS:
            borough_main = borough_coefficients[borough]
            for platform in ("Uber", "Lyft"):
                platform_total = 0.0
                if platform == "Lyft":
                    platform_total = (
                        platform_coefficient
                        + float(params["lyft_x_year"]) * centered
                        + interaction_coefficients[borough]
                    )
                scenario_log = (
                    base_log + year_coefficient * centered + borough_main + platform_total
                )
                adjusted_mean = float(np.mean(np.exp(scenario_log) * smearing_factor))
                rows.append(
                    {
                        "year": year,
                        "borough": borough,
                        "platform": platform,
                        "adjusted_mean_tip": adjusted_mean,
                    }
                )
    return pd.DataFrame(rows)


def joint_wald_test(result: Any, terms: tuple[str, ...]) -> dict[str, float | int]:
    """Jointly test one frozen family of interaction coefficients."""
    names = list(result.params.index)
    restriction = np.zeros((len(terms), len(names)), dtype=float)
    for row, term in enumerate(terms):
        restriction[row, names.index(term)] = 1.0
    test = result.wald_test(restriction, scalar=True)
    return {
        "statistic": float(test.statistic),
        "degrees_of_freedom": len(terms),
        "p_value": float(test.pvalue),
    }


def create_figures(
    contrasts: pd.DataFrame,
    predictions: pd.DataFrame,
    figure_dir: Path,
    suffix: str,
) -> list[str]:
    """Plot interaction contrasts and standardized predictions."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    contrast_path = figure_dir / f"bq3_platform_contrasts_{suffix}.png"
    prediction_path = figure_dir / f"bq3_standardized_predictions_{suffix}.png"

    fig, axis = plt.subplots(figsize=(10, 5.5))
    for borough in BOROUGHS:
        subset = contrasts.loc[contrasts["borough"] == borough]
        years = subset["year"].to_numpy(dtype=int)
        differences = subset["percent_difference"].to_numpy(dtype=float)
        ci_low = subset["percent_ci_low"].to_numpy(dtype=float)
        ci_high = subset["percent_ci_high"].to_numpy(dtype=float)
        axis.plot(
            years,
            differences,
            marker="o",
            label=borough,
        )
        axis.fill_between(
            years,
            ci_low,
            ci_high,
            alpha=0.12,
        )
    axis.axhline(0, linestyle="--", color="gray")
    axis.set_xticks([2022, 2023, 2024, 2025])
    axis.set_xlabel("Pickup year")
    axis.set_ylabel("Adjusted Lyft-versus-Uber difference in positive tip (%)")
    axis.set_title("Adjusted Lyft-versus-Uber differences by borough and year")
    axis.legend(title="Pickup borough", ncol=2)
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(contrast_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True)
    for axis, borough in zip(axes.flat, BOROUGHS, strict=False):
        subset = predictions.loc[predictions["borough"] == borough]
        for platform, color in (("Uber", "#35618f"), ("Lyft", "#b45f06")):
            platform_data = subset.loc[subset["platform"] == platform]
            axis.plot(
                platform_data["year"],
                platform_data["adjusted_mean_tip"],
                marker="o",
                color=color,
                label=platform,
            )
        axis.set_title(borough)
        axis.set_xticks([2022, 2023, 2024, 2025])
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.015), ncol=2)
    fig.supxlabel("Pickup year", y=0.065)
    fig.supylabel("Adjusted mean positive tip ($)")
    fig.suptitle("Standardized platform predictions by borough and year")
    fig.tight_layout(rect=(0, 0.11, 1, 0.96))
    fig.savefig(prediction_path, dpi=160)
    plt.close(fig)
    return [contrast_path.name, prediction_path.name]


def run_interactions(
    sample_dir: Path,
    config_path: Path,
    report_output: Path,
    contrasts_output: Path,
    predictions_output: Path,
    figure_dir: Path,
    maximum_rows: int | None,
) -> dict[str, Any]:
    """Run B-Q3 and write auditable outputs."""
    import pyarrow.dataset as ds
    import statsmodels.formula.api as smf

    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_INTERACTION_FIT":
        raise ValueError("B-Q3 configuration was not frozen before fitting")

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

    fitted: dict[str, Any] = {}
    formulas = {
        "M1_primary_main_effects": main_effect_formula("M1_primary_main_effects"),
        "M2_prespecified_interactions": interaction_formula(),
    }
    for model_name, formula in formulas.items():
        fitted[model_name] = smf.ols(formula, data=frame).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": frame["year_month"],
                "use_correction": True,
            },
        )

    main_result = fitted["M1_primary_main_effects"]
    interaction_result = fitted["M2_prespecified_interactions"]
    smearing_factor = float(np.mean(np.exp(interaction_result.resid)))
    contrasts = interaction_contrasts(interaction_result)
    predictions = standardized_predictions(interaction_result, frame, smearing_factor)
    run_mode = "FULL" if maximum_rows is None else "PILOT"
    figures = create_figures(contrasts, predictions, figure_dir, run_mode.lower())

    models = {
        name: {"formula": formulas[name], **_model_metrics(result)}
        for name, result in fitted.items()
    }
    joint_tests = {
        "platform_by_borough": joint_wald_test(interaction_result, INTERACTION_TERMS[:4]),
        "platform_by_year": joint_wald_test(interaction_result, ("lyft_x_year",)),
    }
    report = {
        "status": "PASS",
        "analysis_id": config["analysis_id"],
        "analysis_status": config["status"],
        "run_mode": run_mode,
        "observational_boundary": (
            "Conditional platform heterogeneity among valid trips with recorded positive tips; no causal interpretation."
        ),
        "sample_directory_name": sample_dir.name,
        "group_b_rows_available": int(available_rows),
        "rows_used": int(len(frame)),
        "cluster_count": cluster_count,
        "smearing_factor": smearing_factor,
        "models": models,
        "incremental_fit": {
            "m2_minus_m1_r_squared": float(interaction_result.rsquared - main_result.rsquared),
            "m2_minus_m1_adjusted_r_squared": float(
                interaction_result.rsquared_adj - main_result.rsquared_adj
            ),
        },
        "joint_tests": joint_tests,
        "contrast_count": int(len(contrasts)),
        "prediction_count": int(len(predictions)),
        "contrasts_output": str(contrasts_output),
        "predictions_output": str(predictions_output),
        "figures": figures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    passed = (
        available_rows == 903434
        and len(frame) > 0
        and cluster_count == 48
        and len(contrasts) == 20
        and len(predictions) == 40
        and interaction_result.rsquared >= main_result.rsquared
        and math.isfinite(smearing_factor)
    )
    report["status"] = "PASS" if passed else "FAIL"

    report_output.parent.mkdir(parents=True, exist_ok=True)
    contrasts_output.parent.mkdir(parents=True, exist_ok=True)
    predictions_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    contrasts.to_csv(contrasts_output, index=False)
    predictions.to_csv(predictions_output, index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--contrasts-output", type=Path, required=True)
    parser.add_argument("--predictions-output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_interactions(
        sample_dir=args.sample_dir,
        config_path=args.config,
        report_output=args.report_output,
        contrasts_output=args.contrasts_output,
        predictions_output=args.predictions_output,
        figure_dir=args.figure_dir,
        maximum_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
