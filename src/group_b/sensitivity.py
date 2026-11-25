"""Run the frozen Group B B-Q4 sensitivity and temporal-stability analyses."""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.group_b.main_effects import PLATFORM_TERM, main_effect_formula
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

PRIMARY_FORMULA = main_effect_formula("M1_primary_main_effects")
FARE_ONLY_FORMULA = PRIMARY_FORMULA.replace(" + log1p_miles", "", 1)
DISTANCE_ONLY_FORMULA = PRIMARY_FORMULA.replace("log_fare + ", "", 1)
GAMMA_FORMULA = PRIMARY_FORMULA.replace("log_tips ~", "tips ~", 1)
TIP_RATE_FORMULA = PRIMARY_FORMULA.replace("log_tips ~", "log_tip_rate ~", 1).replace(
    "log_fare + ", "", 1
)

ABSOLUTE_SCENARIOS = (
    "primary",
    "exclude_tips_above_p99_9",
    "exclude_tip_rate_above_1",
    "fare_only",
    "distance_only",
    "gamma_log_link",
)

EFFECT_FIELDS = (
    "fare_10_percent",
    "distance_plus_one_mile_percent",
    "one_year_percent",
    "lyft_vs_uber_percent",
)


def sensitivity_formulas() -> dict[str, str]:
    """Return the frozen formula assigned to every B-Q4 scenario."""
    return {
        "primary": PRIMARY_FORMULA,
        "exclude_tips_above_p99_9": PRIMARY_FORMULA,
        "exclude_tip_rate_above_1": PRIMARY_FORMULA,
        "fare_only": FARE_ONLY_FORMULA,
        "distance_only": DISTANCE_ONLY_FORMULA,
        "gamma_log_link": GAMMA_FORMULA,
        "log_tip_rate_without_fare_predictor": TIP_RATE_FORMULA,
    }


def _interval(result: Any, term: str) -> tuple[float, float, float]:
    names = list(result.params.index)
    index = names.index(term)
    estimate = float(result.params.iloc[index])
    interval = np.asarray(result.conf_int(), dtype=float)[index]
    return estimate, float(interval[0]), float(interval[1])


def _percent(coefficient: float) -> float:
    return 100.0 * math.expm1(coefficient)


def _distance_effect(coefficient: float, miles: np.ndarray) -> float:
    delta = np.log1p(miles + 1.0) - np.log1p(miles)
    return 100.0 * float(np.mean(np.expm1(coefficient * delta)))


def extract_core_effects(result: Any, frame: pd.DataFrame) -> dict[str, float | None]:
    """Extract common interpretable effects and confidence limits."""
    names = set(result.params.index)
    output: dict[str, float | None] = {}

    if "log_fare" in names:
        estimate, low, high = _interval(result, "log_fare")
        change = math.log(1.1)
        output.update(
            {
                "fare_10_percent": _percent(estimate * change),
                "fare_ci_low": _percent(low * change),
                "fare_ci_high": _percent(high * change),
            }
        )
    else:
        output.update(
            {
                "fare_10_percent": None,
                "fare_ci_low": None,
                "fare_ci_high": None,
            }
        )

    if "log1p_miles" in names:
        estimate, low, high = _interval(result, "log1p_miles")
        miles = frame["trip_miles"].to_numpy(dtype=float)
        output.update(
            {
                "distance_plus_one_mile_percent": _distance_effect(estimate, miles),
                "distance_ci_low": _distance_effect(low, miles),
                "distance_ci_high": _distance_effect(high, miles),
            }
        )
    else:
        output.update(
            {
                "distance_plus_one_mile_percent": None,
                "distance_ci_low": None,
                "distance_ci_high": None,
            }
        )

    for prefix, term in (
        ("one_year", "year_centered"),
        ("lyft_vs_uber", PLATFORM_TERM),
    ):
        estimate, low, high = _interval(result, term)
        output.update(
            {
                f"{prefix}_percent": _percent(estimate),
                f"{prefix}_ci_low": _percent(low),
                f"{prefix}_ci_high": _percent(high),
            }
        )
    return output


def _fit_ols(frame: pd.DataFrame, formula: str) -> Any:
    import statsmodels.formula.api as smf

    return smf.ols(formula, data=frame).fit(
        cov_type="cluster",
        cov_kwds={
            "groups": frame["year_month"],
            "use_correction": True,
        },
    )


def _fit_gamma(frame: pd.DataFrame, formula: str) -> Any:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf

    return smf.glm(
        formula,
        data=frame,
        family=sm.families.Gamma(link=sm.families.links.Log()),
    ).fit(
        maxiter=100,
        disp=0,
        cov_type="cluster",
        cov_kwds={
            "groups": frame["year_month"],
            "use_correction": True,
        },
    )


def scenario_row(
    scenario: str,
    estimand: str,
    model_family: str,
    formula: str,
    frame: pd.DataFrame,
    result: Any,
) -> dict[str, Any]:
    """Build one sensitivity result row."""
    return {
        "scenario": scenario,
        "estimand": estimand,
        "model_family": model_family,
        "formula": formula,
        "rows": len(frame),
        "cluster_count": int(frame["year_month"].nunique()),
        "parameter_count": len(result.params),
        **extract_core_effects(result, frame),
    }


def effect_range_summary(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize effect ranges without treating instability as execution failure."""
    summary: dict[str, Any] = {}
    for field in EFFECT_FIELDS:
        values = pd.to_numeric(results[field], errors="coerce").dropna()
        summary[field] = {
            "minimum": float(values.min()),
            "maximum": float(values.max()),
            "direction_stable": bool((values > 0).all() or (values < 0).all()),
            "scenario_count": len(values),
        }
    return summary


def stability_summary(results: pd.DataFrame) -> dict[str, Any]:
    """Summarize ranges across comparable absolute-tip scenarios."""
    comparable = results.loc[results["estimand"] == "absolute_tip"]
    return effect_range_summary(comparable)


def create_figures(
    results: pd.DataFrame,
    leave_one_year_out: pd.DataFrame,
    figure_dir: Path,
    suffix: str,
) -> list[str]:
    """Plot scenario and leave-one-year-out stability."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    scenario_path = figure_dir / f"bq4_sensitivity_summary_{suffix}.png"
    loo_path = figure_dir / f"bq4_leave_one_year_out_{suffix}.png"

    labels = {
        "fare_10_percent": "Fare +10%",
        "distance_plus_one_mile_percent": "Distance +1 mile",
        "one_year_percent": "Calendar year +1",
        "lyft_vs_uber_percent": "Lyft vs Uber",
    }
    comparable = results.loc[results["estimand"] == "absolute_tip"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, field in zip(axes.flat, EFFECT_FIELDS, strict=True):
        subset = comparable.loc[comparable[field].notna()]
        axis.scatter(subset[field], subset["scenario"], color="#35618f")
        primary = subset.loc[subset["scenario"] == "primary", field]
        if not primary.empty:
            axis.axvline(float(primary.iloc[0]), linestyle="--", color="gray")
        axis.set_title(labels[field])
        axis.set_xlabel("Associated percentage difference")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Primary Group B conclusions across frozen sensitivity scenarios")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(scenario_path, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    for axis, field in zip(axes.flat, EFFECT_FIELDS, strict=True):
        subset = leave_one_year_out.loc[leave_one_year_out[field].notna()]
        axis.plot(
            subset["omitted_year"].astype(str),
            subset[field],
            marker="o",
            color="#b45f06",
        )
        axis.set_title(labels[field])
        axis.set_xlabel("Omitted year")
        axis.set_ylabel("Associated percentage difference")
        axis.grid(alpha=0.25)
    fig.suptitle("Leave-one-year-out estimate stability")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(loo_path, dpi=160)
    plt.close(fig)
    return [scenario_path.name, loo_path.name]


def run_sensitivity(
    sample_dir: Path,
    config_path: Path,
    report_output: Path,
    scenarios_output: Path,
    leave_one_year_out_output: Path,
    figure_dir: Path,
    maximum_rows: int | None,
) -> dict[str, Any]:
    """Run all frozen B-Q4 scenarios and write evidence."""
    import pyarrow.dataset as ds

    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_SENSITIVITY_FITS":
        raise ValueError("B-Q4 configuration was not frozen before fitting")

    dataset = ds.dataset(sample_dir, format="parquet")
    population_filter = ds.field("valid_for_primary_model").isin([True]) & (ds.field("tips") > 0)
    full_frame = dataset.to_table(
        columns=list(SOURCE_COLUMNS), filter=population_filter
    ).to_pandas()
    available_rows = len(full_frame)
    full_frame = prepare_frame(full_frame)
    full_frame["tip_rate"] = full_frame["tips"] / full_frame["base_passenger_fare"]
    full_frame["log_tip_rate"] = np.log(full_frame["tip_rate"])
    full_frame["year_month"] = full_frame["year"].astype(int) * 100 + full_frame["month"].astype(
        int
    )
    tip_p999 = float(full_frame["tips"].quantile(0.999))
    frame = deterministic_cap(full_frame, maximum_rows, int(config["random_seed"]))

    formulas = sensitivity_formulas()
    scenario_frames = {
        "primary": frame,
        "exclude_tips_above_p99_9": frame.loc[frame["tips"] <= tip_p999].copy(),
        "exclude_tip_rate_above_1": frame.loc[frame["tip_rate"] <= 1.0].copy(),
        "fare_only": frame,
        "distance_only": frame,
        "gamma_log_link": frame,
        "log_tip_rate_without_fare_predictor": frame,
    }

    rows: list[dict[str, Any]] = []
    for scenario in config["sensitivity_scenarios"]:
        scenario_frame = scenario_frames[scenario]
        formula = formulas[scenario]
        if scenario == "gamma_log_link":
            result = _fit_gamma(scenario_frame, formula)
            model_family = "gamma_glm_log_link"
        else:
            result = _fit_ols(scenario_frame, formula)
            model_family = "ols_log_response"
        estimand = (
            "relative_tip_rate"
            if scenario == "log_tip_rate_without_fare_predictor"
            else "absolute_tip"
        )
        rows.append(
            scenario_row(
                scenario,
                estimand,
                model_family,
                formula,
                scenario_frame,
                result,
            )
        )
        del result
        gc.collect()
    scenario_results = pd.DataFrame(rows)

    loo_rows: list[dict[str, Any]] = []
    for omitted_year in (2022, 2023, 2024, 2025):
        loo_frame = frame.loc[frame["year"] != omitted_year].copy()
        result = _fit_ols(loo_frame, PRIMARY_FORMULA)
        loo_rows.append(
            {
                "omitted_year": omitted_year,
                "rows": len(loo_frame),
                "cluster_count": int(loo_frame["year_month"].nunique()),
                **extract_core_effects(result, loo_frame),
            }
        )
        del result
        gc.collect()
    loo_results = pd.DataFrame(loo_rows)

    run_mode = "FULL" if maximum_rows is None else "PILOT"
    figures = create_figures(scenario_results, loo_results, figure_dir, run_mode.lower())
    stability = stability_summary(scenario_results)
    temporal_stability = effect_range_summary(loo_results)
    common_scenario_values = scenario_results[
        ["one_year_percent", "lyft_vs_uber_percent"]
    ].to_numpy(dtype=float)
    leave_one_year_out_values = loo_results[list(EFFECT_FIELDS)].to_numpy(dtype=float)
    passed = (
        available_rows == 903434
        and len(scenario_results) == 7
        and len(loo_results) == 4
        and set(scenario_results["scenario"]) == set(config["sensitivity_scenarios"])
        and np.isfinite(common_scenario_values).all()
        and np.isfinite(leave_one_year_out_values).all()
        and (loo_results["cluster_count"] == 36).all()
    )
    report = {
        "status": "PASS" if passed else "FAIL",
        "analysis_id": config["analysis_id"],
        "analysis_status": config["status"],
        "run_mode": run_mode,
        "observational_boundary": (
            "Sensitivity checks for conditional associations; no causal interpretation."
        ),
        "group_b_rows_available": int(available_rows),
        "primary_rows_used": len(frame),
        "pooled_tip_p99_9": tip_p999,
        "full_population_tip_rate_above_1": int((full_frame["tip_rate"] > 1.0).sum()),
        "scenario_count": len(scenario_results),
        "leave_one_year_out_count": len(loo_results),
        "stability_summary": stability,
        "leave_one_year_out_stability": temporal_stability,
        "scenarios_output": str(scenarios_output),
        "leave_one_year_out_output": str(leave_one_year_out_output),
        "figures": figures,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    report_output.parent.mkdir(parents=True, exist_ok=True)
    scenarios_output.parent.mkdir(parents=True, exist_ok=True)
    leave_one_year_out_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    scenario_results.to_csv(scenarios_output, index=False)
    loo_results.to_csv(leave_one_year_out_output, index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--scenarios-output", type=Path, required=True)
    parser.add_argument("--leave-one-year-out-output", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--max-rows", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_sensitivity(
        sample_dir=args.sample_dir,
        config_path=args.config,
        report_output=args.report_output,
        scenarios_output=args.scenarios_output,
        leave_one_year_out_output=args.leave_one_year_out_output,
        figure_dir=args.figure_dir,
        maximum_rows=args.max_rows,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
