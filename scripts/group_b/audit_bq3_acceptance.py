"""Independently validate Group B B-Q3 interaction outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--contrasts", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=["PILOT", "FULL"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    contrasts = pd.read_csv(args.contrasts)
    predictions = pd.read_csv(args.predictions)
    errors: list[str] = []

    if report.get("status") != "PASS":
        errors.append("report status is not PASS")
    if report.get("run_mode") != args.expected_mode:
        errors.append("run mode mismatch")
    if report.get("analysis_status") != "FROZEN_BEFORE_INTERACTION_FIT":
        errors.append("analysis was not frozen before fitting")
    if report.get("group_b_rows_available") != 903434:
        errors.append("available Group B rows do not reconcile")
    if report.get("cluster_count") != 48:
        errors.append("cluster count is not 48")
    expected_rows = 903434 if args.expected_mode == "FULL" else 250000
    if report.get("rows_used") != expected_rows:
        errors.append("used row count does not match frozen run mode")

    models = report.get("models", {})
    expected_models = {
        "M1_primary_main_effects",
        "M2_prespecified_interactions",
    }
    if set(models) != expected_models:
        errors.append("nested interaction model set is incomplete")
    elif (
        models["M2_prespecified_interactions"]["r_squared"]
        < models["M1_primary_main_effects"]["r_squared"]
    ):
        errors.append("M2 R-squared is unexpectedly lower than M1")

    if len(contrasts) != 20 or report.get("contrast_count") != 20:
        errors.append("expected 20 borough-year contrasts")
    if len(predictions) != 40 or report.get("prediction_count") != 40:
        errors.append("expected 40 standardized predictions")
    if set(contrasts["year"]) != {2022, 2023, 2024, 2025}:
        errors.append("contrast year coverage is incomplete")
    expected_boroughs = {
        "Manhattan",
        "Bronx",
        "Brooklyn",
        "Queens",
        "Staten Island",
    }
    if set(contrasts["borough"]) != expected_boroughs:
        errors.append("contrast borough coverage is incomplete")
    if set(predictions["platform"]) != {"Uber", "Lyft"}:
        errors.append("prediction platform coverage is incomplete")

    contrast_numeric = contrasts[
        ["percent_difference", "percent_ci_low", "percent_ci_high"]
    ].to_numpy(dtype=float)
    prediction_numeric = predictions[["adjusted_mean_tip"]].to_numpy(dtype=float)
    if not np.isfinite(contrast_numeric).all():
        errors.append("contrasts contain nonfinite values")
    if not np.isfinite(prediction_numeric).all():
        errors.append("predictions contain nonfinite values")

    for family in ("platform_by_borough", "platform_by_year"):
        test = report.get("joint_tests", {}).get(family, {})
        if not all(key in test and math.isfinite(test[key]) for key in ("statistic", "p_value")):
            errors.append(f"joint test is missing or nonfinite: {family}")

    for figure_name in report.get("figures", []):
        figure = args.figure_dir / figure_name
        if not figure.is_file() or figure.stat().st_size == 0:
            errors.append(f"missing or empty figure: {figure_name}")

    if errors:
        print("GROUP B B-Q3 ACCEPTANCE: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"run_mode={report['run_mode']}")
    print(f"rows_used={report['rows_used']}")
    print(f"cluster_count={report['cluster_count']}")
    print(f"m2_minus_m1_r_squared={report['incremental_fit']['m2_minus_m1_r_squared']:.8f}")
    print(f"platform_by_borough_p={report['joint_tests']['platform_by_borough']['p_value']:.8g}")
    print(f"platform_by_year_p={report['joint_tests']['platform_by_year']['p_value']:.8g}")
    print("GROUP B B-Q3 ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
