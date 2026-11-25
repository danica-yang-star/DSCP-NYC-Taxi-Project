"""Independently validate Group B B-Q2 outputs."""

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
    parser.add_argument("--effects", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=["PILOT", "FULL"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    effects = pd.read_csv(args.effects)
    predictions = pd.read_csv(args.predictions)
    errors: list[str] = []

    if report.get("status") != "PASS":
        errors.append("report status is not PASS")
    if report.get("run_mode") != args.expected_mode:
        errors.append("run mode mismatch")
    if report.get("analysis_status") != "FROZEN_BEFORE_MAIN_EFFECTS_FIT":
        errors.append("analysis was not frozen before fitting")
    if report.get("group_b_rows_available") != 903434:
        errors.append("available Group B rows do not reconcile to B1/B2")
    if report.get("cluster_count") != 48:
        errors.append("cluster count is not 48")
    if args.expected_mode == "FULL" and report.get("rows_used") != 903434:
        errors.append("full run did not use all Group B rows")
    if args.expected_mode == "PILOT" and report.get("rows_used") != 200000:
        errors.append("pilot did not use the frozen 200,000 rows")

    models = report.get("models", {})
    if set(models) != {"M0_trip_time", "M1_primary_main_effects"}:
        errors.append("nested model set is incomplete")
    elif models["M1_primary_main_effects"]["r_squared"] < models["M0_trip_time"]["r_squared"]:
        errors.append("M1 R-squared is unexpectedly lower than M0")

    required_terms = {
        "log_fare",
        "log1p_miles",
        "year_centered",
        'C(platform, Treatment(reference="Uber"))[T.Lyft]',
        'C(borough_for_model, Treatment(reference="Manhattan"))[T.Bronx]',
        'C(borough_for_model, Treatment(reference="Manhattan"))[T.Brooklyn]',
        'C(borough_for_model, Treatment(reference="Manhattan"))[T.Queens]',
        'C(borough_for_model, Treatment(reference="Manhattan"))[T.Staten Island]',
    }
    if not required_terms.issubset(set(effects["term"])):
        errors.append("effect table is missing prespecified terms")
    numeric_effects = effects.loc[effects["term"].isin(required_terms), "estimate_log_scale"]
    if not all(math.isfinite(value) for value in numeric_effects):
        errors.append("primary coefficients contain nonfinite values")

    platform_levels = set(predictions.loc[predictions["family"] == "platform", "level"])
    borough_levels = set(predictions.loc[predictions["family"] == "borough", "level"])
    if platform_levels != {"Uber", "Lyft"}:
        errors.append("platform standardized predictions are incomplete")
    if borough_levels != {"Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"}:
        errors.append("borough standardized predictions are incomplete")
    if not np_all_finite(predictions["adjusted_mean_tip"]):
        errors.append("adjusted predictions contain nonfinite values")

    for figure_name in report.get("figures", []):
        figure = args.figure_dir / figure_name
        if not figure.is_file() or figure.stat().st_size == 0:
            errors.append(f"missing or empty figure: {figure_name}")

    if errors:
        print("GROUP B B-Q2 ACCEPTANCE: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"run_mode={report['run_mode']}")
    print(f"rows_used={report['rows_used']}")
    print(f"cluster_count={report['cluster_count']}")
    print(f"m0_r_squared={models['M0_trip_time']['r_squared']:.6f}")
    print(f"m1_r_squared={models['M1_primary_main_effects']['r_squared']:.6f}")
    print(f"smearing_factor={report['smearing_factor']:.6f}")
    print("GROUP B B-Q2 ACCEPTANCE: PASS")


def np_all_finite(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return bool(np.isfinite(values).all())


if __name__ == "__main__":
    main()
