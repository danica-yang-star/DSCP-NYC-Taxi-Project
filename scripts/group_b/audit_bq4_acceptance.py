"""Independently validate Group B B-Q4 sensitivity outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SCENARIOS = {
    "primary",
    "exclude_tips_above_p99_9",
    "exclude_tip_rate_above_1",
    "fare_only",
    "distance_only",
    "gamma_log_link",
    "log_tip_rate_without_fare_predictor",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--bq2-report", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path, required=True)
    parser.add_argument("--leave-one-year-out", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=["PILOT", "FULL"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    bq2_report = json.loads(args.bq2_report.read_text(encoding="utf-8"))
    scenarios = pd.read_csv(args.scenarios)
    loo = pd.read_csv(args.leave_one_year_out)
    errors: list[str] = []

    if report.get("status") != "PASS":
        errors.append("report status is not PASS")
    if report.get("run_mode") != args.expected_mode:
        errors.append("run mode mismatch")
    if report.get("analysis_status") != "FROZEN_BEFORE_SENSITIVITY_FITS":
        errors.append("analysis was not frozen before fitting")
    if report.get("group_b_rows_available") != 903434:
        errors.append("available Group B rows do not reconcile")
    expected_primary_rows = 903434 if args.expected_mode == "FULL" else 120000
    if report.get("primary_rows_used") != expected_primary_rows:
        errors.append("primary rows do not match frozen run mode")
    if set(scenarios["scenario"]) != EXPECTED_SCENARIOS:
        errors.append("sensitivity scenario set is incomplete")
    if args.expected_mode == "FULL":
        primary = scenarios.loc[scenarios["scenario"] == "primary"]
        if len(primary) != 1:
            errors.append("primary sensitivity row is missing or duplicated")
        else:
            expected = {
                "fare_10_percent": bq2_report["primary_effects"]["fare_10_percent"][
                    "percent_change"
                ],
                "distance_plus_one_mile_percent": bq2_report["primary_effects"][
                    "distance_plus_one_mile"
                ]["average_percent_change"],
                "one_year_percent": bq2_report["primary_effects"]["one_year_trend"][
                    "percent_change"
                ],
                "lyft_vs_uber_percent": bq2_report["primary_effects"]["lyft_vs_uber"][
                    "percent_change"
                ],
            }
            for field, expected_value in expected.items():
                observed = float(primary.iloc[0][field])
                if not np.isclose(observed, expected_value, rtol=0.0, atol=1e-10):
                    errors.append(
                        f"B-Q2/B-Q4 primary mismatch for {field}: "
                        f"{observed} versus {expected_value}"
                    )
    if len(loo) != 4 or set(loo["omitted_year"]) != {2022, 2023, 2024, 2025}:
        errors.append("leave-one-year-out set is incomplete")
    if not np.isfinite(report.get("pooled_tip_p99_9", np.nan)):
        errors.append("pooled p99.9 threshold is missing")
    if report.get("full_population_tip_rate_above_1", 0) <= 0:
        errors.append("tip-rate tail count is missing")

    stability = report.get("stability_summary", {})
    expected_effects = {
        "fare_10_percent",
        "distance_plus_one_mile_percent",
        "one_year_percent",
        "lyft_vs_uber_percent",
    }
    if set(stability) != expected_effects:
        errors.append("stability summary is incomplete")
    elif not all(isinstance(item.get("direction_stable"), bool) for item in stability.values()):
        errors.append("scenario stability flags are missing")

    temporal_stability = report.get("leave_one_year_out_stability", {})
    if set(temporal_stability) != expected_effects:
        errors.append("leave-one-year-out stability summary is incomplete")
    elif not all(
        isinstance(item.get("direction_stable"), bool) for item in temporal_stability.values()
    ):
        errors.append("leave-one-year-out stability flags are missing")

    core_numeric = scenarios[["one_year_percent", "lyft_vs_uber_percent"]].to_numpy(dtype=float)
    if not np.isfinite(core_numeric).all():
        errors.append("common scenario effects contain nonfinite values")
    loo_numeric = loo[
        [
            "fare_10_percent",
            "distance_plus_one_mile_percent",
            "one_year_percent",
            "lyft_vs_uber_percent",
        ]
    ].to_numpy(dtype=float)
    if not np.isfinite(loo_numeric).all():
        errors.append("leave-one-year-out effects contain nonfinite values")
    if not ((loo["cluster_count"] == 36).all()):
        errors.append("leave-one-year-out cluster counts are not 36")

    for figure_name in report.get("figures", []):
        figure = args.figure_dir / figure_name
        if not figure.is_file() or figure.stat().st_size == 0:
            errors.append(f"missing or empty figure: {figure_name}")

    if errors:
        print("GROUP B B-Q4 ACCEPTANCE: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"run_mode={report['run_mode']}")
    print(f"primary_rows_used={report['primary_rows_used']}")
    print(f"pooled_tip_p99_9={report['pooled_tip_p99_9']}")
    print(f"full_population_tip_rate_above_1={report['full_population_tip_rate_above_1']}")
    for effect, item in stability.items():
        print(
            f"scenario_{effect}_range={item['minimum']:.6f} to "
            f"{item['maximum']:.6f}; direction_stable={item['direction_stable']}"
        )
    for effect, item in temporal_stability.items():
        print(
            f"leave_one_year_out_{effect}_range={item['minimum']:.6f} to "
            f"{item['maximum']:.6f}; direction_stable={item['direction_stable']}"
        )
    if args.expected_mode == "FULL":
        print("bq2_bq4_primary_consistency=PASS")
    else:
        print("bq2_bq4_primary_consistency=NOT_APPLICABLE_FOR_PILOT")
    print("GROUP B B-Q4 ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
