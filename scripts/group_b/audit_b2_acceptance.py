"""Independently validate B2 candidate-model outputs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

EXPECTED_MODELS = {
    "lognormal_ols_duan_smearing",
    "gamma_glm_log_link",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--figure-dir", type=Path, required=True)
    parser.add_argument("--expected-mode", choices=["PILOT", "FULL"], required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.comparison.read_text(encoding="utf-8"))
    calibration = pd.read_csv(args.calibration)
    errors: list[str] = []

    if report.get("status") != "PASS":
        errors.append("comparison status is not PASS")
    if report.get("run_mode") != args.expected_mode:
        errors.append("run mode does not match the requested acceptance mode")
    if report.get("model_spec_status") != "FROZEN_BEFORE_CANDIDATE_FIT":
        errors.append("model specification was not frozen before fitting")
    if set(report.get("models", {})) != EXPECTED_MODELS:
        errors.append("candidate model set is incomplete")
    if report.get("candidate_selection_status") != "AWAITING_DIAGNOSTIC_REVIEW":
        errors.append("a final model was selected prematurely")
    if report.get("development_rows_used", 0) < 1000:
        errors.append("too few development rows")
    if report.get("validation_rows_used", 0) < 1000:
        errors.append("too few validation rows")
    if report.get("group_b_rows") != 903434:
        errors.append("Group B row count does not reconcile to B1")

    for model_name in EXPECTED_MODELS:
        metrics = report.get("models", {}).get(model_name, {}).get("metrics_2025", {})
        required = ("mae", "rmse", "mean_error_actual_minus_predicted")
        if not all(key in metrics and math.isfinite(metrics[key]) for key in required):
            errors.append(f"nonfinite or missing metrics for {model_name}")
        if metrics.get("mae", -1) < 0 or metrics.get("rmse", -1) < 0:
            errors.append(f"negative error metric for {model_name}")

    if set(calibration["model"].unique()) != EXPECTED_MODELS:
        errors.append("calibration model set is incomplete")
    if calibration.groupby("model")["prediction_decile"].nunique().min() < 8:
        errors.append("calibration has fewer than eight usable deciles")
    for figure_name in report.get("figures", []):
        figure = args.figure_dir / figure_name
        if not figure.is_file() or figure.stat().st_size == 0:
            errors.append(f"missing or empty figure: {figure_name}")

    if errors:
        print("GROUP B B2 ACCEPTANCE: FAIL")
        for error in errors:
            print(f"- {error}")
        raise SystemExit(1)

    print(f"run_mode={report['run_mode']}")
    print(f"group_b_rows={report['group_b_rows']}")
    print(f"development_rows_used={report['development_rows_used']}")
    print(f"validation_rows_used={report['validation_rows_used']}")
    print(f"metric_leader_by_rmse_only={report['metric_leader_by_rmse_only']}")
    print(f"two_predictor_vif={report['collinearity']['two_predictor_vif']:.4f}")
    print("GROUP B B2 ACCEPTANCE: PASS")


if __name__ == "__main__":
    main()
