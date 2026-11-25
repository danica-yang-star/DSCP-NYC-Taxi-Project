"""Audit the approved shared sample for Group B before model fitting."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = (
    "year",
    "month",
    "pickup_hour",
    "pickup_weekday",
    "pickup_borough_model",
    "platform",
    "base_passenger_fare",
    "trip_miles",
    "tips",
    "tip_positive",
    "valid_for_primary_model",
    "sample_weight",
)

NUMERIC_COLUMNS = (
    "base_passenger_fare",
    "trip_miles",
    "tips",
    "sample_weight",
)

QUANTILE_PROBABILITIES = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0)


def _json_number(value: Any) -> int | float | None:
    """Convert NumPy scalars to finite JSON-compatible numbers."""
    if pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    if number.is_integer():
        return int(number)
    return number


def _counts(series: pd.Series) -> dict[str, int]:
    values = series.astype("string").fillna("<MISSING>")
    return {str(key): int(value) for key, value in values.value_counts().sort_index().items()}


def _quantiles(series: pd.Series) -> dict[str, int | float | None]:
    result = series.quantile(QUANTILE_PROBABILITIES)
    return {
        str(probability): _json_number(result.loc[probability])
        for probability in QUANTILE_PROBABILITIES
    }


def _correlation(first: pd.Series, second: pd.Series, method: str) -> float | None:
    if method == "spearman":
        value = first.rank(method="average").corr(second.rank(method="average"))
    else:
        value = first.corr(second)
    converted = _json_number(value)
    return None if converted is None else float(converted)


def summarize_group_b_frame(frame: pd.DataFrame) -> dict[str, Any]:
    """Return deterministic Group B quality and distribution summaries."""
    missing_columns = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    data = frame.loc[:, REQUIRED_COLUMNS].copy()
    data["tip_rate"] = data["tips"] / data["base_passenger_fare"]
    data["log_tips"] = np.log(data["tips"])
    data["log_fare"] = np.log(data["base_passenger_fare"])
    data["log1p_miles"] = np.log1p(data["trip_miles"])

    missing_counts = {column: int(data[column].isna().sum()) for column in REQUIRED_COLUMNS}
    nonfinite_counts = {
        column: int((~np.isfinite(data[column].to_numpy(dtype=float, na_value=np.nan))).sum())
        for column in NUMERIC_COLUMNS
    }

    year_month = (
        data["year"].astype(int).astype(str)
        + "-"
        + data["month"].astype(int).astype(str).str.zfill(2)
    )
    observed_months = sorted(year_month.unique().tolist())

    return {
        "group_b_rows": int(len(data)),
        "observed_month_count": int(len(observed_months)),
        "first_observed_month": observed_months[0] if observed_months else None,
        "last_observed_month": observed_months[-1] if observed_months else None,
        "missing_counts": missing_counts,
        "nonfinite_counts": nonfinite_counts,
        "invalid_counts": {
            "tips_not_positive": int((data["tips"] <= 0).sum()),
            "fare_not_positive": int((data["base_passenger_fare"] <= 0).sum()),
            "miles_not_positive": int((data["trip_miles"] <= 0).sum()),
            "tip_positive_not_true": int((~data["tip_positive"].astype(bool)).sum()),
            "valid_for_primary_model_not_true": int(
                (~data["valid_for_primary_model"].astype(bool)).sum()
            ),
            "unsupported_platform": int((~data["platform"].isin(["Uber", "Lyft"])).sum()),
        },
        "tail_counts": {
            "tips_above_50": int((data["tips"] > 50).sum()),
            "tips_above_100": int((data["tips"] > 100).sum()),
            "tip_rate_above_1": int((data["tip_rate"] > 1).sum()),
            "tip_rate_above_2": int((data["tip_rate"] > 2).sum()),
        },
        "category_counts": {
            "platform": _counts(data["platform"]),
            "pickup_borough_model": _counts(data["pickup_borough_model"]),
            "pickup_weekday": _counts(data["pickup_weekday"]),
        },
        "sample_weight": {
            "minimum": _json_number(data["sample_weight"].min()),
            "maximum": _json_number(data["sample_weight"].max()),
            "unique_count": int(data["sample_weight"].nunique(dropna=False)),
        },
        "quantiles": {
            "tips": _quantiles(data["tips"]),
            "log_tips": _quantiles(data["log_tips"]),
            "base_passenger_fare": _quantiles(data["base_passenger_fare"]),
            "trip_miles": _quantiles(data["trip_miles"]),
            "tip_rate": _quantiles(data["tip_rate"]),
        },
        "correlations": {
            "fare_vs_miles_pearson": _correlation(
                data["base_passenger_fare"], data["trip_miles"], "pearson"
            ),
            "fare_vs_miles_spearman": _correlation(
                data["base_passenger_fare"], data["trip_miles"], "spearman"
            ),
            "log_fare_vs_log1p_miles_pearson": _correlation(
                data["log_fare"], data["log1p_miles"], "pearson"
            ),
            "tips_vs_fare_spearman": _correlation(
                data["tips"], data["base_passenger_fare"], "spearman"
            ),
            "tips_vs_miles_spearman": _correlation(data["tips"], data["trip_miles"], "spearman"),
        },
    }


def _monthly_counts(frame: pd.DataFrame) -> pd.DataFrame:
    monthly = (
        frame.groupby(["year", "month"], observed=True)
        .agg(
            group_b_rows=("tips", "size"),
            weighted_trip_estimate=("sample_weight", "sum"),
            mean_tip=("tips", "mean"),
            median_tip=("tips", "median"),
        )
        .reset_index()
        .sort_values(["year", "month"])
    )
    monthly["year"] = monthly["year"].astype(int)
    monthly["month"] = monthly["month"].astype(int)
    return monthly


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_audit(
    sample_dir: Path,
    config_path: Path,
    audit_output: Path,
    monthly_output: Path,
) -> dict[str, Any]:
    """Load the Arrow dataset, apply the frozen population, and write audit evidence."""
    import pyarrow.dataset as ds

    started = time.perf_counter()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("status") != "FROZEN_BEFORE_MODELING":
        raise ValueError("Analysis configuration must be frozen before the B1 audit")

    files = sorted(sample_dir.glob("*.parquet"))
    dataset = ds.dataset(sample_dir, format="parquet")
    missing_schema = sorted(set(REQUIRED_COLUMNS) - set(dataset.schema.names))
    if missing_schema:
        raise ValueError(f"Dataset schema is missing: {missing_schema}")

    population_filter = (
        (ds.field("valid_for_primary_model") == True)  # noqa: E712
        & (ds.field("tips") > 0)
    )
    table = dataset.to_table(columns=list(REQUIRED_COLUMNS), filter=population_filter)
    frame = table.to_pandas()
    summary = summarize_group_b_frame(frame)
    monthly = _monthly_counts(frame)

    all_missing_zero = all(value == 0 for value in summary["missing_counts"].values())
    all_nonfinite_zero = all(value == 0 for value in summary["nonfinite_counts"].values())
    all_invalid_zero = all(value == 0 for value in summary["invalid_counts"].values())
    monthly_complete = (
        len(monthly) == 48
        and (int(monthly.iloc[0]["year"]), int(monthly.iloc[0]["month"])) == (2022, 1)
        and (int(monthly.iloc[-1]["year"]), int(monthly.iloc[-1]["month"])) == (2025, 12)
    )
    passed = (
        len(files) == 48
        and summary["group_b_rows"] > 0
        and summary["observed_month_count"] == 48
        and all_missing_zero
        and all_nonfinite_zero
        and all_invalid_zero
        and monthly_complete
    )

    report: dict[str, Any] = {
        "status": "PASS" if passed else "FAIL",
        "analysis_id": config["analysis_id"],
        "analysis_status": config["status"],
        "observational_boundary": config["estimand_boundary"],
        "population_definition": config["population"],
        "sample_directory_name": sample_dir.name,
        "sample_file_count": len(files),
        "sample_file_bytes": int(sum(path.stat().st_size for path in files)),
        "shared_sample_rows": int(dataset.count_rows()),
        "primary_valid_sample_rows": int(
            dataset.count_rows(filter=ds.field("valid_for_primary_model") == True)  # noqa: E712
        ),
        "config_sha256": _sha256(config_path),
        "schema": {field.name: str(field.type) for field in dataset.schema},
        **summary,
        "monthly_output": str(monthly_output),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    audit_output.parent.mkdir(parents=True, exist_ok=True)
    monthly_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    monthly.to_csv(monthly_output, index=False)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--monthly-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(
        sample_dir=args.sample_dir,
        config_path=args.config,
        audit_output=args.audit_output,
        monthly_output=args.monthly_output,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
