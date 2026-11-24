from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REQUIRED_COLUMNS = [
    "hvfhs_license_num",
    "pickup_datetime",
    "PULocationID",
    "base_passenger_fare",
    "trip_miles",
    "tips",
]

PLATFORM_MAP = {
    "HV0003": "Uber",
    "HV0005": "Lyft",
}

SPECIAL_BOROUGHS = {"EWR", "N/A", "Unknown", ""}

SAMPLE_COLUMNS = [
    "year",
    "month",
    "pickup_datetime",
    "pickup_hour",
    "pickup_weekday",
    "PULocationID",
    "pickup_borough_raw",
    "pickup_borough_model",
    "hvfhs_license_num",
    "platform",
    "base_passenger_fare",
    "trip_miles",
    "tips",
    "tip_positive",
    "fare_zero_flag",
    "miles_zero_flag",
    "valid_for_primary_model",
    "sample_weight",
]

GROUP_COLUMNS = [
    "year",
    "month",
    "platform",
    "pickup_borough_model",
    "pickup_weekday",
    "pickup_hour",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create shared monthly audits, summaries, and a deterministic sample."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--zone-lookup", type=Path, required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--month", type=int, required=True)
    parser.add_argument("--sample-fraction", type=float, default=0.005)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--sample-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--audit-output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=500_000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> None:
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.zone_lookup.is_file():
        raise FileNotFoundError(args.zone_lookup)
    if not 1 <= args.month <= 12:
        raise ValueError("month must be between 1 and 12")
    if not 0 < args.sample_fraction <= 1:
        raise ValueError("sample-fraction must be in (0, 1]")
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    for path in (args.sample_output, args.summary_output, args.audit_output):
        if path.exists() and not args.force:
            raise FileExistsError(f"{path} exists; use --force to replace it")


def load_zone_map(path: Path) -> dict[int, str]:
    zones = pd.read_csv(path, keep_default_na=False)
    expected = {"LocationID", "Borough", "Zone", "service_zone"}
    missing = expected - set(zones.columns)
    if missing:
        raise ValueError(f"Zone lookup missing columns: {sorted(missing)}")
    if zones["LocationID"].duplicated().any():
        raise ValueError("Zone lookup contains duplicate LocationID values")
    return {
        int(location_id): str(borough)
        for location_id, borough in zip(zones["LocationID"], zones["Borough"], strict=True)
    }


def model_borough(raw_borough: str) -> str:
    return "Other/Unknown" if raw_borough in SPECIAL_BOROUGHS else raw_borough


def initialize_counts() -> dict[str, int]:
    return {
        "raw_rows": 0,
        "missing_required": 0,
        "nonfinite_numeric": 0,
        "negative_tips": 0,
        "negative_fare": 0,
        "negative_miles": 0,
        "pickup_outside_expected_month": 0,
        "invalid_pickup_location": 0,
        "unsupported_platform": 0,
        "true_unmatched_zone": 0,
        "shared_valid_rows": 0,
        "shared_positive_tip_rows": 0,
        "shared_zero_fare_rows": 0,
        "shared_zero_miles_rows": 0,
        "primary_model_eligible_rows": 0,
        "sample_rows": 0,
    }


def reject_condition(
    remaining: pd.Series,
    condition: pd.Series,
    counts: dict[str, int],
    key: str,
) -> pd.Series:
    rejected = remaining & condition
    counts[key] += int(rejected.sum())
    return remaining & ~condition


def batch_summary(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["positive_tip_amount"] = np.where(frame["tip_positive"], frame["tips"], 0.0)
    grouped = (
        frame.groupby(GROUP_COLUMNS, observed=True, dropna=False)
        .agg(
            trip_count=("tips", "size"),
            positive_tip_count=("tip_positive", "sum"),
            positive_tip_amount_sum=("positive_tip_amount", "sum"),
            fare_sum=("base_passenger_fare", "sum"),
            miles_sum=("trip_miles", "sum"),
            zero_fare_count=("fare_zero_flag", "sum"),
            zero_miles_count=("miles_zero_flag", "sum"),
            primary_model_eligible_count=("valid_for_primary_model", "sum"),
        )
        .reset_index()
    )
    return grouped


def combine_summaries(frames: list[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(frames, ignore_index=True)
    value_columns = [
        "trip_count",
        "positive_tip_count",
        "positive_tip_amount_sum",
        "fare_sum",
        "miles_sum",
        "zero_fare_count",
        "zero_miles_count",
        "primary_model_eligible_count",
    ]
    return (
        combined.groupby(GROUP_COLUMNS, observed=True, dropna=False)[value_columns]
        .sum()
        .reset_index()
        .sort_values(GROUP_COLUMNS)
        .reset_index(drop=True)
    )


def clean_month(args: argparse.Namespace) -> dict[str, object]:
    validate_arguments(args)
    started = time.perf_counter()
    zone_map = load_zone_map(args.zone_lookup)
    parquet = pq.ParquetFile(args.input)
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in parquet.schema_arrow.names]
    if missing_columns:
        raise ValueError(f"Input missing required columns: {missing_columns}")

    args.sample_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_output.parent.mkdir(parents=True, exist_ok=True)

    sample_temp = args.sample_output.with_suffix(".tmp.parquet")
    summary_temp = args.summary_output.with_suffix(".tmp.csv")
    audit_temp = args.audit_output.with_suffix(".tmp.json")
    for path in (sample_temp, summary_temp, audit_temp):
        path.unlink(missing_ok=True)

    seed = args.seed if args.seed is not None else args.year * 100 + args.month
    random_generator = np.random.default_rng(seed)
    counts = initialize_counts()
    summary_frames: list[pd.DataFrame] = []
    sample_writer: pq.ParquetWriter | None = None

    month_start = pd.Timestamp(year=args.year, month=args.month, day=1)
    month_end = month_start + pd.DateOffset(months=1)

    try:
        for batch in parquet.iter_batches(
            batch_size=args.batch_size,
            columns=REQUIRED_COLUMNS,
        ):
            frame = batch.to_pandas()
            counts["raw_rows"] += len(frame)
            draws = random_generator.random(len(frame))

            missing_required = frame[REQUIRED_COLUMNS].isna().any(axis=1)
            finite_numeric = (
                np.isfinite(frame["tips"])
                & np.isfinite(frame["base_passenger_fare"])
                & np.isfinite(frame["trip_miles"])
            )
            negative_tips = frame["tips"] < 0
            negative_fare = frame["base_passenger_fare"] < 0
            negative_miles = frame["trip_miles"] < 0
            outside_month = ~frame["pickup_datetime"].between(
                month_start,
                month_end,
                inclusive="left",
            )
            invalid_location = ~frame["PULocationID"].between(1, 265, inclusive="both")
            unsupported_platform = ~frame["hvfhs_license_num"].isin(PLATFORM_MAP)

            remaining = pd.Series(True, index=frame.index)
            remaining = reject_condition(remaining, missing_required, counts, "missing_required")
            remaining = reject_condition(remaining, ~finite_numeric, counts, "nonfinite_numeric")
            remaining = reject_condition(remaining, negative_tips, counts, "negative_tips")
            remaining = reject_condition(remaining, negative_fare, counts, "negative_fare")
            remaining = reject_condition(remaining, negative_miles, counts, "negative_miles")
            remaining = reject_condition(
                remaining,
                outside_month,
                counts,
                "pickup_outside_expected_month",
            )
            remaining = reject_condition(
                remaining,
                invalid_location,
                counts,
                "invalid_pickup_location",
            )
            remaining = reject_condition(
                remaining,
                unsupported_platform,
                counts,
                "unsupported_platform",
            )

            shared = frame.loc[remaining].copy()
            shared["pickup_borough_raw"] = shared["PULocationID"].map(zone_map)
            unmatched_zone = shared["pickup_borough_raw"].isna()
            counts["true_unmatched_zone"] += int(unmatched_zone.sum())
            shared = shared.loc[~unmatched_zone].copy()

            shared["year"] = args.year
            shared["month"] = args.month
            shared["pickup_hour"] = shared["pickup_datetime"].dt.hour
            shared["pickup_weekday"] = shared["pickup_datetime"].dt.day_name()
            shared["pickup_borough_model"] = shared["pickup_borough_raw"].map(model_borough)
            shared["platform"] = shared["hvfhs_license_num"].map(PLATFORM_MAP)
            shared["tip_positive"] = shared["tips"] > 0
            shared["fare_zero_flag"] = shared["base_passenger_fare"] == 0
            shared["miles_zero_flag"] = shared["trip_miles"] == 0
            shared["valid_for_primary_model"] = (shared["base_passenger_fare"] > 0) & (
                shared["trip_miles"] > 0
            )

            counts["shared_valid_rows"] += len(shared)
            counts["shared_positive_tip_rows"] += int(shared["tip_positive"].sum())
            counts["shared_zero_fare_rows"] += int(shared["fare_zero_flag"].sum())
            counts["shared_zero_miles_rows"] += int(shared["miles_zero_flag"].sum())
            counts["primary_model_eligible_rows"] += int(shared["valid_for_primary_model"].sum())

            if not shared.empty:
                summary_frames.append(batch_summary(shared))

                selected_draws = draws[shared.index.to_numpy()]
                sampled = shared.loc[selected_draws < args.sample_fraction].copy()
                sampled["sample_weight"] = 1.0 / args.sample_fraction
                sampled = sampled[SAMPLE_COLUMNS]
                counts["sample_rows"] += len(sampled)

                if not sampled.empty:
                    sample_table = pa.Table.from_pandas(sampled, preserve_index=False)
                    if sample_writer is None:
                        sample_writer = pq.ParquetWriter(
                            sample_temp,
                            sample_table.schema,
                            compression="zstd",
                        )
                    sample_writer.write_table(sample_table)
    finally:
        if sample_writer is not None:
            sample_writer.close()

    if sample_writer is None:
        raise RuntimeError("No rows were selected for the modeling sample")
    if not summary_frames:
        raise RuntimeError("No shared-valid rows were available for summary output")

    summary = combine_summaries(summary_frames)
    summary.to_csv(summary_temp, index=False)

    rejected_total = sum(
        counts[key]
        for key in [
            "missing_required",
            "nonfinite_numeric",
            "negative_tips",
            "negative_fare",
            "negative_miles",
            "pickup_outside_expected_month",
            "invalid_pickup_location",
            "unsupported_platform",
            "true_unmatched_zone",
        ]
    )
    reconciled = counts["raw_rows"] == rejected_total + counts["shared_valid_rows"]

    audit: dict[str, object] = {
        "status": "PASS" if reconciled else "FAIL",
        "rules_status": "PROVISIONAL_PENDING_GROUP_APPROVAL",
        "input": str(args.input),
        "zone_lookup": str(args.zone_lookup),
        "year": args.year,
        "month": args.month,
        "sample_fraction": args.sample_fraction,
        "sample_seed": seed,
        "input_metadata_rows": parquet.metadata.num_rows,
        "counts": counts,
        "rejected_total": rejected_total,
        "row_reconciliation_pass": reconciled,
        "positive_tip_rate_shared": (
            counts["shared_positive_tip_rows"] / counts["shared_valid_rows"]
            if counts["shared_valid_rows"]
            else None
        ),
        "sample_output": str(args.sample_output),
        "sample_output_bytes": sample_temp.stat().st_size,
        "summary_output": str(args.summary_output),
        "summary_rows": len(summary),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }

    audit_temp.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    sample_temp.replace(args.sample_output)
    summary_temp.replace(args.summary_output)
    audit_temp.replace(args.audit_output)

    print(json.dumps(audit, indent=2, sort_keys=True))
    return audit


def main() -> None:
    clean_month(parse_args())


if __name__ == "__main__":
    main()
