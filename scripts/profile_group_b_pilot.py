from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

TRIP_PATH = Path("data/raw/fhvhv_tripdata_2024-01.parquet")
ZONE_PATH = Path("data/reference/taxi_zone_lookup.csv")
OUTPUT_PATH = Path("reports/audits/group_b_profile_2024-01.json")

COLUMNS = [
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

QUANTILES = [0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]


def quantile_summary(series: pd.Series) -> dict[str, float]:
    return {str(quantile): float(series.quantile(quantile)) for quantile in QUANTILES}


def main() -> None:
    started = time.perf_counter()
    if not TRIP_PATH.is_file():
        raise FileNotFoundError(TRIP_PATH)
    if not ZONE_PATH.is_file():
        raise FileNotFoundError(ZONE_PATH)

    table = pq.read_table(
        TRIP_PATH,
        columns=COLUMNS,
        filters=[("tips", ">", 0)],
    )
    frame = table.to_pandas()
    zones = pd.read_csv(ZONE_PATH, keep_default_na=False)

    required_nonmissing = frame[COLUMNS].notna().all(axis=1)
    finite_numeric = (
        np.isfinite(frame["tips"])
        & np.isfinite(frame["base_passenger_fare"])
        & np.isfinite(frame["trip_miles"])
    )
    positive_fare = frame["base_passenger_fare"] > 0
    positive_miles = frame["trip_miles"] > 0
    valid_location = frame["PULocationID"].between(1, 265, inclusive="both")
    valid_month = frame["pickup_datetime"].between(
        pd.Timestamp("2024-01-01"),
        pd.Timestamp("2024-02-01"),
        inclusive="left",
    )
    supported_platform = frame["hvfhs_license_num"].isin(PLATFORM_MAP)

    candidate_counts = {
        "positive_tip_rows": int(len(frame)),
        "rows_with_any_required_missing": int((~required_nonmissing).sum()),
        "rows_with_nonfinite_numeric": int((~finite_numeric).sum()),
        "rows_with_nonpositive_fare": int((~positive_fare).sum()),
        "rows_with_nonpositive_miles": int((~positive_miles).sum()),
        "rows_with_invalid_pickup_location": int((~valid_location).sum()),
        "rows_outside_expected_month": int((~valid_month).sum()),
        "rows_with_unsupported_platform": int((~supported_platform).sum()),
    }

    eligible_before_zone = (
        required_nonmissing
        & finite_numeric
        & positive_fare
        & positive_miles
        & valid_location
        & valid_month
        & supported_platform
    )

    analysis = frame.loc[eligible_before_zone].copy()
    zone_map = zones.set_index("LocationID")["Borough"]
    analysis["pickup_borough"] = analysis["PULocationID"].map(zone_map)
    unmatched_zone = analysis["pickup_borough"].isna()
    unmatched_zone_count = int(unmatched_zone.sum())
    analysis = analysis.loc[~unmatched_zone].copy()

    analysis["platform"] = analysis["hvfhs_license_num"].map(PLATFORM_MAP)
    analysis["pickup_hour"] = analysis["pickup_datetime"].dt.hour
    analysis["pickup_weekday"] = analysis["pickup_datetime"].dt.day_name()
    analysis["tip_rate"] = analysis["tips"] / analysis["base_passenger_fare"]

    profile = {
        "status": "PASS",
        "source_file": str(TRIP_PATH),
        "candidate_exclusion_counts_independent": candidate_counts,
        "eligible_rows_before_zone_join": int(eligible_before_zone.sum()),
        "unmatched_zone_rows": unmatched_zone_count,
        "analysis_eligible_rows": int(len(analysis)),
        "rows_removed_from_positive_tip_population": int(len(frame) - len(analysis)),
        "retention_rate_from_positive_tip_population": float(len(analysis) / len(frame)),
        "platform_counts": {
            str(key): int(value)
            for key, value in analysis["platform"].value_counts().sort_index().items()
        },
        "borough_counts": {
            str(key): int(value)
            for key, value in analysis["pickup_borough"].value_counts().sort_index().items()
        },
        "hour_counts": {
            str(key): int(value)
            for key, value in analysis["pickup_hour"].value_counts().sort_index().items()
        },
        "weekday_counts": {
            str(key): int(value)
            for key, value in analysis["pickup_weekday"].value_counts().sort_index().items()
        },
        "quantiles": {
            "tips": quantile_summary(analysis["tips"]),
            "base_passenger_fare": quantile_summary(analysis["base_passenger_fare"]),
            "trip_miles": quantile_summary(analysis["trip_miles"]),
            "tip_rate": quantile_summary(analysis["tip_rate"]),
        },
        "scan_seconds": round(time.perf_counter() - started, 3),
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(profile, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(profile, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
