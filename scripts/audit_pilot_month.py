from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

TRIP_PATH = Path("data/raw/fhvhv_tripdata_2024-01.parquet")
ZONE_PATH = Path("data/reference/taxi_zone_lookup.csv")
OUTPUT_PATH = Path("reports/audits/pilot_2024-01.json")

REQUIRED_COLUMNS = [
    "hvfhs_license_num",
    "pickup_datetime",
    "PULocationID",
    "base_passenger_fare",
    "trip_miles",
    "tips",
]


def count_true(mask: pa.Array) -> int:
    cleaned = pc.fill_null(mask, False)
    value = pc.sum(pc.cast(cleaned, pa.int64())).as_py()
    return int(value or 0)


def update_range(current: list[object | None], array: pa.Array) -> None:
    minimum = pc.min(array).as_py()
    maximum = pc.max(array).as_py()
    if minimum is not None and (current[0] is None or minimum < current[0]):
        current[0] = minimum
    if maximum is not None and (current[1] is None or maximum > current[1]):
        current[1] = maximum


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not TRIP_PATH.is_file():
        raise FileNotFoundError(TRIP_PATH)
    if not ZONE_PATH.is_file():
        raise FileNotFoundError(ZONE_PATH)

    parquet = pq.ParquetFile(TRIP_PATH)
    schema = parquet.schema_arrow
    missing = [name for name in REQUIRED_COLUMNS if name not in schema.names]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    summary: dict[str, object] = {
        "status": "RUNNING",
        "source_file": str(TRIP_PATH),
        "file_bytes": TRIP_PATH.stat().st_size,
        "sha256": sha256_file(TRIP_PATH),
        "metadata_rows": parquet.metadata.num_rows,
        "row_groups": parquet.metadata.num_row_groups,
        "schema": {field.name: str(field.type) for field in schema},
        "required_columns": REQUIRED_COLUMNS,
        "missing_required_columns": missing,
        "null_counts": {name: 0 for name in REQUIRED_COLUMNS},
    }

    counts = {
        "rows_scanned": 0,
        "tips_positive": 0,
        "tips_zero": 0,
        "tips_negative": 0,
        "fare_nonpositive": 0,
        "miles_nonpositive": 0,
        "pickup_location_outside_1_265": 0,
        "pickup_outside_2024_01": 0,
    }
    ranges: dict[str, list[object | None]] = {
        "tips": [None, None],
        "base_passenger_fare": [None, None],
        "trip_miles": [None, None],
        "pickup_datetime": [None, None],
    }
    license_values: set[str] = set()
    started = time.perf_counter()

    for batch in parquet.iter_batches(batch_size=500_000, columns=REQUIRED_COLUMNS):
        columns = {
            name: batch.column(batch.schema.get_field_index(name)) for name in REQUIRED_COLUMNS
        }
        counts["rows_scanned"] += batch.num_rows
        for name, array in columns.items():
            summary["null_counts"][name] += array.null_count

        tips = columns["tips"]
        fare = columns["base_passenger_fare"]
        miles = columns["trip_miles"]
        pickup = columns["pickup_datetime"]
        pickup_location = columns["PULocationID"]
        license_number = columns["hvfhs_license_num"]

        counts["tips_positive"] += count_true(pc.greater(tips, 0))
        counts["tips_zero"] += count_true(pc.equal(tips, 0))
        counts["tips_negative"] += count_true(pc.less(tips, 0))
        counts["fare_nonpositive"] += count_true(pc.less_equal(fare, 0))
        counts["miles_nonpositive"] += count_true(pc.less_equal(miles, 0))
        counts["pickup_location_outside_1_265"] += count_true(
            pc.or_(pc.less(pickup_location, 1), pc.greater(pickup_location, 265))
        )
        counts["pickup_outside_2024_01"] += count_true(
            pc.or_(
                pc.less(pickup, datetime(2024, 1, 1)),
                pc.greater_equal(pickup, datetime(2024, 2, 1)),
            )
        )

        update_range(ranges["tips"], tips)
        update_range(ranges["base_passenger_fare"], fare)
        update_range(ranges["trip_miles"], miles)
        update_range(ranges["pickup_datetime"], pickup)

        license_values.update(
            str(value) for value in pc.unique(license_number).to_pylist() if value is not None
        )

    zones = pd.read_csv(ZONE_PATH)
    expected_zone_columns = {"LocationID", "Borough", "Zone", "service_zone"}
    missing_zone_columns = sorted(expected_zone_columns - set(zones.columns))

    summary["audit_counts"] = counts
    summary["ranges"] = ranges
    summary["hvfhs_license_values"] = sorted(license_values)
    summary["positive_tip_rate_among_all_rows"] = (
        counts["tips_positive"] / counts["rows_scanned"] if counts["rows_scanned"] else None
    )
    summary["zone_lookup"] = {
        "rows": len(zones),
        "columns": zones.columns.tolist(),
        "missing_expected_columns": missing_zone_columns,
        "duplicate_location_ids": int(zones["LocationID"].duplicated().sum()),
        "minimum_location_id": int(zones["LocationID"].min()),
        "maximum_location_id": int(zones["LocationID"].max()),
    }
    summary["scan_seconds"] = round(time.perf_counter() - started, 3)
    summary["status"] = (
        "PASS"
        if counts["rows_scanned"] == parquet.metadata.num_rows and not missing_zone_columns
        else "FAIL"
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(summary, indent=2, default=str, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    display = {
        "status": summary["status"],
        "file_bytes": summary["file_bytes"],
        "metadata_rows": summary["metadata_rows"],
        "row_groups": summary["row_groups"],
        "audit_counts": counts,
        "ranges": ranges,
        "hvfhs_license_values": summary["hvfhs_license_values"],
        "positive_tip_rate_among_all_rows": summary["positive_tip_rate_among_all_rows"],
        "zone_lookup": summary["zone_lookup"],
        "scan_seconds": summary["scan_seconds"],
        "audit_file": str(OUTPUT_PATH),
    }
    print(json.dumps(display, indent=2, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
