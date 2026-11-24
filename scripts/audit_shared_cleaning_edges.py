from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

INPUT_PATH = Path("data/raw/fhvhv_tripdata_2024-01.parquet")
OUTPUT_PATH = Path("reports/audits/shared_cleaning_edge_counts_2024-01.json")
COLUMNS = ["tips", "base_passenger_fare", "trip_miles"]


def count_true(mask: pa.Array) -> int:
    value = pc.sum(pc.cast(pc.fill_null(mask, False), pa.int64())).as_py()
    return int(value or 0)


def main() -> None:
    counts = {
        "rows": 0,
        "positive_tip_rows": 0,
        "fare_negative_all": 0,
        "fare_zero_all": 0,
        "miles_negative_all": 0,
        "miles_zero_all": 0,
        "fare_negative_positive_tip": 0,
        "fare_zero_positive_tip": 0,
        "miles_negative_positive_tip": 0,
        "miles_zero_positive_tip": 0,
        "fare_or_miles_nonpositive_positive_tip": 0,
        "fare_and_miles_nonpositive_positive_tip": 0,
        "positive_tip_with_tip_rate_above_1": 0,
        "positive_tip_with_tip_rate_above_2": 0,
        "positive_tip_above_50": 0,
        "positive_tip_above_100": 0,
    }

    parquet = pq.ParquetFile(INPUT_PATH)
    for batch in parquet.iter_batches(batch_size=500_000, columns=COLUMNS):
        tips = batch.column(batch.schema.get_field_index("tips"))
        fare = batch.column(batch.schema.get_field_index("base_passenger_fare"))
        miles = batch.column(batch.schema.get_field_index("trip_miles"))

        positive_tip = pc.greater(tips, 0)
        fare_negative = pc.less(fare, 0)
        fare_zero = pc.equal(fare, 0)
        fare_nonpositive = pc.less_equal(fare, 0)
        fare_positive = pc.greater(fare, 0)
        miles_negative = pc.less(miles, 0)
        miles_zero = pc.equal(miles, 0)
        miles_nonpositive = pc.less_equal(miles, 0)

        counts["rows"] += batch.num_rows
        counts["positive_tip_rows"] += count_true(positive_tip)
        counts["fare_negative_all"] += count_true(fare_negative)
        counts["fare_zero_all"] += count_true(fare_zero)
        counts["miles_negative_all"] += count_true(miles_negative)
        counts["miles_zero_all"] += count_true(miles_zero)
        counts["fare_negative_positive_tip"] += count_true(pc.and_(positive_tip, fare_negative))
        counts["fare_zero_positive_tip"] += count_true(pc.and_(positive_tip, fare_zero))
        counts["miles_negative_positive_tip"] += count_true(pc.and_(positive_tip, miles_negative))
        counts["miles_zero_positive_tip"] += count_true(pc.and_(positive_tip, miles_zero))
        counts["fare_or_miles_nonpositive_positive_tip"] += count_true(
            pc.and_(positive_tip, pc.or_(fare_nonpositive, miles_nonpositive))
        )
        counts["fare_and_miles_nonpositive_positive_tip"] += count_true(
            pc.and_(positive_tip, pc.and_(fare_nonpositive, miles_nonpositive))
        )

        tip_rate = pc.divide(tips, fare)
        valid_tip_rate = pc.and_(positive_tip, fare_positive)
        counts["positive_tip_with_tip_rate_above_1"] += count_true(
            pc.and_(valid_tip_rate, pc.greater(tip_rate, 1))
        )
        counts["positive_tip_with_tip_rate_above_2"] += count_true(
            pc.and_(valid_tip_rate, pc.greater(tip_rate, 2))
        )
        counts["positive_tip_above_50"] += count_true(pc.and_(positive_tip, pc.greater(tips, 50)))
        counts["positive_tip_above_100"] += count_true(pc.and_(positive_tip, pc.greater(tips, 100)))

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(counts, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(counts, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
