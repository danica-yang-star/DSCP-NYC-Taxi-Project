import pandas as pd

from src.group_b.audit_sample import summarize_group_b_frame


def synthetic_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2022, 2022, 2023, 2023],
            "month": [1, 2, 1, 2],
            "pickup_hour": [8, 9, 17, 18],
            "pickup_weekday": ["Monday", "Tuesday", "Wednesday", "Thursday"],
            "pickup_borough_model": ["Manhattan", "Queens", "Brooklyn", "Bronx"],
            "platform": ["Uber", "Lyft", "Uber", "Lyft"],
            "base_passenger_fare": [10.0, 20.0, 30.0, 40.0],
            "trip_miles": [1.0, 2.0, 3.0, 4.0],
            "tips": [2.0, 4.0, 6.0, 8.0],
            "tip_positive": [True, True, True, True],
            "valid_for_primary_model": [True, True, True, True],
            "sample_weight": [200.0, 200.0, 200.0, 200.0],
        }
    )


def test_summary_counts_and_ranges() -> None:
    result = summarize_group_b_frame(synthetic_frame())

    assert result["group_b_rows"] == 4
    assert result["observed_month_count"] == 4
    assert result["missing_counts"]["tips"] == 0
    assert result["invalid_counts"]["tips_not_positive"] == 0
    assert result["sample_weight"]["unique_count"] == 1
    assert result["quantiles"]["tips"]["0.0"] == 2
    assert result["quantiles"]["tips"]["1.0"] == 8


def test_tail_counts_are_reported_not_deleted() -> None:
    frame = synthetic_frame()
    frame.loc[0, "tips"] = 101.0
    result = summarize_group_b_frame(frame)

    assert result["group_b_rows"] == 4
    assert result["tail_counts"]["tips_above_100"] == 1
