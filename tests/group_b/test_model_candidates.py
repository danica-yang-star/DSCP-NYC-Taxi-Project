import numpy as np
import pandas as pd

from src.group_b.model_candidates import (
    deterministic_cap,
    model_formula,
    prediction_metrics,
    prepare_frame,
)


def sample_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2022, 2023, 2024, 2025],
            "month": [1, 2, 3, 4],
            "pickup_hour": [0, 8, 16, 23],
            "pickup_weekday": ["Monday", "Tuesday", "Wednesday", "Thursday"],
            "pickup_borough_model": [
                "Manhattan",
                "Bronx",
                "Brooklyn",
                "Other/Unknown",
            ],
            "platform": ["Uber", "Lyft", "Uber", "Lyft"],
            "base_passenger_fare": [10.0, 20.0, 30.0, 40.0],
            "trip_miles": [1.0, 2.0, 3.0, 4.0],
            "tips": [2.0, 4.0, 6.0, 8.0],
            "valid_for_primary_model": [True, True, True, True],
        }
    )


def test_prepare_frame_creates_frozen_terms() -> None:
    result = prepare_frame(sample_frame())

    assert np.allclose(result["year_centered"], [0, 1, 2, 3])
    assert result.loc[3, "borough_for_model"] == "Manhattan"
    assert result.loc[1, "lyft_x_bronx"] == 1
    assert result.loc[1, "lyft_x_year"] == 1
    assert np.isfinite(result["log_tips"]).all()


def test_deterministic_cap_is_reproducible() -> None:
    frame = pd.DataFrame({"value": range(100)})
    first = deterministic_cap(frame, 10, 20261125)
    second = deterministic_cap(frame, 10, 20261125)

    pd.testing.assert_frame_equal(first, second)


def test_prediction_metrics_perfect_predictions() -> None:
    actual = np.array([1.0, 2.0, 3.0])
    result = prediction_metrics(actual, actual)

    assert result["mae"] == 0
    assert result["rmse"] == 0
    assert result["mean_error_actual_minus_predicted"] == 0


def test_formula_contains_prespecified_terms() -> None:
    formula = model_formula("log_tips")

    assert formula.startswith("log_tips ~")
    assert "log_fare" in formula
    assert "log1p_miles" in formula
    assert "lyft_x_year" in formula
    assert "lyft_x_brooklyn" in formula
