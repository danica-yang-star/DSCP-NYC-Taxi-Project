import math

import numpy as np
import pandas as pd

from src.group_b.interactions import (
    interaction_contrasts,
    interaction_formula,
    linear_contrast,
    standardized_predictions,
)
from src.group_b.main_effects import BOROUGH_PREFIX, PLATFORM_TERM


class FakeResult:
    params = pd.Series(
        {
            PLATFORM_TERM: math.log(1.10),
            "lyft_x_bronx": math.log(0.95),
            "lyft_x_brooklyn": math.log(1.00),
            "lyft_x_queens": math.log(1.05),
            "lyft_x_staten_island": math.log(0.90),
            "lyft_x_year": math.log(1.02),
        }
    )

    def cov_params(self) -> pd.DataFrame:
        return pd.DataFrame(
            np.eye(len(self.params)) * 0.0001,
            index=self.params.index,
            columns=self.params.index,
        )


def test_interaction_formula_contains_only_frozen_interactions() -> None:
    formula = interaction_formula()

    assert "lyft_x_bronx" in formula
    assert "lyft_x_brooklyn" in formula
    assert "lyft_x_queens" in formula
    assert "lyft_x_staten_island" in formula
    assert "lyft_x_year" in formula


def test_linear_contrast_recovers_platform_effect() -> None:
    result = linear_contrast(FakeResult(), {PLATFORM_TERM: 1.0})

    assert math.isclose(result["percent_difference"], 10.0)
    assert result["cluster_robust_se"] > 0


def test_contrast_grid_is_complete() -> None:
    result = interaction_contrasts(FakeResult())

    assert len(result) == 20
    assert set(result["year"]) == {2022, 2023, 2024, 2025}
    assert set(result["borough"]) == {
        "Manhattan",
        "Bronx",
        "Brooklyn",
        "Queens",
        "Staten Island",
    }
    manhattan_2022 = result.loc[
        (result["borough"] == "Manhattan") & (result["year"] == 2022),
        "percent_difference",
    ].iloc[0]
    queens_2025 = result.loc[
        (result["borough"] == "Queens") & (result["year"] == 2025),
        "percent_difference",
    ].iloc[0]

    assert math.isclose(manhattan_2022, 10.0)
    assert queens_2025 > manhattan_2022


class FakePredictionResult:
    params = pd.Series(
        {
            "year_centered": math.log(1.01),
            PLATFORM_TERM: math.log(1.10),
            f"{BOROUGH_PREFIX}Bronx]": math.log(0.90),
            f"{BOROUGH_PREFIX}Brooklyn]": math.log(0.95),
            f"{BOROUGH_PREFIX}Queens]": math.log(1.05),
            f"{BOROUGH_PREFIX}Staten Island]": math.log(1.00),
            "lyft_x_bronx": math.log(0.98),
            "lyft_x_brooklyn": 0.0,
            "lyft_x_queens": math.log(1.02),
            "lyft_x_staten_island": math.log(0.97),
            "lyft_x_year": math.log(1.03),
        }
    )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        borough_main = {
            "Manhattan": 0.0,
            "Bronx": math.log(0.90),
            "Brooklyn": math.log(0.95),
            "Queens": math.log(1.05),
            "Staten Island": 0.0,
        }
        borough_interaction = {
            "Manhattan": 0.0,
            "Bronx": math.log(0.98),
            "Brooklyn": 0.0,
            "Queens": math.log(1.02),
            "Staten Island": math.log(0.97),
        }
        values = []
        for row in frame.itertuples(index=False):
            is_lyft = row.platform == "Lyft"
            log_value = (
                math.log(5.0)
                + self.params["year_centered"] * row.year_centered
                + borough_main[row.borough_for_model]
            )
            if is_lyft:
                log_value += (
                    self.params[PLATFORM_TERM]
                    + self.params["lyft_x_year"] * row.year_centered
                    + borough_interaction[row.borough_for_model]
                )
            values.append(log_value)
        return pd.Series(values)


def test_standardized_predictions_recover_scenarios() -> None:
    frame = pd.DataFrame(
        {
            "year_centered": [0, 3],
            "platform": ["Uber", "Lyft"],
            "borough_for_model": ["Manhattan", "Queens"],
        }
    )
    result = standardized_predictions(FakePredictionResult(), frame, 1.0)
    manhattan_uber_2022 = result.loc[
        (result["year"] == 2022)
        & (result["borough"] == "Manhattan")
        & (result["platform"] == "Uber"),
        "adjusted_mean_tip",
    ].iloc[0]
    queens_lyft_2025 = result.loc[
        (result["year"] == 2025) & (result["borough"] == "Queens") & (result["platform"] == "Lyft"),
        "adjusted_mean_tip",
    ].iloc[0]

    expected_queens_lyft_2025 = 5.0 * 1.01**3 * 1.05 * 1.10 * 1.03**3 * 1.02
    assert math.isclose(manhattan_uber_2022, 5.0)
    assert math.isclose(queens_lyft_2025, expected_queens_lyft_2025)
