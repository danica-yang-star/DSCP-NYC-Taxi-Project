import math

import numpy as np
import pandas as pd

from src.group_b.main_effects import PLATFORM_TERM
from src.group_b.sensitivity import (
    DISTANCE_ONLY_FORMULA,
    FARE_ONLY_FORMULA,
    GAMMA_FORMULA,
    PRIMARY_FORMULA,
    TIP_RATE_FORMULA,
    extract_core_effects,
    sensitivity_formulas,
    stability_summary,
)


def test_frozen_sensitivity_formulas() -> None:
    formulas = sensitivity_formulas()

    assert formulas["primary"] == PRIMARY_FORMULA
    assert "log1p_miles" not in FARE_ONLY_FORMULA
    assert "log_fare" not in DISTANCE_ONLY_FORMULA
    assert GAMMA_FORMULA.startswith("tips ~")
    assert TIP_RATE_FORMULA.startswith("log_tip_rate ~")
    assert "log_fare" not in TIP_RATE_FORMULA


class FakeResult:
    params = pd.Series(
        {
            "log_fare": 0.8,
            "log1p_miles": 0.1,
            "year_centered": -0.01,
            PLATFORM_TERM: math.log(1.1),
        }
    )

    def conf_int(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                0: self.params - 0.01,
                1: self.params + 0.01,
            }
        )


def test_core_effect_extraction() -> None:
    frame = pd.DataFrame({"trip_miles": [1.0, 2.0, 4.0]})
    effects = extract_core_effects(FakeResult(), frame)

    assert effects["fare_10_percent"] > 0
    assert effects["distance_plus_one_mile_percent"] > 0
    assert effects["one_year_percent"] < 0
    assert math.isclose(effects["lyft_vs_uber_percent"], 10.0)


def test_stability_summary_reports_direction() -> None:
    frame = pd.DataFrame(
        {
            "estimand": ["absolute_tip", "absolute_tip", "relative_tip_rate"],
            "fare_10_percent": [8.0, 8.2, np.nan],
            "distance_plus_one_mile_percent": [0.2, 0.3, 0.4],
            "one_year_percent": [-0.5, -0.7, 1.0],
            "lyft_vs_uber_percent": [12.0, 13.0, 5.0],
        }
    )
    summary = stability_summary(frame)

    assert summary["fare_10_percent"]["direction_stable"]
    assert summary["one_year_percent"]["direction_stable"]
    assert summary["fare_10_percent"]["minimum"] == 8.0
