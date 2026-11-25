import math

import pandas as pd

from src.group_b.main_effects import (
    adjusted_predictions,
    main_effect_formula,
    percent_change,
)


def test_nested_formulas_have_expected_scope() -> None:
    m0 = main_effect_formula("M0_trip_time")
    m1 = main_effect_formula("M1_primary_main_effects")

    assert "log_fare" in m0
    assert "log1p_miles" in m0
    assert "borough_for_model" not in m0
    assert "platform" not in m0
    assert "borough_for_model" in m1
    assert "platform" in m1
    assert "lyft_x_year" not in m1


def test_percent_change_transform() -> None:
    assert math.isclose(percent_change(0.0), 0.0)
    assert math.isclose(percent_change(math.log(1.1)), 10.0)


class FakeResult:
    params = pd.Series(
        {
            'C(platform, Treatment(reference="Uber"))[T.Lyft]': math.log(1.1),
            'C(borough_for_model, Treatment(reference="Manhattan"))[T.Bronx]': math.log(0.9),
            'C(borough_for_model, Treatment(reference="Manhattan"))[T.Brooklyn]': math.log(1.0),
            'C(borough_for_model, Treatment(reference="Manhattan"))[T.Queens]': math.log(1.2),
            'C(borough_for_model, Treatment(reference="Manhattan"))[T.Staten Island]': math.log(
                0.8
            ),
        }
    )

    def predict(self, frame: pd.DataFrame) -> pd.Series:
        borough_effect = {
            "Manhattan": 0.0,
            "Bronx": math.log(0.9),
            "Brooklyn": 0.0,
            "Queens": math.log(1.2),
            "Staten Island": math.log(0.8),
        }
        platform = frame["platform"].astype("string").map({"Uber": 0.0, "Lyft": math.log(1.1)})
        borough = frame["borough_for_model"].astype("string").map(borough_effect)
        return pd.Series(math.log(5.0) + platform + borough)


def test_adjusted_predictions_recover_known_ratios() -> None:
    frame = pd.DataFrame(
        {
            "platform": pd.Categorical(["Uber", "Lyft"], categories=["Uber", "Lyft"]),
            "borough_for_model": pd.Categorical(
                ["Manhattan", "Queens"],
                categories=["Manhattan", "Bronx", "Brooklyn", "Queens", "Staten Island"],
            ),
        }
    )
    result = adjusted_predictions(FakeResult(), frame, 1.0)
    lyft = result.loc[
        (result["family"] == "platform") & (result["level"] == "Lyft"),
        "ratio_to_reference",
    ].iloc[0]
    queens = result.loc[
        (result["family"] == "borough") & (result["level"] == "Queens"),
        "ratio_to_reference",
    ].iloc[0]

    assert math.isclose(lyft, 1.1)
    assert math.isclose(queens, 1.2)
