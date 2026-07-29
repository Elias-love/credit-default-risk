import numpy as np
import pandas as pd

from src.modeling import FinancialFeatureTransformer, make_tier_bins


def _sample():
    return pd.DataFrame({
        "credit_util": [0.1, 0.3, 50.0],
        "age": [30, 40, 50],
        "pd_30_59": [0, 1, 96],
        "debt_ratio": [0.2, 0.4, 10000.0],
        "monthly_income": [1000.0, 3000.0, np.nan],
        "open_credit_lines": [2, 4, 0],
        "pd_90": [0, 0, 98],
        "real_estate_loans": [0, 1, 0],
        "pd_60_89": [0, 0, 96],
        "dependents": [0, 1, np.nan],
    })


def test_transformer_reuses_training_statistics():
    train = _sample().iloc[:2]
    test = _sample().iloc[[2]]
    transformer = FinancialFeatureTransformer().fit(train)

    out = transformer.transform(test)

    assert out["monthly_income"].iloc[0] == 2000.0
    assert out["pd_special_flag"].iloc[0] == 1
    assert out["pd_30_59"].iloc[0] == 1
    assert out["credit_util"].iloc[0] <= transformer.quantile_caps_["credit_util"]


def test_transform_does_not_mutate_input():
    frame = _sample()
    original = frame.copy(deep=True)
    FinancialFeatureTransformer().fit(frame).transform(frame)
    pd.testing.assert_frame_equal(frame, original)


def test_tier_bins_are_strict_and_cover_all_probabilities():
    bins = make_tier_bins([0.1] * 10)
    assert np.all(np.diff(bins) > 0)
    assert np.isneginf(bins[0])
    assert np.isposinf(bins[-1])
