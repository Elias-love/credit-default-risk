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


# ============================================================
# 代码审查后补充：锁住两处已修复的评估口径问题
# ============================================================

def test_feature_selector_aligns_cv_and_final_model_features():
    """CV Pipeline 必须与最终模型用同一套特征。

    FinancialFeatureTransformer.transform 会额外产出 *_capped_flag 等诊断列，
    旧的 CV Pipeline 直接把它们喂给模型，导致 CV AUC 比最终模型多用了特征、
    两个数不可比。ModelFeatureSelector 负责把口径收敛回 MODEL_FEATURES。
    """
    from src.modeling import MODEL_FEATURES, ModelFeatureSelector

    frame = _sample()
    engineered = FinancialFeatureTransformer().fit(frame).transform(frame)
    # 特征工程本身确实会多出诊断列
    assert set(engineered.columns) - set(MODEL_FEATURES)

    selected = ModelFeatureSelector().fit(engineered).transform(engineered)
    assert list(selected.columns) == MODEL_FEATURES


def test_feature_selector_rejects_missing_features():
    from src.modeling import ModelFeatureSelector
    import pytest

    with pytest.raises(ValueError, match="缺少建模特征"):
        ModelFeatureSelector().fit(pd.DataFrame({"age": [1, 2]}))


def test_tier_bins_strict_mode_rejects_degenerate_cutoffs():
    """等温校准器在自身训练样本上的输出会大量落到同一平台值，
    分位切点因此重合。strict=True 必须报错，而不是用 nextafter 悄悄补一个
    相邻浮点数造出"看起来有区分度"的阈值。"""
    import pytest
    from src.modeling import make_tier_bins

    degenerate = [0.02] * 90 + [0.9] * 10   # 只有两个取值 -> 内部切点重合
    with pytest.raises(ValueError, match="分层阈值退化"):
        make_tier_bins(degenerate, strict=True)

    # 非 strict 仍保持向后兼容：返回严格递增的边界
    bins = make_tier_bins(degenerate)
    assert np.all(np.diff(bins) > 0)


def test_tier_bin_health_flags_collapsed_cutoffs():
    from src.modeling import tier_bin_health

    healthy = tier_bin_health(np.linspace(0.01, 0.6, 500))
    assert healthy["collapsed_cutoffs"] == 0
    assert healthy["distinct_probabilities"] == 500

    collapsed = tier_bin_health([0.02] * 90 + [0.9] * 10)
    assert collapsed["collapsed_cutoffs"] > 0
