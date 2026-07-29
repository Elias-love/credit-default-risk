"""可复用的无泄漏特征工程组件。

所有需要从数据估计的统计量（中位数、分位数、正常逾期上限）只在 ``fit``
阶段学习。模型训练时把本转换器放入 sklearn Pipeline，交叉验证的每个折都会
独立拟合，避免测试集或验证折信息进入特征工程。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


PAST_DUE_COLUMNS = ["pd_30_59", "pd_60_89", "pd_90"]
MODEL_FEATURES = [
    "credit_util", "age", "pd_30_59", "pd_60_89", "pd_90", "debt_ratio",
    "monthly_income", "open_credit_lines", "real_estate_loans", "dependents",
    "delinq_score", "ever_past_due", "monthly_debt", "income_per_capita",
    "unsecured_ratio", "income_missing", "pd_special_flag",
]


class FinancialFeatureTransformer(BaseEstimator, TransformerMixin):
    """训练集拟合、任意数据集复用的清洗与财务特征工程。"""

    def fit(self, X: pd.DataFrame, y=None):
        frame = X.copy()
        self.monthly_income_median_ = float(frame["monthly_income"].median())
        self.dependents_mode_ = float(
            frame["dependents"].mode(dropna=True).iloc[0]
            if not frame["dependents"].mode(dropna=True).empty else 0
        )
        self.quantile_caps_ = {
            col: float(frame[col].quantile(0.995))
            for col in ("credit_util", "debt_ratio")
        }
        self.past_due_caps_ = {}
        for col in PAST_DUE_COLUMNS:
            normal = frame.loc[frame[col] < 90, col]
            self.past_due_caps_[col] = float(normal.max()) if not normal.empty else 0.0
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()

        frame["pd_special_flag"] = (
            frame[PAST_DUE_COLUMNS].ge(90).any(axis=1).astype(int)
        )
        for col in PAST_DUE_COLUMNS:
            frame[col] = frame[col].clip(upper=self.past_due_caps_[col])

        frame["income_missing"] = frame["monthly_income"].isna().astype(int)
        frame["monthly_income"] = frame["monthly_income"].fillna(
            self.monthly_income_median_
        )
        frame["dependents"] = frame["dependents"].fillna(self.dependents_mode_)

        for col, cap in self.quantile_caps_.items():
            frame[f"{col}_capped_flag"] = (frame[col] > cap).astype(int)
            frame[col] = frame[col].clip(upper=cap)

        frame["delinq_score"] = (
            frame["pd_30_59"]
            + 2 * frame["pd_60_89"]
            + 3 * frame["pd_90"]
        )
        frame["ever_past_due"] = (
            frame[PAST_DUE_COLUMNS].sum(axis=1) > 0
        ).astype(int)
        frame["monthly_debt"] = frame["debt_ratio"] * frame["monthly_income"]
        frame["income_per_capita"] = (
            frame["monthly_income"] / (1 + frame["dependents"])
        )
        denom = frame["open_credit_lines"].replace(0, np.nan)
        frame["unsecured_ratio"] = (
            (frame["open_credit_lines"] - frame["real_estate_loans"])
            .clip(lower=0)
            .div(denom)
            .fillna(0)
        )
        return frame


def make_tier_bins(probabilities) -> np.ndarray:
    """用验证集概率生成可冻结到测试/生产数据的 A-E 分层阈值。"""
    bins = np.quantile(probabilities, [0, 0.40, 0.70, 0.90, 0.97, 1.0])
    bins[0], bins[-1] = -np.inf, np.inf
    for i in range(1, len(bins) - 1):
        if bins[i] <= bins[i - 1]:
            bins[i] = np.nextafter(bins[i - 1], np.inf)
    return bins
