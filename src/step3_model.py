# -*- coding: utf-8 -*-
"""Step 3: 无泄漏建模、验证集选模、概率校准与冻结风险分层。"""

import os
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from modeling import (
    FinancialFeatureTransformer,
    MODEL_FEATURES,
    ModelFeatureSelector,
    make_tier_bins,
    tier_bin_health,
)


os.makedirs("output", exist_ok=True)


def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))


raw = pd.read_pickle("data/df_raw.pkl")
raw = raw[raw["age"] > 0].copy()  # 确定性数据校验，不估计任何统计量
X_raw = raw.drop(columns=["default_2y"])
y = raw["default_2y"]

# 外层测试集只在最终一次评估时使用。
X_train, X_te_raw, y_train, y_te = train_test_split(
    X_raw, y, test_size=0.30, random_state=42, stratify=y
)
# 内层验证集再拆成两份互不相交的子集，避免同一批样本同时承担三种用途：
#   val_select —— 选树数，并在校准之后用来确定风险分层阈值
#   val_cal    —— 只用于拟合等温校准器
# 这样分层阈值读到的是"校准器没见过"的概率，不会因等温回归的平台值而退化。
X_fit_raw, X_val_raw, y_fit, y_val = train_test_split(
    X_train, y_train, test_size=0.20, random_state=43, stratify=y_train
)
X_sel_raw, X_cal_raw, y_sel, y_cal = train_test_split(
    X_val_raw, y_val, test_size=0.50, random_state=44, stratify=y_val
)
print(
    f"建模集 {len(X_fit_raw):,} | 验证集 {len(X_val_raw):,} "
    f"(选模 {len(X_sel_raw):,} / 校准 {len(X_cal_raw):,}) | "
    f"冻结测试集 {len(X_te_raw):,}（违约率 {y_te.mean():.2%}）"
)

# 只用建模集学习中位数、截断分位数和特殊编码上限。
feature_transformer = FinancialFeatureTransformer().fit(X_fit_raw)
X_fit = feature_transformer.transform(X_fit_raw)[MODEL_FEATURES]
X_sel = feature_transformer.transform(X_sel_raw)[MODEL_FEATURES]
X_cal = feature_transformer.transform(X_cal_raw)[MODEL_FEATURES]
X_te = feature_transformer.transform(X_te_raw)[MODEL_FEATURES]

results = {}

# ---------- 模型1: 逻辑回归基线 ----------
scaler = StandardScaler().fit(X_fit)
lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=0.1)
lr.fit(scaler.transform(X_fit), y_fit)
p_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]
results["LogisticRegression"] = {
    "auc": roc_auc_score(y_te, p_lr),
    "ks": ks_stat(y_te, p_lr),
}

# ---------- 模型2: LightGBM；仅在验证集选择树数 ----------
scale_pos_weight = (y_fit == 0).sum() / (y_fit == 1).sum()
candidate_estimators = [100, 300, 500, 800]
validation_rows = []
candidate_models = {}
for n_estimators in candidate_estimators:
    candidate = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        verbose=-1,
    )
    candidate.fit(X_fit, y_fit)
    sel_prob = candidate.predict_proba(X_sel)[:, 1]
    validation_rows.append({
        "n_estimators": n_estimators,
        "validation_auc": roc_auc_score(y_sel, sel_prob),
    })
    candidate_models[n_estimators] = candidate

validation_df = pd.DataFrame(validation_rows)
best_n = int(
    validation_df.sort_values(
        ["validation_auc", "n_estimators"], ascending=[False, True]
    ).iloc[0]["n_estimators"]
)
lgbm = candidate_models[best_n]
print("\n===== 选模子集选树数（校准子集与测试集均未参与） =====")
print(validation_df.to_string(index=False, float_format=lambda v: f"{v:.5f}"))
print(f"选择 n_estimators={best_n}")

# ---------- 概率校准：只用 val_cal 拟合 ----------
p_cal_raw = lgbm.predict_proba(X_cal)[:, 1]
p_sel_raw = lgbm.predict_proba(X_sel)[:, 1]
p_gbm_raw = lgbm.predict_proba(X_te)[:, 1]
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(p_cal_raw, y_cal)
# 分层阈值读的是选模子集——校准器没有在它上面拟合过，因此不会落在平台值上。
p_sel = calibrator.transform(p_sel_raw)
p_gbm = calibrator.transform(p_gbm_raw)

results["LightGBM"] = {
    "auc": roc_auc_score(y_te, p_gbm_raw),
    "ks": ks_stat(y_te, p_gbm_raw),
    "brier_raw": brier_score_loss(y_te, p_gbm_raw),
    "brier_calibrated": brier_score_loss(y_te, p_gbm),
    "logloss_raw": log_loss(y_te, p_gbm_raw),
    "logloss_calibrated": log_loss(y_te, p_gbm),
}

# 5 折 CV 中每一折独立拟合 FinancialFeatureTransformer，避免预处理泄漏。
# ModelFeatureSelector 让 CV 用到的特征与最终模型完全一致（17 个）。
# 少了这一步，Pipeline 会把 *_capped_flag 等诊断列也喂给模型，
# CV AUC 与冻结测试集 AUC 就成了两个口径，并列在报告里会误导。
cv_model = Pipeline([
    ("features", FinancialFeatureTransformer()),
    ("select", ModelFeatureSelector()),
    ("model", lgb.LGBMClassifier(
        n_estimators=best_n,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=5,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=5.0,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        verbose=-1,
    )),
])
cv = cross_val_score(
    cv_model,
    X_train,
    y_train,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc",
    # 单进程保证在受限容器/CI/macOS 环境均可复现；数据量下耗时仍可接受。
    n_jobs=1,
)

print("\n===== 冻结测试集表现 =====")
for name, result in results.items():
    print(f"{name:20s} AUC={result['auc']:.4f}  KS={result['ks']:.4f}")
print(f"LightGBM 5折CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")
print(
    "校准 Brier: "
    f"{results['LightGBM']['brier_raw']:.4f} → "
    f"{results['LightGBM']['brier_calibrated']:.4f}"
)

# ---------- 特征重要性 ----------
imp = pd.DataFrame({
    "feature": MODEL_FEATURES,
    "gain": lgbm.booster_.feature_importance("gain"),
})
imp["gain_pct"] = imp["gain"] / imp["gain"].sum()
imp = imp.sort_values("gain", ascending=False).reset_index(drop=True)

# ---------- 风险分层：阈值由"校准器未见过"的选模子集确定，冻结后应用到测试集 ----------
tier_health = tier_bin_health(p_sel)
print(
    "\n===== 分层阈值健康度（阈值样本：选模子集，校准器未见过）====="
    f"\n样本量 {tier_health['sample_size']:,}｜"
    f"不同概率取值 {tier_health['distinct_probabilities']:,}｜"
    f"退化切点 {tier_health['collapsed_cutoffs']}"
)
bins = make_tier_bins(p_sel, strict=True)
te = X_te.copy()
te["y"] = y_te.values
te["prob_raw"] = p_gbm_raw
te["prob"] = p_gbm
te["tier"] = pd.cut(
    te["prob"], bins=bins, labels=["A", "B", "C", "D", "E"], include_lowest=True
)
tier = te.groupby("tier", observed=True).agg(
    客户数=("y", "size"),
    实际违约率=("y", "mean"),
    平均校准PD=("prob", "mean"),
)
tier["坏账捕获占比"] = (
    te.groupby("tier", observed=True)["y"].sum() / te["y"].sum()
)
tier["客户占比"] = tier["客户数"] / tier["客户数"].sum()
print("\n===== 冻结阈值后的测试集风险分层 =====")
print(
    tier.assign(
        实际违约率=tier["实际违约率"].map("{:.1%}".format),
        平均校准PD=tier["平均校准PD"].map("{:.1%}".format),
        坏账捕获占比=tier["坏账捕获占比"].map("{:.1%}".format),
        客户占比=tier["客户占比"].map("{:.1%}".format),
    ).to_string()
)

calibration = calibration_curve(y_te, p_gbm, n_bins=10, strategy="quantile")
with open("data/model_artifacts.pkl", "wb") as f:
    pickle.dump({
        "lgbm": lgbm,
        "lr": lr,
        "scaler": scaler,
        "feature_transformer": feature_transformer,
        "calibrator": calibrator,
        "features": MODEL_FEATURES,
        "X_te": X_te,
        "y_te": y_te,
        "p_gbm": p_gbm,
        "p_gbm_raw": p_gbm_raw,
        "p_sel": p_sel,
        "y_sel": y_sel,
        "tier_health": tier_health,
        "p_lr": p_lr,
        "imp": imp,
        "tier": tier,
        "te": te,
        "bins": bins,
        "results": results,
        "cv": cv,
        "validation_search": validation_df,
        "calibration_curve": calibration,
    }, f)
print("\n模型、校准器、冻结分层阈值与评估结果已保存")
