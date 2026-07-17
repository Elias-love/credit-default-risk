# -*- coding: utf-8 -*-
"""Step 3: 建模(逻辑回归基线 + LightGBM)、评估(AUC/KS)、风险分层"""
import os
os.makedirs("output", exist_ok=True)
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
import lightgbm as lgb
import pickle

df = pd.read_pickle('data/df_clean.pkl')

FEATURES = ['credit_util','age','pd_30_59','pd_60_89','pd_90','debt_ratio',
            'monthly_income','open_credit_lines','real_estate_loans','dependents',
            'delinq_score','ever_past_due','monthly_debt','income_per_capita',
            'unsecured_ratio','income_missing','pd_special_flag']
X, y = df[FEATURES], df['default_2y']

# 分层切分:保持训练/测试集违约率一致(不均衡数据的必要操作)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"训练集 {len(X_tr):,} (违约率 {y_tr.mean():.2%}) | 测试集 {len(X_te):,} (违约率 {y_te.mean():.2%})")

def ks_stat(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return float(np.max(tpr - fpr))

results = {}

# ---------- 模型1: 逻辑回归(评分卡基线,可解释) ----------
scaler = StandardScaler().fit(X_tr)
lr = LogisticRegression(max_iter=2000, class_weight='balanced', C=0.1)
lr.fit(scaler.transform(X_tr), y_tr)
p_lr = lr.predict_proba(scaler.transform(X_te))[:, 1]
results['LogisticRegression'] = dict(auc=roc_auc_score(y_te, p_lr), ks=ks_stat(y_te, p_lr))

# ---------- 模型2: LightGBM(主模型) ----------
# scale_pos_weight 处理 1:13 不均衡;浅树+强正则防过拟合
# n_estimators 通过 100/300/500/800 网格在测试集外验证选定300
# (早停在本数据上不稳定:强特征使验证AUC早期即平台化)
lgbm = lgb.LGBMClassifier(
    n_estimators=300, learning_rate=0.03, num_leaves=31, max_depth=5,
    min_child_samples=100, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=1.0, reg_lambda=5.0,
    scale_pos_weight=(y_tr == 0).sum() / (y_tr == 1).sum(),
    random_state=42, verbose=-1)
lgbm.fit(X_tr, y_tr)
p_gbm = lgbm.predict_proba(X_te)[:, 1]
results['LightGBM'] = dict(auc=roc_auc_score(y_te, p_gbm), ks=ks_stat(y_te, p_gbm))

# 5折交叉验证确认稳定性(防"一次切分运气好")
cv = cross_val_score(lgbm, X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                     scoring='roc_auc', n_jobs=-1)
print("\n===== 模型表现(测试集) =====")
for name, r in results.items():
    print(f"{name:20s} AUC={r['auc']:.4f}  KS={r['ks']:.4f}")

print(f"LightGBM 5折CV AUC: {cv.mean():.4f} ± {cv.std():.4f}")
print(f"训练集AUC {roc_auc_score(y_tr, lgbm.predict_proba(X_tr)[:,1]):.4f} vs 测试集 {results['LightGBM']['auc']:.4f} (差距小=未过拟合)")

# ---------- 特征重要性 ----------
imp = pd.DataFrame({'feature': FEATURES,
                    'gain': lgbm.booster_.feature_importance('gain')})
imp['gain_pct'] = imp['gain'] / imp['gain'].sum()
imp = imp.sort_values('gain', ascending=False).reset_index(drop=True)
print("\n===== LightGBM 特征重要性(gain) =====")
print(imp.assign(gain_pct=imp['gain_pct'].map('{:.1%}'.format))[['feature','gain_pct']].to_string(index=False))

# ---------- 风险分层(业务落地核心) ----------
te = X_te.copy()
te['y'] = y_te.values
te['prob'] = p_gbm
# 按概率分5层: A(最安全)~E(最危险),按分位切
bins = te['prob'].quantile([0, .40, .70, .90, .97, 1.0]).values.copy()
bins[0], bins[-1] = -1e-9, 1 + 1e-9
te['tier'] = pd.cut(te['prob'], bins=bins, labels=['A', 'B', 'C', 'D', 'E'])
tier = te.groupby('tier', observed=True).agg(
    客户数=('y', 'size'), 实际违约率=('y', 'mean'), 平均预测概率=('prob', 'mean'))
tier['坏账捕获占比'] = te.groupby('tier', observed=True)['y'].sum() / te['y'].sum()
tier['客户占比'] = tier['客户数'] / tier['客户数'].sum()
print("\n===== 风险五级分层(测试集) =====")
print(tier.assign(实际违约率=tier['实际违约率'].map('{:.1%}'.format),
                  平均预测概率=tier['平均预测概率'].map('{:.1%}'.format),
                  坏账捕获占比=tier['坏账捕获占比'].map('{:.1%}'.format),
                  客户占比=tier['客户占比'].map('{:.1%}'.format)).to_string())

de = te[te['tier'].isin(['D','E'])]
print(f"\n关键结论: D+E 两层仅占客户 {len(de)/len(te):.0%},却捕获了 {de['y'].sum()/te['y'].sum():.0%} 的全部违约")

# 保存
with open('data/model_artifacts.pkl', 'wb') as f:
    pickle.dump(dict(lgbm=lgbm, lr=lr, scaler=scaler, features=FEATURES,
                     X_te=X_te, y_te=y_te, p_gbm=p_gbm, p_lr=p_lr,
                     imp=imp, tier=tier, te=te, bins=bins, results=results, cv=cv), f)
print("\n模型与评估结果已保存")
