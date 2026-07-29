# -*- coding: utf-8 -*-
"""Step 2: 数据清洗、特征工程与 WOE/IV 分析"""
import os
os.makedirs("output", exist_ok=True)
import pandas as pd
import numpy as np
from modeling import FinancialFeatureTransformer

df = pd.read_pickle('data/df_raw.pkl')
print(f"清洗前: {df.shape}")

# ---------- 探索性清洗与 WOE/IV ----------
# age=0 属确定性录入错误(1条),删除。其余需要估计统计量的清洗统一走
# FinancialFeatureTransformer；此处在全量数据上拟合仅用于探索性 WOE/IV，
# 生产模型会在 step3 中严格只用训练折拟合，不复用这里的统计量。
df = df[df['age'] > 0].copy()
target = df.pop("default_2y")
transformer = FinancialFeatureTransformer().fit(df)
df = transformer.transform(df)
df["default_2y"] = target

print(f"清洗后: {df.shape}, 违约率: {df['default_2y'].mean():.2%}")

# ---------- WOE / IV(风控标准变量筛选) ----------
def woe_iv(series, target, bins=8):
    """等频分箱计算 WOE/IV;离散少值变量按取值分箱"""
    d = pd.DataFrame({'x': series, 'y': target})
    # 逾期次数等零膨胀离散变量即使有十几个取值也不应强行 qcut；
    # 大量 0 会让分位边界重复，严重时只剩一个箱、IV 被错误算成 0。
    if d['x'].nunique() <= 20:
        d['bin'] = d['x']
    else:
        d['bin'] = pd.qcut(d['x'], q=bins, duplicates='drop')
    g = d.groupby('bin', observed=True)['y'].agg(['count', 'sum'])
    g.columns = ['total', 'bad']
    g['good'] = g['total'] - g['bad']
    g = g[(g['bad'] > 0) & (g['good'] > 0)]
    g['bad_pct'] = g['bad'] / g['bad'].sum()
    g['good_pct'] = g['good'] / g['good'].sum()
    g['woe'] = np.log(g['good_pct'] / g['bad_pct'])
    g['iv'] = (g['good_pct'] - g['bad_pct']) * g['woe']
    g['bad_rate'] = g['bad'] / g['total']
    return g, g['iv'].sum()

features = ['credit_util','age','pd_30_59','debt_ratio','monthly_income',
            'open_credit_lines','pd_90','real_estate_loans','pd_60_89','dependents',
            'delinq_score','ever_past_due','monthly_debt','income_per_capita',
            'unsecured_ratio','income_missing','pd_special_flag']

iv_rows = []
detail = {}
y = df['default_2y']
for f in features:
    g, iv = woe_iv(df[f], y)
    iv_rows.append({'feature': f, 'IV': round(iv, 4)})
    detail[f] = g

iv_df = pd.DataFrame(iv_rows).sort_values('IV', ascending=False).reset_index(drop=True)
def iv_level(v):
    if v >= 0.5: return '极强(注意是否泄漏)'
    if v >= 0.3: return '强'
    if v >= 0.1: return '中等'
    if v >= 0.02: return '弱'
    return '无预测力'
iv_df['预测力'] = iv_df['IV'].map(iv_level)
print("\n===== IV 排名(信息价值) =====")
print(iv_df.to_string(index=False))

# 展示两个代表性变量的分箱明细
for f in ['credit_util', 'age']:
    g = detail[f]
    print(f"\n===== {f} 分箱明细 =====")
    print(g[['total','bad','bad_rate','woe']].assign(
        bad_rate=lambda t: t['bad_rate'].map('{:.1%}'.format),
        woe=lambda t: t['woe'].round(3)).to_string())

iv_df.to_csv('output/iv_ranking.csv', index=False)
df.to_pickle('data/df_clean.pkl')
import pickle
with open('data/woe_detail.pkl','wb') as fp: pickle.dump(detail, fp)
print("\n清洗数据与 WOE 明细已保存")
