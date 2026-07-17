# -*- coding: utf-8 -*-
"""Step 2: 数据清洗、特征工程与 WOE/IV 分析"""
import os
os.makedirs("output", exist_ok=True)
import pandas as pd
import numpy as np

df = pd.read_pickle('data/df_raw.pkl')
print(f"清洗前: {df.shape}")

# ---------- 清洗规则(每条都有业务理由,写进报告) ----------
# R1: age=0 属录入错误(1条),删除
df = df[df['age'] > 0].copy()

# R2: 逾期次数 96/98 为数据源特殊编码(269条,三个字段同时为96/98)。
#     这类客户违约率极高(~55%),不是随机错误 => 不删除,截断至正常上限,
#     并保留一个"曾出现特殊编码"标记(等价于"状态异常客户")
df['pd_special_flag'] = (df['pd_30_59'] >= 90).astype(int)
for c in ['pd_30_59', 'pd_60_89', 'pd_90']:
    cap = df.loc[df[c] < 90, c].max()
    df[c] = df[c].clip(upper=cap)

# R3: 收入缺失(19.8%) => 不删除(缺失本身可能有信息),
#     用中位数填充 + 缺失标记列
df['income_missing'] = df['monthly_income'].isnull().astype(int)
df['monthly_income'] = df['monthly_income'].fillna(df['monthly_income'].median())

# R4: dependents 缺失(2.6%) => 众数0填充
df['dependents'] = df['dependents'].fillna(0)

# R5: credit_util 与 debt_ratio 极端值 => 99.5分位截断(保留排序信息,消除量纲爆炸)
for c in ['credit_util', 'debt_ratio']:
    q = df[c].quantile(0.995)
    df[f'{c}_capped_flag'] = (df[c] > q).astype(int)
    df[c] = df[c].clip(upper=q)

# ---------- 特征工程(财务视角) ----------
# F1: 逾期严重度加权分 —— 类似账龄加权,90+权重最高
df['delinq_score'] = df['pd_30_59']*1 + df['pd_60_89']*2 + df['pd_90']*3
# F2: 是否有任何逾期史(最强单一二元信号)
df['ever_past_due'] = ((df['pd_30_59']+df['pd_60_89']+df['pd_90']) > 0).astype(int)
# F3: 月债务额 = 债务比 × 月收入(还原绝对负担)
df['monthly_debt'] = df['debt_ratio'] * df['monthly_income']
# F4: 人均可支配收入 = 收入 / (1+抚养人数)
df['income_per_capita'] = df['monthly_income'] / (1 + df['dependents'])
# F5: 无抵押账户占比(信用卡类账户 vs 房贷类账户结构)
df['unsecured_ratio'] = (df['open_credit_lines'] - df['real_estate_loans']).clip(lower=0) / df['open_credit_lines'].replace(0, np.nan)
df['unsecured_ratio'] = df['unsecured_ratio'].fillna(0)

print(f"清洗后: {df.shape}, 违约率: {df['default_2y'].mean():.2%}")

# ---------- WOE / IV(风控标准变量筛选) ----------
def woe_iv(series, target, bins=8):
    """等频分箱计算 WOE/IV;离散少值变量按取值分箱"""
    d = pd.DataFrame({'x': series, 'y': target})
    if d['x'].nunique() <= 10:
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
