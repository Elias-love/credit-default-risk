# -*- coding: utf-8 -*-
"""Step 1: 探索性数据分析 (EDA)"""
import os
os.makedirs("output", exist_ok=True)
import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 20)

df = pd.read_csv('data/cs-training.csv', index_col=0)
# 列名重命名为可读的中英对照简称
cols = {
    'SeriousDlqin2yrs': 'default_2y',              # 未来两年发生90天以上严重逾期(目标)
    'RevolvingUtilizationOfUnsecuredLines': 'credit_util',  # 循环信用额度使用率
    'age': 'age',
    'NumberOfTime30-59DaysPastDueNotWorse': 'pd_30_59',     # 30-59天逾期次数
    'DebtRatio': 'debt_ratio',                     # 债务收入比
    'MonthlyIncome': 'monthly_income',
    'NumberOfOpenCreditLinesAndLoans': 'open_credit_lines', # 未结清信贷账户数
    'NumberOfTimes90DaysLate': 'pd_90',            # 90天以上逾期次数
    'NumberRealEstateLoansOrLines': 'real_estate_loans',
    'NumberOfTime60-89DaysPastDueNotWorse': 'pd_60_89',
    'NumberOfDependents': 'dependents',
}
df = df.rename(columns=cols)

print("=" * 70)
print("1. 数据规模与目标分布")
print("=" * 70)
print(f"样本量: {len(df):,}  特征数: {df.shape[1]-1}")
vc = df['default_2y'].value_counts()
print(f"违约样本: {vc[1]:,} ({vc[1]/len(df):.2%})  正常样本: {vc[0]:,} ({vc[0]/len(df):.2%})")
print(f"不均衡比例约 1 : {vc[0]//vc[1]}")

print("\n" + "=" * 70)
print("2. 缺失值")
print("=" * 70)
miss = df.isnull().sum()
miss_pct = miss / len(df)
print(pd.DataFrame({'缺失数': miss[miss>0], '缺失率': miss_pct[miss>0].map('{:.2%}'.format)}))

print("\n" + "=" * 70)
print("3. 描述性统计(重点看异常值)")
print("=" * 70)
print(df.describe(percentiles=[.01,.25,.5,.75,.99]).T[['min','1%','50%','99%','max','mean']])

print("\n" + "=" * 70)
print("4. 已知异常模式核查")
print("=" * 70)
print(f"age=0 的记录数: {(df['age']==0).sum()}")
print(f"credit_util > 1 (超额使用): {(df['credit_util']>1).sum():,} ({(df['credit_util']>1).mean():.2%})")
print(f"credit_util > 10 (显然异常): {(df['credit_util']>10).sum():,}")
# 逾期字段的 96/98 编码 = 数据源的特殊标记
for c in ['pd_30_59','pd_60_89','pd_90']:
    print(f"{c} 中 96/98 编码: {(df[c]>=90).sum()}  正常最大值: {df.loc[df[c]<90, c].max()}")
print(f"缺失收入的样本违约率: {df.loc[df['monthly_income'].isnull(),'default_2y'].mean():.2%} vs 全体 {df['default_2y'].mean():.2%}")
print(f"debt_ratio>10000 样本数: {(df['debt_ratio']>10000).sum()} (多为收入缺失时债务比失真)")

# 逾期字段与违约率的关系速览
print("\n" + "=" * 70)
print("5. 历史逾期次数 vs 违约率(风控最强信号预检)")
print("=" * 70)
for c in ['pd_30_59','pd_60_89','pd_90']:
    t = df[df[c]<90].groupby(df[c].clip(upper=5))['default_2y'].agg(['count','mean'])
    t.columns=['样本数','违约率']
    t['违约率']=t['违约率'].map('{:.1%}'.format)
    print(f"\n[{c}] (clip至5)"); print(t)

df.to_pickle('data/df_raw.pkl')
print("\n原始数据已保存 df_raw.pkl")
