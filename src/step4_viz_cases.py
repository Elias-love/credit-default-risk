# -*- coding: utf-8 -*-
"""Step 4: 可视化 + 个体案例归因分析(SHAP)"""
import os
os.makedirs("output", exist_ok=True)
import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from sklearn.metrics import roc_curve, roc_auc_score

# 中文字体(跨平台自动选择)
import platform
_sys = platform.system()
if _sys == 'Darwin':
    _candidates = ['PingFang SC', 'Heiti TC', 'Hiragino Sans GB', 'Arial Unicode MS', 'STHeiti']
    _available = {f.name for f in font_manager.fontManager.ttflist}
    _chosen = next((c for c in _candidates if c in _available), 'Heiti TC')
    plt.rcParams['font.family'] = _chosen
elif _sys == 'Windows':
    plt.rcParams['font.family'] = 'Microsoft YaHei'
else:
    for f in font_manager.findSystemFonts(fontpaths=['/usr/share/fonts/opentype/noto']):
        if 'NotoSansCJK' in f:
            font_manager.fontManager.addfont(f)
    plt.rcParams['font.family'] = 'Noto Sans CJK HK'
plt.rcParams['axes.unicode_minus'] = False

with open('data/model_artifacts.pkl', 'rb') as f:
    A = pickle.load(f)
lgbm, X_te, y_te, p_gbm, p_lr = A['lgbm'], A['X_te'], A['y_te'], A['p_gbm'], A['p_lr']
imp, tier, te, FEATURES = A['imp'], A['tier'], A['te'], A['features']
df = pd.read_pickle('data/df_clean.pkl')

fig = plt.figure(figsize=(16, 20))

# ---- 图1: ROC 曲线 ----
ax1 = fig.add_subplot(4, 2, 1)
for p, name, color in [(p_gbm, 'LightGBM', '#d4622a'), (p_lr, '逻辑回归', '#3a7ca5')]:
    fpr, tpr, _ = roc_curve(y_te, p)
    ax1.plot(fpr, tpr, label=f'{name} AUC={roc_auc_score(y_te,p):.3f}', color=color, lw=2)
ax1.plot([0,1],[0,1],'k--',lw=1,alpha=0.4)
ax1.set_xlabel('假阳性率 FPR'); ax1.set_ylabel('真阳性率 TPR')
ax1.set_title('图1  ROC 曲线', fontsize=13, fontweight='bold'); ax1.legend()

# ---- 图2: KS 曲线 ----
ax2 = fig.add_subplot(4, 2, 2)
fpr, tpr, thr = roc_curve(y_te, p_gbm)
ks_idx = np.argmax(tpr - fpr)
ax2.plot(thr, tpr, label='累计坏客户捕获率(TPR)', color='#c0392b', lw=2)
ax2.plot(thr, fpr, label='累计好客户误伤率(FPR)', color='#27ae60', lw=2)
ax2.vlines(thr[ks_idx], fpr[ks_idx], tpr[ks_idx], color='#555', ls='--',
           label=f'KS={tpr[ks_idx]-fpr[ks_idx]:.3f}')
ax2.set_xlim(0, 1); ax2.invert_xaxis()
ax2.set_xlabel('预测违约概率阈值'); ax2.set_ylabel('累计占比')
ax2.set_title('图2  KS 曲线(好坏客户区分度)', fontsize=13, fontweight='bold'); ax2.legend(fontsize=9)

# ---- 图3: 特征重要性 ----
ax3 = fig.add_subplot(4, 2, 3)
top = imp.head(10).iloc[::-1]
name_cn = {'delinq_score':'逾期严重度加权分','credit_util':'信用额度使用率','ever_past_due':'是否有逾期史',
           'age':'年龄','debt_ratio':'债务收入比','monthly_debt':'月债务额','real_estate_loans':'不动产贷款数',
           'open_credit_lines':'未结清账户数','monthly_income':'月收入','pd_30_59':'30-59天逾期次数',
           'income_per_capita':'人均可支配收入','pd_90':'90天以上逾期次数'}
ax3.barh([name_cn.get(x, x) for x in top['feature']], top['gain_pct']*100, color='#d4622a')
ax3.set_xlabel('贡献占比 %')
ax3.set_title('图3  LightGBM 特征重要性 Top10 (gain)', fontsize=13, fontweight='bold')

# ---- 图4: 风险五级分层 ----
ax4 = fig.add_subplot(4, 2, 4)
tt = tier.reset_index()
bars = ax4.bar(tt['tier'].astype(str), tt['实际违约率']*100,
               color=['#2ecc71','#a3d977','#f5c542','#f08a3c','#c0392b'])
for b, v, n in zip(bars, tt['实际违约率'], tt['客户数']):
    ax4.text(b.get_x()+b.get_width()/2, v*100+1, f'{v:.1%}\n({n:,}人)', ha='center', fontsize=9)
ax4.axhline(y_te.mean()*100, color='#555', ls='--', lw=1, label=f'整体违约率 {y_te.mean():.1%}')
ax4.set_ylabel('实际违约率 %'); ax4.set_ylim(0, 68)
ax4.set_title('图4  风险五级分层:实际违约率(测试集)', fontsize=13, fontweight='bold'); ax4.legend()

# ---- 图5: 额度使用率 vs 违约率 ----
ax5 = fig.add_subplot(4, 2, 5)
d = df.copy()
d['bin'] = pd.qcut(d['credit_util'], 10, duplicates='drop')
g = d.groupby('bin', observed=True)['default_2y'].mean()
ax5.plot(range(len(g)), g.values*100, marker='o', color='#8e44ad', lw=2)
ax5.set_xticks(range(len(g)))
ax5.set_xticklabels([f'{iv.right:.0%}' if iv.right<=1.5 else f'{iv.right:.1f}' for iv in g.index], rotation=45, fontsize=8)
ax5.set_xlabel('信用额度使用率(分箱上限)'); ax5.set_ylabel('违约率 %')
ax5.set_title('图5  额度使用率越高,违约率单调上升', fontsize=13, fontweight='bold')

# ---- 图6: 年龄 vs 违约率 ----
ax6 = fig.add_subplot(4, 2, 6)
d['age_bin'] = pd.cut(d['age'], bins=[20,30,35,40,45,50,55,60,65,70,110])
g2 = d.groupby('age_bin', observed=True)['default_2y'].mean()
ax6.bar(range(len(g2)), g2.values*100, color='#3a7ca5')
ax6.set_xticks(range(len(g2)))
ax6.set_xticklabels([f'{int(iv.left)}-{int(iv.right)}' for iv in g2.index], rotation=45, fontsize=8)
ax6.set_xlabel('年龄段'); ax6.set_ylabel('违约率 %')
ax6.set_title('图6  年龄越轻违约率越高', fontsize=13, fontweight='bold')

# ---- 图7: 分层坏账捕获(累计增益) ----
ax7 = fig.add_subplot(4, 2, 7)
srt = te.sort_values('prob', ascending=False).reset_index(drop=True)
srt['cum_bad'] = srt['y'].cumsum() / srt['y'].sum()
srt['cum_pop'] = (srt.index + 1) / len(srt)
ax7.plot(srt['cum_pop']*100, srt['cum_bad']*100, color='#c0392b', lw=2, label='模型排序')
ax7.plot([0,100],[0,100],'k--',lw=1,alpha=0.4,label='随机')
ax7.fill_between(srt['cum_pop']*100, srt['cum_pop']*100, srt['cum_bad']*100, alpha=0.08, color='#c0392b')
for pct in [10, 20, 30]:
    v = srt.loc[(srt['cum_pop']*100).sub(pct).abs().idxmin(), 'cum_bad']
    ax7.annotate(f'前{pct}%客户\n捕获{v:.0%}坏账', xy=(pct, v*100), xytext=(pct+8, v*100-12),
                 fontsize=9, arrowprops=dict(arrowstyle='->', color='#555'))
ax7.set_xlabel('按风险从高到低排序的客户占比 %'); ax7.set_ylabel('累计捕获坏账占比 %')
ax7.set_title('图7  累计增益:催收/审查资源投放依据', fontsize=13, fontweight='bold'); ax7.legend()

# ---- 图8: 概率校准(分层预测vs实际) ----
ax8 = fig.add_subplot(4, 2, 8)
d10 = te.copy(); d10['dec'] = pd.qcut(d10['prob'], 10, labels=False, duplicates='drop')
gg = d10.groupby('dec').agg(pred=('prob','mean'), act=('y','mean'))
ax8.plot(gg['pred']*100, gg['act']*100, marker='o', color='#d4622a', label='分箱均值')
ax8.plot([0,100],[0,100],'k--',lw=1,alpha=0.4,label='完美校准线')
ax8.set_xlabel('平均预测概率 %(scale_pos_weight 加权后偏高)'); ax8.set_ylabel('实际违约率 %')
ax8.set_title('图8  校准检查:排序有效,绝对概率偏高需校准', fontsize=13, fontweight='bold'); ax8.legend()

plt.tight_layout()
plt.savefig('output/analysis_charts.png', dpi=110, bbox_inches='tight')
print("图表已保存 output/analysis_charts.png")

# ============ 个体案例归因(LightGBM 内置 SHAP) ============
contrib = lgbm.predict_proba(X_te, pred_contrib=True)
# 二分类 predict_proba(pred_contrib) 返回 shape (n, n_feat+1) 的对数几率贡献
if isinstance(contrib, list): contrib = contrib[1]
shap_vals = pd.DataFrame(contrib[:, :-1], columns=FEATURES, index=X_te.index)

te2 = te.copy()
# 选3个代表案例: E层真违约 / D-C层边界 / A层安全
case_high = te2[(te2['tier']=='E') & (te2['y']==1)].sort_values('prob').iloc[[-1]]
case_mid  = te2[(te2['tier']=='C')].sort_values('prob').iloc[[len(te2[te2['tier']=='C'])//2]]
case_low  = te2[(te2['tier']=='A') & (te2['y']==0)].sort_values('prob').iloc[[0]]

print("\n" + "="*72)
print("个体案例归因分析(SHAP 对数几率贡献,正值=推高风险)")
print("="*72)
cases_out = []
for label, case in [('案例1: 高风险(E层,实际违约)', case_high),
                    ('案例2: 边界客户(C层)', case_mid),
                    ('案例3: 低风险(A层,实际正常)', case_low)]:
    idx = case.index[0]
    row = X_te.loc[idx]; sv = shap_vals.loc[idx].sort_values(key=abs, ascending=False)
    print(f"\n--- {label} | 预测排序分 {case['prob'].iloc[0]:.1%} | 实际违约: {'是' if case['y'].iloc[0]==1 else '否'} ---")
    print(f"画像: 年龄{int(row['age'])} | 额度使用率{row['credit_util']:.0%} | 月收入{row['monthly_income']:,.0f} | "
          f"债务比{row['debt_ratio']:.2f} | 逾期30-59天{int(row['pd_30_59'])}次 60-89天{int(row['pd_60_89'])}次 90+天{int(row['pd_90'])}次")
    print("Top5 风险归因:")
    for feat, v in sv.head(5).items():
        direction = '↑推高风险' if v > 0 else '↓降低风险'
        print(f"  {name_cn.get(feat, feat):16s} 值={row[feat]:>10.2f}  SHAP={v:+.3f} {direction}")
    cases_out.append(dict(label=label, idx=int(idx), prob=float(case['prob'].iloc[0]),
                          actual=int(case['y'].iloc[0]), row=row.to_dict(),
                          shap=sv.head(6).to_dict()))

with open('data/cases.pkl','wb') as f: pickle.dump(cases_out, f)
print("\n案例数据已保存")
