# 信用违约风险预测模型(Credit Default Risk Prediction)

> 机器学习 · 风控建模 · 财务视角 | LightGBM + 评分卡方法论(WOE/IV) + SHAP 可解释性

基于 **15 万条真实信贷数据**(Kaggle "Give Me Some Credit")构建违约概率模型,并将模型输出翻译为**应收账款管理的财务决策语言**:差异化坏账计提、催收优先级排序、授信预警线。

| 指标 | 结果 |
|---|---|
| 测试集 AUC | **0.868**(该赛题 Kaggle 冠军私榜约 0.869) |
| 测试集 KS | **0.578**(业界 KS>0.4 即为优秀) |
| 5 折 CV 稳定性 | 0.865 ± 0.004 |
| 业务转化 | 前 10% 高风险客户捕获 **56%** 全部坏账 |

![分析图表](assets/analysis_charts.png)

## 项目亮点

1. **真实脏数据处理**:19.8% 非随机收入缺失(缺失标记而非删除)、96/98 特殊编码(截断+标记,因其违约率约 55%,是信号非噪音)、比率字段量纲爆炸(99.5 分位截断)——每条清洗决策都有业务理由
2. **财务视角特征工程**:自建"账龄加权逾期严重度分"(delinq_score),IV 达 1.43 登顶全部 17 个特征,验证账龄加权这一财务直觉在数据上成立
3. **评分卡方法论**:WOE/IV 变量筛选,量化验证"历史付款行为的预测力是财务状况类变量的 10 倍以上"
4. **严谨实验设计**:分层切分防标签失衡、scale_pos_weight 处理 1:13 不均衡、浅树+正则+网格选树数防过拟合、5 折 CV 验证稳定性、诚实披露概率校准问题
5. **可解释性落地**:SHAP 单客户归因,三个代表案例(高危/边界/白名单)展示"每个模型决策可追溯到具体特征",回应财务/审计场景对黑箱模型的质疑
6. **业务闭环**:五级风险分层(E 层实际违约率 55.6% vs A 层 0.8%)→ 三项可落地财务策略

完整分析过程、结果解读与案例分析见 **[docs/analysis_report.md](docs/analysis_report.md)**。

## 快速开始

```bash
# 1. 克隆并安装依赖
git clone https://github.com/Elias-love/credit-default-risk.git
cd credit-default-risk
pip install -r requirements.txt

# 2. 下载数据(公开镜像,约 7MB)
curl -L "https://codeload.github.com/streety/GiveMeSomeCredit/tar.gz/refs/heads/master" -o gmsc.tar.gz
tar xzf gmsc.tar.gz && cp GiveMeSomeCredit-master/cs-training.csv data/ && rm -rf gmsc.tar.gz GiveMeSomeCredit-master

# 3. 按顺序运行(step2-4 依赖上一步输出)
python src/step1_eda.py          # 探索性分析:目标分布/缺失/异常
python src/step2_clean_woe.py    # 清洗 + 特征工程 + WOE/IV
python src/step3_model.py        # 逻辑回归基线 + LightGBM + 风险分层
python src/step4_viz_cases.py    # 8 张图表 + SHAP 个体案例归因
```

全流程固定 `random_state=42`,结果可复现。

## 目录结构

```
├── src/                 # 4 个分析脚本(按序运行)
├── docs/                # 完整分析报告(过程/解读/案例)
├── assets/              # 图表与 IV 排名
├── data/                # 数据目录(需按上述命令自行下载)
└── requirements.txt
```

## 局限与说明

- 数据为**个人信贷**场景;企业应收账款建模需替换特征体系(回款账期偏离度、订单集中度、争议发票率等),但方法论完全同构
- 数据无时间戳,未做 out-of-time 验证;生产环境须按时间切分并以 PSI 监控模型漂移
- scale_pos_weight 使输出概率绝对值偏高但排序有效;用于计提金额需加 Isotonic 校准
- 数据源为 Kaggle 公开竞赛数据,仅用于学习研究

## 路线图

- [ ] Streamlit 交互式风险看板(客户列表 + 风险分 + SHAP 瀑布图)
- [ ] 与 [finance-rag-qa](https://github.com/Elias-love/finance-rag-qa) 的 NL2SQL 模块打通,实现"一句话查客户风险画像"
- [ ] Isotonic 概率校准层
