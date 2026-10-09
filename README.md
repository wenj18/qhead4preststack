# HCT-QHead：叠前地震弹性参数概率化反演 / Probabilistic Pre-Stack Seismic Elastic Parameter Inversion

[中文说明](#中文说明) | [English](#english)

论文 / Manuscript: **A Quantile Estimation Head for Probabilistic Pre-Stack Seismic Elastic Parameter Inversion**  
[PDF](./main.pdf)

<a id="中文说明"></a>

## 中文说明

### 1. 项目简介

本目录提供上述论文中 HCT-PHead 与 HCT-QHead 的模型实现、训练、井上评价、剖面和整个工区预测，以及结果绘图与不确定性分析代码。

模型以 **15°、30°、45°** 的三角度地震序列为输入，联合预测 **密度、P 波阻抗和 S 波阻抗**。代码中的输入与输出顺序固定为：

```text
输入：Seis15, Seis30, Seis45
输出：Dn, Ip, Is
```

| 模型 | 组件组合 | 预测结果 | 训练目标 |
| --- | --- | --- | --- |
| HCT-PHead | HCT 主干 + 点预测头 PHead | 三参数确定性预测 | 目标加权 MSE |
| HCT-QHead | HCT 主干 + 分位数估计头 QHead | 三参数条件分位数 | 目标加权 Huber 分位数损失 |

模型按预测头命名：**PHead（Point Prediction Head，点预测头）** 与 **QHead（Quantile Estimation Head，分位数估计头）**。代码变量分别使用 `phead` / `qhead`，目录和模型显示名称使用 `HCT-PHead` / `HCT-QHead`。
两个模型共享相同的 HCT 主干结构。QHead 将变换后的分位水平与主干特征拼接，默认学习 `0.025、0.5、0.975` 三个关键分位数；其中 `0.5` 为中位数预测，`0.025` 和 `0.975` 构成名义覆盖率为 95% 的预测区间。

当前实验使用基于实际工区弹性参数模型生成的**合成叠前地震数据**。论文第 2 节的传统正演、确定性与贝叶斯反演属于理论背景，本目录的运行入口使用已生成的 CSV 数据。当前模型组装接口支持 `hct` 主干；其他主干的接入需另行实现。

### 2. 代码与论文对应关系

以下按当前 `main.tex` 的章节和图表主题对应，图表编号随论文排版可能变化。

| 文件 | 对应论文内容 | 功能 |
| --- | --- | --- |
| [model.py](./Code/model.py) | 第 3 节 Methodology | 通过 `build_model` 组装主干、预测头和损失；提供共享主干特征的多分位数推理接口。 |
| [backbone/hct_backbone.py](./Code/backbone/hct_backbone.py) | 第 3.4 节；HCT 结构图 | 一维卷积、残差混合空洞卷积、位置编码与 Transformer 特征提取。 |
| [head/qhead.py](./Code/head/qhead.py) | 第 3.2 节；QHead 结构图 | 分位水平变换、特征拼接与条件分位数预测。 |
| [head/phead.py](./Code/head/phead.py) | 第 3.4 节；HCT 结构图 | HCT-PHead 的确定性 MLP 预测头。 |
| [loss/quantile_loss.py](./Code/loss/quantile_loss.py)、[loss/mse_loss.py](./Code/loss/mse_loss.py) | 第 3.1、3.3 节；第 4.1 节 | Huber 分位数损失与加权 MSE；目标权重按 `Dn、Ip、Is` 排列。 |
| [HCT-PHead.ipynb](./Code/HCT-PHead.ipynb) | 第 4.1、4.2 节的确定性基准 | 训练、井评价、四条剖面预测及整个工区预测。 |
| [HCT-QHead.ipynb](./Code/HCT-QHead.ipynb) | 第 4.1、4.2 节的分位数预测 | 训练、井评价、四条剖面预测及整个工区预测。 |
| [plt_well.ipynb](./Code/plt_well.ipynb) | 第 4.1 节 W1 输入输出示例；第 4.2 节井曲线与性能表 | 两模型井曲线对比、18 行指标汇总及 LaTeX 表格导出。 |
| [plt_section.ipynb](./Code/plt_section.ipynb) | 第 4.1 节三角度地震图；第 4.2 节四剖面中心预测图 | HCT-PHead、HCT-QHead 和地震剖面组合图。 |
| [plt_uncertainty.ipynb](./Code/plt_uncertainty.ipynb) | 第 4.2 节的不确定性分析 | 相对误差与不确定性剖面、全样本与高误差样本分布、统计表。 |

整个工区预测作为应用流程的扩展入口保留在两个模型 notebook 中；当前论文主要展示井上与四条剖面的实验结果。

### 3. 目录与数据

以下路径均相对于本 `Code/` 目录。编号范围代表多个独立文件或文件夹。

```text
Code/
├── README.md
├── model.py
├── backbone/hct_backbone.py
├── head/                       # phead.py、qhead.py
├── loss/                       # mse_loss.py、quantile_loss.py
├── HCT-PHead.ipynb
├── HCT-QHead.ipynb
├── plt_well.ipynb
├── plt_section.ipynb
├── plt_uncertainty.ipynb
├── well/W1.csv … W20.csv
├── section/profile_s1/ … profile_s4/
│   ├── seismic_theta15.csv
│   ├── seismic_theta30.csv
│   ├── seismic_theta45.csv
│   ├── Dn.csv、Ip.csv、Is.csv
│   └── profile_wells_location.csv
├── workarea/
│   └── seismic_theta15.csv、seismic_theta30.csv、seismic_theta45.csv
├── HCT-PHead/                    # 本方法的权重、标准化器和预测结果
├── HCT-QHead/                  # 本方法的权重、标准化器和预测结果
└── analysis/
    ├── well/
    ├── section/
    └── uncertainty/
```

**测井数据。** `well/W1.csv`～`W20.csv` 包含 `time_s、Dn、Ip、Is、Vp、Vs、Seis15、Seis30、Seis45`。每行对应一个时间采样点，时间单位为秒。当前网络使用三角度地震和 `Dn、Ip、Is`，不以 `Vp、Vs` 为直接预测目标。

| 数据集 | 井号 | 用途 |
| --- | --- | --- |
| 训练 | W1～W15 | 拟合标准化器、优化网络参数 |
| 验证 | W16～W17 | 学习率调整、早停和最佳权重选择 |
| 盲井测试 | W18～W20 | 评价未参与训练的井位 |

井文件按井号数值排序；配置中的井索引从 `0` 开始，例如 `range(0, 15)` 对应 W1～W15。训练与井评价会剔除必要输入或目标中为零、NaN、Inf 的行。输入和目标各自使用 `StandardScaler`，仅在训练井上拟合，预测后反标准化。

**剖面数据。** `profile_s1`～`profile_s4` 对应 S1～S4。矩阵 CSV 的首列为 `time_s`，后续 `Virt_*` 列为沿剖面排列的地震道。`Dn.csv、Ip.csv、Is.csv` 提供参考值；井位表中的 `wellname、trace_index、trace_no、virt_column` 等字段用于井与剖面道匹配。当前预测使用 15°、30°、45°。

**工区数据。** `workarea/` 中三个 CSV 的时间列为 `Time`，后续列以 `SourceX:SourceY` 坐标命名。三个角度必须具有一致的时间轴和道名集合，输出按首个角度的道顺序排列。当前数据文件包含实际数值，并非空占位文件。

### 4. 运行环境与启动

主要依赖为 Python、PyTorch、NumPy、pandas、SciPy、scikit-learn、joblib、Matplotlib、tqdm、IPython，以及 JupyterLab 或 Jupyter Notebook。Notebook 保存的 Python 内核版本为 3.10.19；本目录没有固定完整依赖版本的环境文件。复用 `.pkl` 标准化器时，应使用与其保存环境匹配的 scikit-learn 版本。

安装依赖的基本命令如下；需要 GPU 时，应选择与本机环境匹配的 PyTorch 构建。

```bash
python -m pip install torch numpy pandas scipy scikit-learn joblib matplotlib tqdm ipython jupyterlab

# 从仓库根目录进入 Code；模型 notebook 要求工作目录中存在 model.py 和 well/。
cd Code
jupyter lab
```

在 IDE 中运行时，也应将 notebook 内核的工作目录设为 `Code/`。代码检测到可用 CUDA 后使用 GPU，否则使用 CPU。

### 5. 建议运行顺序

两个模型 notebook 采用相同的六节结构，可分别运行：

| 节 | 内容 | 运行条件 |
| --- | --- | --- |
| 1 | 导入依赖与模型接口 | 首先执行 |
| 2 | 参数配置与模型组装 | 训练或预测前执行 |
| 3 | 训练与验证 | 首次训练或重新训练时执行 |
| 4 | 井上测试与结果导出 | 已有匹配权重和标准化器 |
| 5 | 剖面预测与结果导出 | 已有匹配权重和标准化器 |
| 6 | 整个工区预测与结果导出 | 已有匹配权重、标准化器和工区输入 |

首次训练执行 **1 → 2 → 3**，再按需运行第 4、5、6 节。复用已有模型时执行 **1 → 2 → 4 / 5 / 6**，三个预测入口不要求相互依赖。第 2 节包含 `WORKAREA_*` 等配置，更新 notebook 或重启内核后需要重新执行。当前目录已有模型权重、标准化器及预测结果；重新运行会更新对应输出。

完成预测后，按用途运行绘图 notebook：

1. `plt_well.ipynb`：先完成两模型第 4 节，再运行井上绘图、汇总表格和 LaTeX 导出。汇总直接读取两个 `bywell/metrics_summary_weighted_role_all_targets.csv`，组成 3 参数 × 2 方法 × 3 数据集共 18 行。
2. `plt_section.ipynb`：先完成相应模型第 5 节，再绘制四剖面中心预测图；地震剖面 cell 直接使用输入数据。
3. `plt_uncertainty.ipynb`：先完成 HCT-QHead 第 5 节，再依次运行 **Ip → Is → Dn → 表格**。Ip cell 定义后续共享函数，最后的表格 cell 在 notebook 中打印 LaTeX。

**默认参数。** 输入维度 3，输出维度 3；HCT 初始通道 16、特征维度 256、4 个注意力头、3 层 Transformer；预测头隐藏层宽度 256，dropout 0.2。目标权重为 `(2, 1, 1)`，按 `Dn、Ip、Is` 排列并在损失中归一化。训练采用 Adam，初始学习率 `4e-4`、权重衰减 `1e-3`、最多 1000 轮、随机种子 4；逐井变长训练的 `BATCH_SIZE=1`。学习率调度的耐心值为 10、衰减因子为 0.9、最低学习率为 `5e-7`，早停耐心值为 100。

HCT-QHead 默认使用离散关键分位数采样，Huber 参数 `beta=0.1`，每次训练目标计算共享一次主干特征并采样 3 次分位水平。`BACKBONE_PARAMS、HEAD_PARAMS、LOSS_PARAMS` 可覆盖组件默认配置。更改结构后，需重新训练或使用配置一致的权重；预测入口会检查 checkpoint 中的模型配置及输入、目标顺序。

### 6. 输出文件

| 路径 | 内容 |
| --- | --- |
| `HCT-PHead/`、`HCT-QHead/` | `best_model.pt`、`scaler_X.pkl`、`scaler_y.pkl` |
| `<方法>/bywell/` | 每口井预测 CSV |
| `<方法>/byprofile/` | 剖面预测结果 |
| `<方法>/workarea/` | 工区预测结果 |
| `analysis/well/` | 两模型井曲线对比图、井指标表格 |
| `analysis/section/` | 两模型四剖面组合图及三角度地震图|
| `analysis/uncertainty/` | 各参数的相对误差图、不确定性图、分布图|


### 7. 指标与不确定性解释

| 指标 | 含义与代码约定 |
| --- | --- |
| RMSRE | 均方根相对误差。 |
| PCC | 皮尔逊相关系数。 |
| PICP | 参考值落入预测区间的比例。 |
| MPIW | 平均预测区间宽度除以该参数在所有参与井评价的参考值中的极差。 |
| 相对不确定性 | `(upper - low) / median`，在反标准化后的参数值上计算。 |

数据集指标按各井有效样本数对**井级指标加权平均**，不是先合并所有井的样本再重新计算 RMSRE 或 PCC。HCT-PHead 没有区间预测，因此 PICP、MPIW 在汇总 CSV 中留空，在 LaTeX 表中显示为 `--`。




---

<a id="english"></a>

## English

### 1. Overview

This directory contains the HCT-PHead and HCT-QHead implementations accompanying the manuscript linked above, including training, well evaluation, profile and full-work-area prediction, visualization, and uncertainty analysis.

The networks take seismic sequences at **15°, 30°, and 45°** as input and jointly predict **density, P-wave impedance, and S-wave impedance**. Channel order is fixed:

```text
Input:  Seis15, Seis30, Seis45
Output: Dn, Ip, Is
```

| Model | Components | Prediction | Training objective |
| --- | --- | --- | --- |
| HCT-PHead | HCT backbone + point prediction head (PHead) | Deterministic estimates of three parameters | Target-weighted MSE |
| HCT-QHead | HCT backbone + Quantile Estimation Head (QHead) | Conditional quantiles of three parameters | Target-weighted Huber quantile loss |

Models are named after their prediction heads: **Point Prediction Head (PHead)** and **Quantile Estimation Head (QHead)**. Variables use `phead` / `qhead`, while directories and model display names use `HCT-PHead` / `HCT-QHead`. 

Both models use the same HCT backbone architecture. QHead concatenates a transformed quantile level with the backbone features. The default key quantiles are `0.025, 0.5, 0.975`: the median provides the central estimate, and the two tail quantiles define a nominal 95% prediction interval.

The experiment uses **synthetic pre-stack seismic data** generated from elastic-parameter models of a field study area. The conventional forward, deterministic, and Bayesian inversion methods in Section 2 provide theoretical background; these notebooks consume the prepared CSV data. The model factory currently supports the `hct` backbone; additional backbones require implementation.

### 2. Relationship to the manuscript

The following mapping uses the sections and figure/table topics in the current `main.tex`; figure numbers may change during typesetting.

| File | Manuscript content | Purpose |
| --- | --- | --- |
| [model.py](./Code/model.py) | Section 3: Methodology | Assemble backbone, head, and loss through `build_model`; reuse backbone features for multi-quantile inference. |
| [backbone/hct_backbone.py](./Code/backbone/hct_backbone.py) | Section 3.4; HCT architecture figure | One-dimensional convolution, residual hybrid dilated convolution, positional encoding, and Transformer feature extraction. |
| [head/qhead.py](./Code/head/qhead.py) | Section 3.2; QHead architecture figure | Quantile-level transformation, feature concatenation, and conditional quantile prediction. |
| [head/phead.py](./Code/head/phead.py) | Section 3.4; HCT architecture figure | Deterministic MLP prediction head for HCT-PHead. |
| [loss/quantile_loss.py](./Code/loss/quantile_loss.py), [loss/mse_loss.py](./Code/loss/mse_loss.py) | Sections 3.1, 3.3, and 4.1 | Huber quantile loss and weighted MSE, with target weights ordered as `Dn, Ip, Is`. |
| [HCT-PHead.ipynb](./Code/HCT-PHead.ipynb) | Sections 4.1–4.2: deterministic baseline | Training, well evaluation, four-profile prediction, and full-work-area prediction. |
| [HCT-QHead.ipynb](./Code/HCT-QHead.ipynb) | Sections 4.1–4.2: quantile prediction | Training, well evaluation, four-profile prediction, and full-work-area prediction. |
| [plt_well.ipynb](./Code/plt_well.ipynb) | Section 4.1: W1 input/target example; Section 4.2: well curves and performance table | Compare well predictions, combine 18 metric rows, and export a LaTeX table. |
| [plt_section.ipynb](./Code/plt_section.ipynb) | Section 4.1: three-angle seismic figures; Section 4.2: central estimates along four profiles | Combined profile figures for HCT-PHead, HCT-QHead, and seismic data. |
| [plt_uncertainty.ipynb](./Code/plt_uncertainty.ipynb) | Section 4.2: uncertainty analysis | Relative-error/uncertainty maps, distributions for all and high-error samples, and summary statistics. |

Full-work-area inference is included as an additional application workflow. The current manuscript primarily presents well and four-profile results.

### 3. Directory layout and data

All paths below are relative to `Code/`. Number ranges denote separate files or directories.

```text
Code/
├── README.md
├── model.py
├── backbone/hct_backbone.py
├── head/                       # phead.py, qhead.py
├── loss/                       # mse_loss.py, quantile_loss.py
├── HCT-PHead.ipynb
├── HCT-QHead.ipynb
├── plt_well.ipynb
├── plt_section.ipynb
├── plt_uncertainty.ipynb
├── well/W1.csv … W20.csv
├── section/profile_s1/ … profile_s4/
│   ├── seismic_theta15.csv
│   ├── seismic_theta30.csv
│   ├── seismic_theta45.csv
│   ├── Dn.csv, Ip.csv, Is.csv
│   └── profile_wells_location.csv
├── workarea/
│   └── seismic_theta15.csv, seismic_theta30.csv, seismic_theta45.csv
├── HCT-PHead/                    # Weights, scalers, and model outputs
├── HCT-QHead/                  # Weights, scalers, and model outputs
└── analysis/
    ├── well/
    ├── section/
    └── uncertainty/
```

**Well data.** `well/W1.csv`–`W20.csv` contain `time_s, Dn, Ip, Is, Vp, Vs, Seis15, Seis30, Seis45`. Each row is a time sample, with time in seconds. The current networks use the three angle channels and directly predict `Dn, Ip, Is`; `Vp, Vs` are not direct output targets.

| Dataset | Wells | Purpose |
| --- | --- | --- |
| Training | W1–W15 | Fit scalers and optimize network parameters |
| Validation | W16–W17 | Learning-rate scheduling, early stopping, and checkpoint selection |
| Blind test | W18–W20 | Evaluate wells excluded from training |

Well files are sorted by numeric well number. Configuration indices start at zero: `range(0, 15)` corresponds to W1–W15. Training and well evaluation remove rows containing zero, NaN, or Inf in required input/target fields. Separate input and target `StandardScaler` objects are fitted only on training wells. Predictions are transformed back to physical units.

**Profile data.** `profile_s1`–`profile_s4` correspond to S1–S4. Matrix CSVs contain a leading `time_s` column followed by `Virt_*` trace columns. `Dn.csv, Ip.csv, Is.csv` provide reference parameters. Fields such as `wellname, trace_index, trace_no, virt_column` in the well-location table link wells to profile traces. The current prediction workflow uses 15°, 30°, and 45°.

**Work-area data.** The three input files in `workarea/` contain a `Time` column followed by traces named `SourceX:SourceY`. They must have matching time axes and trace-name sets; outputs follow the first angle's trace order. The current files contain numerical data and are not empty placeholders.

### 4. Environment and startup

Dependencies include Python, PyTorch, NumPy, pandas, SciPy, scikit-learn, joblib, Matplotlib, tqdm, IPython, and JupyterLab or Jupyter Notebook. Notebook metadata records Python 3.10.19; a fully pinned environment file is not provided. When reusing serialized `.pkl` scalers, use a scikit-learn version compatible with the environment that saved them.

Basic dependency installation:

```bash
python -m pip install torch numpy pandas scipy scikit-learn joblib matplotlib tqdm ipython jupyterlab

# From the repository root; model notebooks expect model.py and well/ in the working directory.
cd Code
jupyter lab
```

For GPU execution, select a PyTorch build suitable for the local environment. The notebooks use CUDA when available and otherwise fall back to CPU. In an IDE, also set the notebook kernel's working directory to `Code/`.

### 5. Recommended workflow

Both model notebooks contain the same six sections and can be run separately:

| Section | Task | Prerequisite |
| --- | --- | --- |
| 1 | Imports and model interface | Run first |
| 2 | Configuration and model assembly | Run before training or prediction |
| 3 | Training and validation | Run for initial training or retraining |
| 4 | Well evaluation and export | Matching weights and scalers |
| 5 | Profile prediction and export | Matching weights and scalers |
| 6 | Full-work-area prediction and export | Matching weights, scalers, and work-area inputs |

For training, run **1 → 2 → 3**, followed by Sections 4, 5, or 6 as needed. With existing artifacts, run **1 → 2 → 4 / 5 / 6**. The three prediction sections do not depend on one another. Section 2 defines settings including `WORKAREA_*`; rerun it after notebook updates or a kernel restart. Model weights, scalers, and prediction outputs are already present in the current directory. Rerunning a workflow updates its outputs.

After prediction:

1. `plt_well.ipynb`: run Section 4 of both models first, then the comparison plots, metric summary, and LaTeX export. The summary reads each model's `bywell/metrics_summary_weighted_role_all_targets.csv`, producing 18 rows: 3 parameters × 2 methods × 3 datasets.
2. `plt_section.ipynb`: run Section 5 of the relevant model before plotting central estimates. The seismic plotting cell uses the input data directly.
3. `plt_uncertainty.ipynb`: run HCT-QHead Section 5 first, then execute **Ip → Is → Dn → table**. The Ip cell defines shared functions; the final table cell prints LaTeX inside the notebook.

**Default settings.** Three input/output channels; HCT initial channels 16, feature dimension 256, four attention heads, and three Transformer blocks; prediction-head hidden width 256 and dropout 0.2. Target weights `(2, 1, 1)` follow `Dn, Ip, Is` and are normalized within the loss. Training uses Adam, learning rate `4e-4`, weight decay `1e-3`, up to 1,000 epochs, and random seed 4. Variable-length wells are processed individually with `BATCH_SIZE=1`. The learning-rate scheduler uses patience 10, factor 0.9, and minimum rate `5e-7`; early-stopping patience is 100.

HCT-QHead defaults to discrete key-quantile sampling with Huber `beta=0.1`. Each training objective reuses one backbone feature computation and averages three quantile-level draws. `BACKBONE_PARAMS, HEAD_PARAMS, LOSS_PARAMS` override component defaults. Architecture changes require retraining or matching weights; prediction checks checkpoint configuration and input/target order.

### 6. Outputs

| Path | Contents |
| --- | --- |
| `HCT-PHead/`, `HCT-QHead/` | `best_model.pt`, `scaler_X.pkl`, `scaler_y.pkl` |
| `<method>/bywell/` | Per-well prediction CSVs |
| `<method>/byprofile/` | Profile prediction results |
| `<method>/workarea/` | Work-area prediction results |
| `analysis/well/` | Well-curve comparisons between the two models and well-metric tables |
| `analysis/section/` | Combined four-profile figures for both models and three-angle seismic figures |
| `analysis/uncertainty/` | Relative-error maps, uncertainty maps, and distribution plots for each parameter |

### 7. Metrics and uncertainty interpretation

| Metric | Definition and convention |
| --- | --- |
| RMSRE | Root-mean-square relative error. |
| PCC | Pearson correlation coefficient. |
| PICP | Fraction of reference values covered by the prediction interval. |
| MPIW | Mean interval width divided by the target's reference-value range across all wells included in the evaluation. |
| Relative uncertainty | `(upper - low) / median`, evaluated after inverse standardization. |

Dataset-level metrics are **sample-count-weighted averages of well-level metrics**. They are not RMSRE or PCC recomputed from pooled samples. HCT-PHead has no interval predictions, so its PICP/MPIW entries are blank in the combined CSV and shown as `--` in LaTeX.
