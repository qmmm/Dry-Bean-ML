# Dry Bean Classification Project

这是一个围绕 Dry Bean 数据集展开的中文项目，包含从数据清洗、特征工程、特征筛选，到 KNN、ID3、随机森林和 BP 神经网络的训练评估，以及鲁棒性、过拟合和推理速度等实验分析。

## 项目特点

- 统一入口由 `main.py` 提供，命令分发由 `drybean_cli.py` 负责。
- 数据处理、模型训练、实验分析按目录分层组织，职责清晰。
- 所有主要流程都支持命令行执行，便于复现实验结果。
- 输出结果会保存到 `DryBeanDataset/` 下的各个结果目录中，包含 CSV、JSON 和图表。

## 环境要求

建议使用 Python 3.10 及以上版本。

依赖以 `requirements.txt` 为准，当前核心依赖包括：

- matplotlib
- numpy
- pandas
- seaborn

## 安装依赖

在项目根目录执行：

```powershell
pip install -r requirements.txt
```

## 快速开始

先查看可用命令：

```powershell
python .\main.py list
```

直接运行某个任务：

```powershell
python .\main.py run knn
python .\main.py run tree
python .\main.py run bp
```

运行一组任务：

```powershell
python .\main.py group train
python .\main.py group evaluate
python .\main.py group experiments
```

只查看将要执行的命令，不实际运行：

```powershell
python .\main.py --dry-run group train
```

需要给底层脚本传参时，在任务名后面加 `--`：

```powershell
python .\main.py run robustness -- --test-limit 300 --bp-epochs 3
python .\main.py run knn -- --k-values 3,5,7,11
```

## 命令说明

### 数据流水线

- `clean`：清洗原始数据
- `features`：生成特征工程数据
- `select`：筛选特征

对应分组命令：

```powershell
python .\main.py group pipeline
```

### 模型训练

- `knn`：训练并评估 KNN
- `tree`：训练并评估 ID3 和随机森林
- `bp`：训练并评估 BP 神经网络
- `train`：训练全部主模型

对应分组命令：

```powershell
python .\main.py group train
```

### 评估与可视化

- `compare`：多模型指标和图表对比
- `speed`：推理速度测试

对应分组命令：

```powershell
python .\main.py group evaluate
```

### 实验分析

- `robustness`：鲁棒性实验
- `overfitting`：过拟合分析

对应分组命令：

```powershell
python .\main.py group experiments
```

## 目录结构

```text
last_work/
|-- main.py
|-- drybean_cli.py
|-- config.py
|-- requirements.txt
|-- data/
|-- models/
|-- experiments/
|-- utils/
|-- DryBeanDataset/
```

其中：

- `data/`：数据清洗、预处理、特征工程和筛选
- `models/`：KNN、树模型和 BP 神经网络实现
- `experiments/`：训练评估、对比、速度、鲁棒性和过拟合分析
- `utils/`：指标、绘图、结果保存等通用工具
- `DryBeanDataset/`：数据集及各类结果输出

## 输出结果

运行后，结果会按任务写入 `DryBeanDataset/` 下的子目录，例如：

- `cleaned/`
- `features/`
- `selected_features/`
- `knn_results/`
- `tree_model_results/`
- `bp_nn_results/`
- `visualizations/`
- `inference_speed_results/`
- `robustness_results/`
- `overfitting_analysis_results/`

这些目录通常会包含：

- 模型评估报告 JSON
- 预测结果 CSV
- 混淆矩阵 CSV
- 训练历史 CSV
- 可视化图表 PNG

## 数据约定

默认情况下，流程会围绕 `DryBeanDataset/selected_features/standardized/` 下的训练、验证和测试集展开。数据文件名与 `config.py` 中的配置保持一致。

## 备注

- 这个项目的统一入口是 `main.py`，不要直接把它当成业务逻辑文件修改。
- 如果你想先确认当前可执行命令，优先用 `python .\main.py list`。
- 如果后续新增脚本，只要接入 `drybean_cli.py`，就可以沿用同一套入口风格。
