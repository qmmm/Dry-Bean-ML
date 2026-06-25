# Dry Bean 项目架构

## 统一入口

主入口是 `main.py`，实际命令注册在 `drybean_cli.py`。新代码按下面几层组织：

```text
LAST_WORK/
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

常用命令：

```powershell
python .\main.py list
python .\main.py run knn
python .\main.py run tree
python .\main.py run bp
python .\main.py group train
python .\main.py group experiments
```

需要给底层脚本传参时，用 `--` 分隔：

```powershell
python .\main.py run robustness -- --test-limit 300 --bp-epochs 3
python .\main.py run knn -- --k-values 3,5,7,11
```

只查看将要执行的命令：

```powershell
python .\main.py --dry-run group train
```

## 分层职责

### 数据流水线

这些模块负责把原始数据逐步处理成模型使用的数据。

- `data/load_data.py`
- `data/preprocess.py`
- `data/clean.py`
- `data/feature_engineering.py`
- `data/select_features.py`

对应统一命令：

```powershell
python .\main.py group pipeline
```

### 算法训练

这些模块负责模型实现和对应训练命令。

- `models/base.py`
- `models/knn.py`
- `models/bp_nn.py`
- `models/tree.py`
- `models/naive_bayes.py`

对应统一命令：

```powershell
python .\main.py group train
```

### 评估与可视化

这些模块负责汇总模型结果、绘图和速度测试。

- `experiments/train_eval.py`
- `experiments/compare_models.py`
- `experiments/inference_speed.py`

对应统一命令：

```powershell
python .\main.py group evaluate
```

### 实验分析

这些模块负责额外实验，不直接替代主模型训练。

- `experiments/robustness.py`
- `experiments/overfitting.py`
- `experiments/sample_size.py`

对应统一命令：

```powershell
python .\main.py group experiments
```

## 当前迁移状态

根目录旧脚本已经迁入 `data/`、`models/`、`experiments/`。根目录只保留：

- `main.py`
- `drybean_cli.py`
- `config.py`

后续如果继续简化，可以把模型模块中的重复指标函数和结果保存函数进一步收敛到 `utils/metrics.py`、`utils/save_results.py`。
