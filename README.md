# SALCA-IB

## 项目简介

SALCA-IB（Self-Adaptive LLM-Driven Continuous Learning Agent for IB Network Failure Prediction）是一种创新的自适应故障预测系统，旨在通过结合大语言模型（LLM）与传统机器学习方法，提升InfiniBand（IB）网络故障预测的准确性和适应性。SALCA-IB利用LLM进行智能决策，包括数据选择、模型选择、超参数优化及反馈生成，实现了模型的动态调整和持续学习，确保系统能够应对IB网络环境的复杂性和动态变化。

## 目录说明

- **README.md**: 项目简介、安装指南、使用说明、贡献指南等。
- **requirements.txt**: 项目所需的Python依赖包列表。
- **config/**: 配置文件目录，包含项目的各种配置文件。
  - `config.yaml`: 参数配置文件。
  - `logging.yaml`: 日志配置文件。
- **data/**: 数据存放目录。
  - `raw/`: 原始数据。
  - `processed/`: 处理后的数据，用于训练和测试。
  - `external/`: 外部数据源。
  - `README.md`: 数据目录说明。
- **src/**: 源代码目录，包含项目的主要功能模块。
  - `data_preprocessing/`: 数据预处理模块。
    - `load_data.py`: 数据加载。
    - `clean_data.py`: 数据清洗。
    - `feature_engineering.py`: 特征工程。
  - `models/`: 各种预测模型及集成模型。
    - `xgboost_model.py`: XGBoost模型。
    - `lstm_model.py`: LSTM模型。
    - `gru_model.py`: GRU模型。
    - `cnn_model.py`: CNN模型。
    - `transformer_model.py`: Transformer编码器模型。
    - `model_ensemble.py`: 模型集成策略。
  - `llm/`: 大语言模型相关模块。
    - `llm_interface.py`: LLM交互接口。
    - `decision_maker.py`: 决策制定核心逻辑。
  - `memory/`: 记忆系统模块。
    - `short_term_memory.py`: 短期记忆实现。
    - `long_term_memory.py`: 长期记忆实现。
  - `online_learning/`: 在线学习模块。
    - `online_update.py`: 模型在线更新逻辑。
  - `evaluation/`: 模型评估模块。
    - `evaluate.py`: 评估指标与方法。
  - `feedback/`: 反馈生成模块。
    - `feedback_generator.py`: 生成反馈信息。
  - `main.py`: 项目入口脚本，整合各模块执行整体工作流程。
- **tests/**: 测试目录，包含各模块的单元测试。
  - `test_data_preprocessing.py`: 数据预处理模块测试。
  - `test_models.py`: 模型模块测试。
  - `test_llm.py`: LLM模块测试。
  - `test_memory.py`: 记忆系统模块测试。
  - `test_online_learning.py`: 在线学习模块测试。
  - `test_evaluation.py`: 评估模块测试。
- **scripts/**: 自动化脚本目录，用于训练、评估和环境配置。
  - `run_training.sh`: 训练模型脚本。
  - `run_evaluation.sh`: 评估模型脚本。
  - `setup_environment.sh`: 环境配置脚本。
- **notebooks/**: Jupyter笔记本目录，用于数据探索、模型训练和结果可视化。
  - `exploratory_data_analysis.ipynb`: 数据探索分析。
  - `model_training.ipynb`: 模型训练记录。
  - `results_visualization.ipynb`: 结果可视化。

## 安装与配置

### 环境配置

1. **克隆仓库**

    ```bash
    git clone https://github.com/您的用户名/SALCA-IB.git
    cd SALCA-IB
    ```

2. **创建虚拟环境（推荐）**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. **安装依赖**

    ```bash
    pip install -r requirements.txt
    ```

4. **配置文件**

    - 在 `config/` 目录下，根据需求编辑 `config.yaml` 和 `logging.yaml` 文件，调整参数配置和日志设置。

### 数据准备

将原始数据放入 `data/raw/` 目录。可以参考 `data/README.md` 了解数据说明。

## 使用说明

### 训练模型

执行以下脚本开始模型训练：

```bash
bash scripts/run_training.sh
```

### 评估模型

执行以下脚本进行模型评估：

```bash
bash scripts/run_evaluation.sh
```

### 数据预处理

您可以使用 `src/data_preprocessing` 目录下的脚本进行数据加载、清洗和特征工程。例如：

```bash
python src/data_preprocessing/load_data.py
python src/data_preprocessing/clean_data.py
python src/data_preprocessing/feature_engineering.py
```

### 运行主程序

直接运行主程序以执行完整的工作流程：

```bash
python src/main.py
```

### Jupyter 笔记本

项目中包含多个 Jupyter 笔记本，用于数据探索和结果可视化。在 `notebooks/` 目录下可以找到：

- `exploratory_data_analysis.ipynb`: 数据探索分析。
- `model_training.ipynb`: 模型训练过程记录。
- `results_visualization.ipynb`: 结果可视化。

启动 Jupyter Notebook：

```bash
jupyter notebook
```

## 贡献指南

欢迎任何形式的贡献！请遵循以下步骤：

1. **Fork 仓库**
2. **创建新分支**

    ```bash
    git checkout -b feature/您的功能
    ```

3. **提交更改**

    ```bash
    git commit -m "描述您的更改"
    ```

4. **推送到分支**

    ```bash
    git push origin feature/您的功能
    ```

5. **创建 Pull Request**

确保您的代码通过了所有测试，并附上必要的文档。

## 许可证

本项目使用 [MIT 许可证](LICENSE) 许可证。详细信息请参见 `LICENSE` 文件。

## 联系方式

如果您有任何问题或建议，请联系 [您的邮箱](mailto:your.email@example.com)。

## 致谢

感谢所有参与和支持本项目的贡献者和研究人员。
