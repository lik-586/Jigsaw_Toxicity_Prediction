# Jigsaw Toxicity Prediction

基于深度学习的多标签有毒评论分类项目，使用 DistilRoBERTa 和 TextCNN 两种模型对评论文本进行毒性检测。

## 项目概述

本项目针对 Jigsaw Toxic Comment Classification 竞赛任务，对评论文本进行六类毒性标签的预测，包括：

- `toxic`（有毒）
- `severe_toxic`（严重有毒）
- `obscene`（淫秽）
- `threat`（威胁）
- `insult`（侮辱）
- `identity_hate`（身份仇恨）

支持两种深度学习模型：

| 模型 | 特点 | 适用场景 |
|------|------|----------|
| DistilRoBERTa | 基于 Transformer 的预训练语言模型，准确率高 | 追求精度的场景 |
| TextCNN | 轻量级卷积神经网络，训练速度快 | 资源受限或快速迭代的场景 |

## 项目结构

```
.
├── data/                        # 数据集目录
│   ├── sample_submission.csv    # 提交样例
│   ├── test_labels.csv          # 测试集标签
│   ├── train.csv                # 训练数据（需自行放置）
│   └── test.csv                 # 测试数据（需自行放置）
├── src/                         # 源代码目录
│   ├── __init__.py              # 主包导出
│   ├── config.py                # 全局配置与超参数
│   ├── data/
│   │   ├── __init__.py          # 数据模块导出
│   │   ├── loader.py            # 数据加载与清洗
│   │   └── dataset.py           # PyTorch Dataset 定义
│   ├── models/
│   │   ├── __init__.py          # 模型模块导出
│   │   ├── roberta.py           # DistilRoBERTa 模型
│   │   └── textcnn.py           # TextCNN 模型
│   ├── scripts/
│   │   ├── train_roberta.py     # RoBERTa 训练脚本
│   │   └── train_cnn.py         # TextCNN 训练脚本
│   └── utils/
│       ├── __init__.py          # 工具模块导出
│       └── trainer.py           # 训练与评估函数
├── test_project.py              # 单元测试脚本
├── outputs/                     # 模型输出目录（自动创建）
├── .gitignore                   # Git 忽略规则
├── README.md                    # 项目说明文档
└── requirements.txt             # Python 依赖列表
```

## 安装步骤

### 环境要求

- Python **3.8 - 3.12**（Python 3.13 暂不支持 PyTorch）
- CUDA >= 11.0（GPU 训练可选）

### 1. 克隆仓库

```bash
git clone <repository-url>
cd Jigsaw_Toxicity_Prediction
```

### 2. 创建虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备数据

将 `train.csv` 和 `test.csv` 放置于 `data/` 目录下。数据文件应包含以下列：

- `id`：评论唯一标识
- `comment_text`：评论文本内容
- `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`：毒性标签（0/1）

## 使用方法

### 训练 DistilRoBERTa 模型

```bash
python -m src.scripts.train_roberta
```

训练过程中会自动保存最优模型至 `outputs/best_model.bin`，并生成损失曲线图 `outputs/roberta_loss.png` 和提交文件 `outputs/submission.csv`。

### 训练 TextCNN 模型

```bash
python -m src.scripts.train_cnn
```

训练完成后会生成：
- `outputs/textcnn_metrics.png`（包含训练/验证损失曲线和准确率曲线）
- `outputs/submission_cnn.csv`（预测提交文件）

### 运行单元测试

```bash
python test_project.py
```

测试覆盖配置模块、数据加载、数据集类、模型定义、训练函数等核心功能，包含正常流程和边界条件测试。

### 自定义配置

修改 `src/config.py` 中的配置类可调整超参数：

```python
class RobertaConfig(Config):
    MODEL_NAME = "distilroberta-base"
    TRAIN_BATCH_SIZE = 128
    VALID_BATCH_SIZE = 128
    MAX_LENGTH = 64
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DROPOUT = 0.2

class TextCNNConfig(Config):
    EMBEDDING_DIM = 100
    MAX_VOCAB_SIZE = 20000
    N_FILTERS = 100
    FILTER_SIZES = [3, 4, 5]
    HIDDEN_DIM = 128
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 0.001
    DROPOUT = 0.5
```

## 功能说明

- **数据清洗**：自动填充缺失文本，过滤标签为 -1 的无效样本
- **分层采样**：训练/验证集划分采用分层抽样，保证标签分布一致
- **多标签分类**：每个评论可同时属于多个毒性类别
- **模型保存**：自动保存验证损失最低的模型权重
- **可视化**：自动生成训练损失与准确率曲线
- **结果提交**：自动生成符合竞赛格式的 CSV 提交文件
- **单元测试**：完整的测试套件覆盖核心功能模块

## 技术实现

### DistilRoBERTa 模型

基于 HuggingFace `transformers` 库的预训练语言模型，通过 fine-tuning 适配多标签分类任务。模型输出层使用 `BCEWithLogitsLoss` 损失函数。

### TextCNN 模型

轻量级卷积神经网络，包含：
- Embedding 层：词向量表示
- 多尺度卷积层：3、4、5 窗口大小
- 最大池化层：提取关键特征
- 全连接层：输出分类概率

## 贡献指南

欢迎提交 Issue 和 Pull Request。

1. Fork 本仓库
2. 创建功能分支：`git checkout -b feature/xxx`
3. 提交修改：`git commit -m "feat: add xxx"`
4. 推送分支：`git push origin feature/xxx`
5. 创建 Pull Request

请确保代码通过语法检查和单元测试，并保持与现有代码风格一致。

## 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
