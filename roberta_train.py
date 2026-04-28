import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import warnings

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']  # 防止中文乱码
plt.rcParams['axes.unicode_minus'] = False

warnings.filterwarnings("ignore")


# ==========================================
# 1. 配置参数
# ==========================================
class Config:
    MODEL_NAME = "distilroberta-base"
    TRAIN_FILE = "/kaggle/input/datasets/lik586/111111/train.csv"  # 训练集
    TEST_FILE = "/kaggle/input/datasets/lik586/111111/test.csv"  # 测试集
    SAMPLE_SUB = "/kaggle/input/datasets/lik586/111111/sample_submission.csv"  # 提交样例
    OUTPUT_FILE = "/kaggle/working/submission.csv"  # 提交文件

    MAX_LENGTH = 64
    TRAIN_BATCH_SIZE = 256
    VALID_BATCH_SIZE = 256
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 目标列名
    TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


config = Config()


# ==========================================
# 2. 数据处理
# ==========================================
def clean_data(df, is_train=True):
    """
    清洗数据：
    1. 处理缺失值
    2. 过滤掉标签为 -1 的行 (Jigsaw 数据集特性)
    """
    # 填充空文本
    text_col = 'comment_text' if 'comment_text' in df.columns else 'text'
    df[text_col] = df[text_col].fillna(" ")

    # 如果是训练集或验证集，过滤 -1
    if is_train and all(col in df.columns for col in config.TARGET_COLS):
        # 只要任意一个标签是 -1，就丢弃该行
        mask = (df[config.TARGET_COLS] != -1).all(axis=1)
        df = df[mask].reset_index(drop=True)

    return df


print("正在加载数据...")
# 加载上传的四个文件
train_df = pd.read_csv(config.TRAIN_FILE)
test_df = pd.read_csv(config.TEST_FILE)
sample_sub = pd.read_csv(config.SAMPLE_SUB)
# 数据清洗
train_df = clean_data(train_df, is_train=True)

print(f"训练集有效行数: {len(train_df)}")
print(f"测试集行数: {len(test_df)}")
print(f"设备: {config.DEVICE}")

# ==========================================
# 3. Dataset 类
# ==========================================
class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_test = is_test

        # 确定文本列名
        self.text_col = 'comment_text' if 'comment_text' in df.columns else 'text'
        self.texts = df[self.text_col].values

        if not is_test:
            self.targets = df[config.TARGET_COLS].values
        else:
            self.ids = df['id'].values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        if not self.is_test:
            targets = self.targets[idx]
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(targets, dtype=torch.float)
            }
        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'ids': self.ids[idx]
            }

# ==========================================
# 4. 模型定义
# ==========================================
class JigsawModel(nn.Module):
    def __init__(self, n_labels):
        super(JigsawModel, self).__init__()
        # 使用 AutoModel 获取基础输出，然后自己加分类头
        self.bert = AutoModel.from_pretrained(config.MODEL_NAME)
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(self.bert.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # 使用 pooler_output 进行分类
        x = self.drop(outputs.pooler_output)
        return self.out(x)


# ==========================================
# 5. 训练与评估函数
# ==========================================
def train_fn(dataloader, model, optimizer, scheduler, device):
    model.train()
    losses = []
    loop = tqdm(dataloader, desc="Training")

    for d in loop:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        targets = d['targets'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, attention_mask)

        # 多标签分类使用 BCEWithLogitsLoss
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, targets)

        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        scheduler.step()

        loop.set_postfix(loss=loss.item())

    # --- 新增：返回平均 Loss ---
    return np.mean(losses)
    # --- 新增结束 ---


def eval_fn(dataloader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []

    with torch.no_grad():
        for d in tqdm(dataloader, desc="Evaluating"):
            input_ids = d['input_ids'].to(device)
            attention_mask = d['attention_mask'].to(device)
            targets = d['targets'].to(device)

            outputs = model(input_ids, attention_mask)

            fin_targets.extend(targets.cpu().detach().numpy())
            # 输出概率 (Sigmoid)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy())

    return np.array(fin_outputs), np.array(fin_targets)


# ==========================================
# 6. 主程序执行
# ==========================================
def run():
    # 1. 划分训练/验证集 (从 train.csv 中划分)
    # 使用 stratify 确保有毒/无毒样本比例均衡
    df_train, df_valid = train_test_split(
        train_df,
        test_size=0.2,
        random_state=42,
        stratify=train_df[config.TARGET_COLS].max(axis=1).values  # 只要有任意一个标签为1就算有毒
    )

    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

    train_dataset = JigsawDataset(df_train, tokenizer, config.MAX_LENGTH, is_test=False)
    valid_dataset = JigsawDataset(df_valid, tokenizer, config.MAX_LENGTH, is_test=False)
    test_dataset = JigsawDataset(test_df, tokenizer, config.MAX_LENGTH, is_test=True)

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=0)

    model = JigsawModel(n_labels=len(config.TARGET_COLS))
    model.to(config.DEVICE)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    best_loss = float('inf')

    # --- 新增：在循环前创建列表 ---
    roberta_train_losses = []  # 注意加前缀，防止和CNN的变量名冲突
    roberta_val_losses = []
    # --- 新增结束 ---

    print("开始训练...")
    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")
        train_fn(train_loader, model, optimizer, scheduler, config.DEVICE)

        # 验证
        outputs, targets = eval_fn(valid_loader, model, config.DEVICE)

        # 计算简单的验证损失 (MSE)
        val_loss = np.mean((outputs - targets) ** 2)
        print(f"Validation Loss: {val_loss}")

        # --- 新增：记录训练 Loss ---
        train_loss = train_fn(train_loader, model, optimizer, scheduler, config.DEVICE)  # 接收返回值
        roberta_val_losses.append(val_loss)
        # --- 新增结束 ---

        if val_loss < best_loss:
            best_loss = val_loss
            # 保存最佳模型
            torch.save(model.state_dict(), "best_model.bin")
            print(f"Model Saved with loss: {best_loss}")

    # --- 新增：画图代码  ---

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, config.EPOCHS + 1), roberta_val_losses, label='RoBERTa 验证 Loss', marker='o', color='orange')
    plt.title('DistilRoBERTa 模型 Loss 变化曲线')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('/kaggle/working/roberta_loss.png')
    plt.close()
    print("RoBERTa 训练图表已生成：/kaggle/working/roberta_loss.png")
    # --- 新增结束 ---

    # ==========================================
    # 7. 预测与提交
    # ==========================================
    print("开始预测测试集...")
    # 加载最佳模型
    model.load_state_dict(torch.load("best_model.bin"))
    model.eval()

    predictions = []
    with torch.no_grad():
        for d in tqdm(test_loader, desc="Predicting"):
            input_ids = d['input_ids'].to(config.DEVICE)
            attention_mask = d['attention_mask'].to(config.DEVICE)
            ids = d['ids']

            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().detach().numpy()

            for i, prob in enumerate(probs):
                predictions.append({
                    'id': ids[i],
                    'toxic': prob[0],
                    'severe_toxic': prob[1],
                    'obscene': prob[2],
                    'threat': prob[3],
                    'insult': prob[4],
                    'identity_hate': prob[5]
                })

    # 生成提交文件
    sub_df = pd.DataFrame(predictions)
    sub_df = sub_df[['id'] + config.TARGET_COLS]
    sub_df.to_csv(config.OUTPUT_FILE, index=False)
    print(f"提交文件已生成: {config.OUTPUT_FILE}")


if __name__ == "__main__":
    run()