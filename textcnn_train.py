import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans'] # 防止中文乱码
plt.rcParams['axes.unicode_minus'] = False

# ==============================
# 1. 配置参数
# ==============================
class Config:
    # --- 路径配置 (适配 Kaggle) ---
    TRAIN_FILE = "/kaggle/input/competitions/jigsaw-toxic-comment-classification-challenge/train.csv.zip"
    TEST_FILE = "/kaggle/input/competitions/jigsaw-toxic-comment-classification-challenge/test.csv.zip"
    SAMPLE_SUB = "/kaggle/input/competitions/jigsaw-toxic-comment-classification-challenge/sample_submission.csv.zip"
    OUTPUT_FILE = "/kaggle/working/submission_cnn.csv"  # 区分文件名

    # --- 模型超参数 ---
    EMBEDDING_DIM = 100  # 词向量维度
    MAX_VOCAB_SIZE = 20000  # 词表大小
    MAX_LENGTH = 64  # 序列长度
    N_FILTERS = 100  # 卷积核数量
    FILTER_SIZES = [3, 4, 5]  # 卷积核尺寸
    HIDDEN_DIM = 128  # 全连接层维度
    DROPOUT = 0.5
    BATCH_SIZE = 64
    EPOCHS = 5  # CNN收敛快，可以多跑几轮
    LEARNING_RATE = 0.001

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TARGET_COLS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


config = Config()


# ==============================
# 2. 数据处理与词表构建
# ==============================
def build_vocab(texts, max_size):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 1) for word in text.split()]  # 1 is <UNK>
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))  # 0 is <PAD>
    else:
        sequence = sequence[:max_len]
    return sequence


print("正在加载数据并构建词表...")
train_df = pd.read_csv(config.TRAIN_FILE)
test_df = pd.read_csv(config.TEST_FILE)

# 构建词表
vocab = build_vocab(train_df['comment_text'].astype(str), config.MAX_VOCAB_SIZE)
print(f"词表大小: {len(vocab)}")

# 转换数据
train_df['input_ids'] = train_df['comment_text'].astype(str).apply(
    lambda x: text_to_sequence(x, vocab, config.MAX_LENGTH))
test_df['input_ids'] = test_df['comment_text'].astype(str).apply(
    lambda x: text_to_sequence(x, vocab, config.MAX_LENGTH))

# 划分训练集和验证集
train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=42)


# ==============================
# 3. Dataset 定义
# ==============================
class ToxicityDataset(Dataset):
    def __init__(self, df, is_test=False):
        self.texts = df['input_ids'].tolist()
        self.is_test = is_test
        if not is_test:
            self.labels = df[config.TARGET_COLS].values.astype(float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        if self.is_test:
            return text
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return text, label


train_dataset = ToxicityDataset(train_data)
val_dataset = ToxicityDataset(val_data)
test_dataset = ToxicityDataset(test_df, is_test=True)

train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)


# ==============================
# 4. TextCNN 模型定义
# ==============================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, n_filters, filter_sizes, dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        # 卷积层列表
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=n_filters, kernel_size=(fs, embed_dim))
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * n_filters, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [Batch, Seq Len]
        x = self.embedding(x)  # [Batch, Seq Len, Embed Dim]
        x = x.unsqueeze(1)  # [Batch, 1, Seq Len, Embed Dim]

        # 卷积 + 激活 + 池化
        conved = [torch.relu(conv(x)).squeeze(3) for conv in self.convs]
        # conved: List of [Batch, N Filters, Seq Len - Filter Size + 1]

        pooled = [torch.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled: List of [Batch, N Filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat: [Batch, N Filters * Len(Filter Sizes)]

        return torch.sigmoid(self.fc(cat))


# 初始化模型
model = TextCNN(
    vocab_size=len(vocab),
    embed_dim=config.EMBEDDING_DIM,
    num_classes=len(config.TARGET_COLS),
    n_filters=config.N_FILTERS,
    filter_sizes=config.FILTER_SIZES,
    dropout=config.DROPOUT
).to(config.DEVICE)

optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
criterion = nn.BCELoss()


# ==============================
# 5. 训练循环
# ==============================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    return epoch_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(config.DEVICE), labels.to(config.DEVICE)
            predictions = model(texts)
            loss = criterion(predictions, labels)

            epoch_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    # 简单的准确率计算 (阈值0.5)
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    binary_preds = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels.flatten(), binary_preds.flatten())

    return epoch_loss / len(loader), acc


print("开始训练 TextCNN...")
best_val_loss = float('inf')

# --- 新增：创建列表来存储数据 ---
train_losses = []
val_losses = []
val_accuracies = []
# --- 新增结束 ---

for epoch in range(config.EPOCHS):
    start_time = time.time()
    train_loss = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_acc = evaluate(model, val_loader, criterion)
    # --- 新增：保存数据 ---
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    # --- 新增结束 ---

    if val_loss < best_val_loss:
        best_val_loss = val_loss

    epoch_mins, epoch_secs = divmod(int(time.time() - start_time), 60)
    print(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f}')
    print(f'\tVal. Loss: {val_loss:.3f} | Val. Acc: {val_acc * 100:.2f}%')

# --- 新增：画图代码 ---
plt.figure(figsize=(12, 5))

# 子图1：Loss 曲线
plt.subplot(1, 2, 1)
plt.plot(range(1, config.EPOCHS+1), train_losses, label='训练 Loss', marker='o')
plt.plot(range(1, config.EPOCHS+1), val_losses, label='验证 Loss', marker='s')
plt.title('TextCNN 模型 Loss 变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# 子图2：Accuracy 曲线
plt.subplot(1, 2, 2)
plt.plot(range(1, config.EPOCHS+1), val_accuracies, label='验证 Accuracy', color='green', marker='^')
plt.title('TextCNN 模型 Accuracy 变化曲线')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# 保存图片
plt.tight_layout()
plt.savefig('/kaggle/working/textcnn_metrics.png') # 保存到输出目录
plt.close() # 关闭画布，防止内存泄漏
print("TextCNN 训练图表已生成：/kaggle/working/textcnn_metrics.png")
# --- 新增结束 ---

# ==============================
# 6. 生成提交文件
# ==============================
print("生成提交文件...")
model.eval()
predictions = []
with torch.no_grad():
    for texts in test_loader:
        texts = texts.to(config.DEVICE)
        preds = model(texts)
        predictions.extend(preds.cpu().numpy())

sub_df = pd.read_csv(config.SAMPLE_SUB)
sub_df[config.TARGET_COLS] = predictions
sub_df.to_csv(config.OUTPUT_FILE, index=False)
print(f"提交文件已保存至: {config.OUTPUT_FILE}")