import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time

from src.config import TextCNNConfig
from src.data.loader import load_data, split_data
from src.data.dataset import build_vocab, text_to_sequence, ToxicityDataset
from src.models.textcnn import TextCNN
from src.utils.trainer import train_epoch_cnn, evaluate_cnn

def main():
    config = TextCNNConfig()
    print("Device:", config.DEVICE)
    train_df, test_df, sample_sub = load_data(config)
    print("Train samples:", len(train_df), "Test samples:", len(test_df))
    
    vocab = build_vocab(train_df["comment_text"].astype(str), config.MAX_VOCAB_SIZE)
    print("Vocabulary size:", len(vocab))
    
    train_df["input_ids"] = train_df["comment_text"].astype(str).apply(lambda x: text_to_sequence(x, vocab, config.MAX_LENGTH))
    test_df["input_ids"] = test_df["comment_text"].astype(str).apply(lambda x: text_to_sequence(x, vocab, config.MAX_LENGTH))
    
    train_data, val_data = split_data(train_df, config.TARGET_COLS, test_size=0.1, random_state=42)
    
    train_dataset = ToxicityDataset(train_data, config.TARGET_COLS)
    val_dataset = ToxicityDataset(val_data, config.TARGET_COLS)
    test_dataset = ToxicityDataset(test_df, config.TARGET_COLS, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE)
    
    model = TextCNN(vocab_size=len(vocab), embed_dim=config.EMBEDDING_DIM, num_classes=len(config.TARGET_COLS), n_filters=config.N_FILTERS, filter_sizes=config.FILTER_SIZES, dropout=config.DROPOUT).to(config.DEVICE)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.BCELoss()
    
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    print("Training TextCNN...")
    for epoch in range(config.EPOCHS):
        start_time = time.time()
        train_loss = train_epoch_cnn(model, train_loader, optimizer, criterion, config.DEVICE)
        val_loss, val_acc = evaluate_cnn(model, val_loader, criterion, config.DEVICE)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
        epoch_mins, epoch_secs = divmod(int(time.time() - start_time), 60)
        print("Epoch:", str(epoch + 1).zfill(2), "|", "Time:", epoch_mins, "m", epoch_secs, "s")
        print("	Train Loss:", "{:.3f}".format(train_loss))
        print("	Val. Loss:", "{:.3f}".format(val_loss), "|", "Val. Acc:", "{:.2f}".format(val_acc * 100), "%")
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, config.EPOCHS + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, config.EPOCHS + 1), val_losses, label="Validation Loss", marker="s")
    plt.title("TextCNN Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, config.EPOCHS + 1), val_accuracies, label="Validation Accuracy", color="green", marker="^")
    plt.title("TextCNN Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.OUTPUT_DIR, "textcnn_metrics.png"))
    plt.close()
    
    print("Generating submission file...")
    model.eval()
    predictions = []
    with torch.no_grad():
        for texts in test_loader:
            texts = texts.to(config.DEVICE)
            preds = model(texts)
            predictions.extend(preds.cpu().numpy())
    
    sub_df = pd.read_csv(config.SAMPLE_SUB)
    sub_df[config.TARGET_COLS] = predictions
    sub_df.to_csv(os.path.join(config.OUTPUT_DIR, "submission_cnn.csv"), index=False)
    print("Submission file saved:", os.path.join(config.OUTPUT_DIR, "submission_cnn.csv"))

if __name__ == "__main__":
    main()
