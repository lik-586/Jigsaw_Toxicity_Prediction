import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from src.config import RobertaConfig
from src.data.loader import load_data, split_data
from src.data.dataset import JigsawDataset
from src.models.roberta import JigsawModel
from src.utils.trainer import train_fn_roberta, eval_fn_roberta

def main():
    config = RobertaConfig()
    print("Device:", config.DEVICE)
    train_df, test_df, sample_sub = load_data(config)
    print("Train samples:", len(train_df), "Test samples:", len(test_df))
    df_train, df_valid = split_data(train_df, config.TARGET_COLS)
    
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    train_dataset = JigsawDataset(df_train, tokenizer, config.MAX_LENGTH, config.TARGET_COLS, is_test=False)
    valid_dataset = JigsawDataset(df_valid, tokenizer, config.MAX_LENGTH, config.TARGET_COLS, is_test=False)
    test_dataset = JigsawDataset(test_df, tokenizer, config.MAX_LENGTH, config.TARGET_COLS, is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config.VALID_BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = JigsawModel(config.MODEL_NAME, len(config.TARGET_COLS), config.DROPOUT)
    model.to(config.DEVICE)
    
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)
    total_steps = len(train_loader) * config.EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    best_loss = float("inf")
    val_losses = []
    
    print("Training started...")
    for epoch in range(config.EPOCHS):
        print("Epoch", epoch + 1, "/", config.EPOCHS)
        train_fn_roberta(train_loader, model, optimizer, scheduler, config.DEVICE)
        outputs, targets = eval_fn_roberta(valid_loader, model, config.DEVICE)
        val_loss = np.mean((outputs - targets) ** 2)
        val_losses.append(val_loss)
        print("Validation Loss:", val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            os.makedirs(config.OUTPUT_DIR, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(config.OUTPUT_DIR, "best_model.bin"))
            print("Model saved with loss:", best_loss)
    
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, config.EPOCHS + 1), val_losses, label="RoBERTa Validation Loss", marker="o", color="orange")
    plt.title("DistilRoBERTa Model Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.OUTPUT_DIR, "roberta_loss.png"))
    plt.close()
    
    print("Predicting on test set...")
    model.load_state_dict(torch.load(os.path.join(config.OUTPUT_DIR, "best_model.bin")))
    model.eval()
    
    predictions = []
    with torch.no_grad():
        for d in tqdm(test_loader, desc="Predicting"):
            input_ids = d["input_ids"].to(config.DEVICE)
            attention_mask = d["attention_mask"].to(config.DEVICE)
            ids = d["ids"]
            outputs = model(input_ids, attention_mask)
            probs = torch.sigmoid(outputs).cpu().detach().numpy()
            for i, prob in enumerate(probs):
                predictions.append({"id": ids[i], "toxic": prob[0], "severe_toxic": prob[1], "obscene": prob[2], "threat": prob[3], "insult": prob[4], "identity_hate": prob[5]})
    
    sub_df = pd.DataFrame(predictions)
    sub_df = sub_df[["id"] + config.TARGET_COLS]
    sub_df.to_csv(os.path.join(config.OUTPUT_DIR, "submission.csv"), index=False)
    print("Submission file saved:", os.path.join(config.OUTPUT_DIR, "submission.csv"))

if __name__ == "__main__":
    main()
