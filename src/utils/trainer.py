import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train_fn_roberta(dataloader, model, optimizer, scheduler, device):
    model.train()
    losses = []
    loop = tqdm(dataloader, desc="Training")
    for d in loop:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(outputs, targets)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        scheduler.step()
        loop.set_postfix(loss=loss.item())
    return np.mean(losses)

def eval_fn_roberta(dataloader, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    with torch.no_grad():
        for d in tqdm(dataloader, desc="Evaluating"):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)
            outputs = model(input_ids, attention_mask)
            fin_targets.extend(targets.cpu().detach().numpy())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy())
    return np.array(fin_outputs), np.array(fin_targets)

def train_epoch_cnn(model, loader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    for texts, labels in loader:
        texts, labels = texts.to(device), labels.to(device)
        optimizer.zero_grad()
        predictions = model(texts)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

def evaluate_cnn(model, loader, criterion, device):
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for texts, labels in loader:
            texts, labels = texts.to(device), labels.to(device)
            predictions = model(texts)
            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            all_preds.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    binary_preds = (all_preds >= 0.5).astype(int)
    acc = accuracy_score(all_labels.flatten(), binary_preds.flatten())
    return epoch_loss / len(loader), acc