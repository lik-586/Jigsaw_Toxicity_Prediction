import torch
from torch.utils.data import Dataset
from collections import Counter


class JigsawDataset(Dataset):
    def __init__(self, df, tokenizer, max_length, target_cols, is_test=False):
        self.df = df
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.target_cols = target_cols
        self.is_test = is_test

        self.text_col = 'comment_text' if 'comment_text' in df.columns else 'text'
        self.texts = df[self.text_col].values

        if not is_test:
            self.targets = df[target_cols].values
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


def build_vocab(texts, max_size):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(max_size - 2):
        vocab[word] = len(vocab)
    return vocab


def text_to_sequence(text, vocab, max_len):
    sequence = [vocab.get(word, 1) for word in text.split()]
    if len(sequence) < max_len:
        sequence += [0] * (max_len - len(sequence))
    else:
        sequence = sequence[:max_len]
    return sequence


class ToxicityDataset(Dataset):
    def __init__(self, df, target_cols, is_test=False):
        self.texts = df['input_ids'].tolist()
        self.target_cols = target_cols
        self.is_test = is_test
        if not is_test:
            self.labels = df[target_cols].values.astype(float)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = torch.tensor(self.texts[idx], dtype=torch.long)
        if self.is_test:
            return text
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float)
            return text, label
