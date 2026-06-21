import torch
import torch.nn as nn
from transformers import AutoModel

class JigsawModel(nn.Module):
    def __init__(self, model_name, n_labels, dropout=0.2):
        super(JigsawModel, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.drop = nn.Dropout(p=dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, n_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = self.drop(outputs.pooler_output)
        return self.out(x)

