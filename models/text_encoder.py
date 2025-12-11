import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizerFast

class BertTextEncoder(nn.Module):
    """
    BERT-based text encoder.
    Returns an L2-normalized sentence embedding (CLS token).
    """
    def __init__(self, model_name: str = "bert-base-uncased", output_dim: int = 768, freeze_bert: bool = False):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.output_dim = output_dim

        # Optional projection to a smaller dim if you want (e.g., 512)
        if output_dim != self.bert.config.hidden_size:
            self.proj = nn.Linear(self.bert.config.hidden_size, output_dim)
        else:
            self.proj = nn.Identity()

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

    def encode_texts(self, texts, device):
        """
        texts: list of strings
        returns: tensor [B, output_dim]
        """
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls = outputs.last_hidden_state[:, 0, :]  # [B, H]
        feats = self.proj(cls)  # [B, output_dim]
        feats = torch.nn.functional.normalize(feats, p=2, dim=-1)
        return feats

    def forward(self, texts, device=None):
        if device is None:
            device = next(self.parameters()).device
        return self.encode_texts(texts, device)
