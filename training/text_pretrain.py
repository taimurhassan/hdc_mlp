import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.text_encoder import BertTextEncoder

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PromptDataset(Dataset):
    """
    Expect a CSV with columns: text,label
    label in {AMD, DME, NORMAL}
    """
    def __init__(self, csv_path):
        self.texts = []
        self.labels = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.texts.append(row["text"])
                self.labels.append(row["label"])
        label_to_idx = {"AMD": 0, "DME": 1, "NORMAL": 2}
        self.labels = [label_to_idx[l] for l in self.labels]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


def collate_fn(batch):
    texts, labels = zip(*batch)
    labels = torch.tensor(labels, dtype=torch.long)
    return list(texts), labels


def main(csv_path: str = "data/clinical_prompts_10k.csv"):
    dataset = PromptDataset(csv_path)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = BertTextEncoder(
        model_name="bert-base-uncased",
        output_dim=768,
        freeze_bert=False
    ).to(DEVICE)
    clf_head = nn.Linear(768, 3).to(DEVICE)  # 3 disease classes

    optim = torch.optim.Adam(
        list(model.parameters()) + list(clf_head.parameters()),
        lr=2e-5
    )
    criterion = nn.CrossEntropyLoss()

    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        clf_head.train()
        total_loss = 0.0
        correct = 0
        total = 0
        for texts, labels in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            labels = labels.to(DEVICE)

            feats = model(texts, device=DEVICE)  # [B, 768]
            logits = clf_head(feats)            # [B, 3]

            loss = criterion(logits, labels)
            optim.zero_grad()
            loss.backward()
            optim.step()

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        print(f"Epoch {epoch+1}: loss={total_loss/total:.4f}, acc={correct/total:.4f}")

    # Save pre-trained encoder (without classifier)
    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/text_encoder_bert_pretrained.pth")
    print("Saved text encoder to checkpoints/text_encoder_bert_pretrained.pth")


if __name__ == "__main__":
    main()
