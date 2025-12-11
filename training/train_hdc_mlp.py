import random
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from models.text_encoder import BertTextEncoder
from models.hdc_mlp import HDCMLPEncoder
from data.oct_dataset import OCTDataset
from loss.la_loss import la_loss

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHOLOGY_LABELS = ["AMD", "DME", "NORMAL"]
LESION_VOCAB = {
    "AMD": "drusen",
    "DME": "retinal fluid",
    "NORMAL": "no lesion",
}


def make_prompt(label: str, lesion: str):
    return f"The OCT sample majorly contains {lesion} and represents {label}."


def build_pos_neg_prompts(labels):
    """
    labels: tensor [B] with idx in {0,1,2}
    returns:
      pos_prompts: list[str]
      neg_prompts: list[str]
    """
    pos_prompts = []
    neg_prompts = []

    for idx in labels.tolist():
        pos_label = PATHOLOGY_LABELS[idx]
        pos_lesion = LESION_VOCAB[pos_label]
        pos_prompts.append(make_prompt(pos_label, pos_lesion))

        # Choose a negative class
        neg_choices = [l for l in PATHOLOGY_LABELS if l != pos_label]
        neg_label = random.choice(neg_choices)
        neg_lesion = LESION_VOCAB[neg_label]
        neg_prompts.append(make_prompt(neg_label, neg_lesion))

    return pos_prompts, neg_prompts


def main(zhang_root: str = "data/Zhang"):
    # 1. Load pre-trained text encoder (BERT) and freeze it
    text_encoder = BertTextEncoder(
        model_name="bert-base-uncased",
        output_dim=768,
        freeze_bert=False
    )
    text_encoder.load_state_dict(
        torch.load("checkpoints/text_encoder_bert_pretrained.pth", map_location="cpu")
    )
    text_encoder.to(DEVICE)

    for p in text_encoder.parameters():
        p.requires_grad = False

    text_encoder.eval()

    # 2. Create HDC+MLP encoder (FLASH-based)
    hdc_mlp = HDCMLPEncoder(
        cnn_name="resnet18",
        hd_dim=4096,
        text_feat_dim=768,
        pretrained_backbone=True,
    ).to(DEVICE)

    # 3. Dataset: train on Zhang train split
    train_dataset = OCTDataset(
        root_dir=zhang_root,
        split="train",
        img_size=224,
        augment=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )

    optimizer = torch.optim.Adam(hdc_mlp.parameters(), lr=1e-4)
    num_epochs = 150
    tau = 1.5  # temperature

    for epoch in range(num_epochs):
        hdc_mlp.train()
        running_loss = 0.0
        num_samples = 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Visual embeddings (anchors)
            anchors = hdc_mlp(images)  # [B, D]

            # Positive & negative prompts
            pos_prompts, neg_prompts = build_pos_neg_prompts(labels)

            # Textual features
            with torch.no_grad():
                pos_feats = text_encoder(pos_prompts, device=DEVICE)  # [B, D]
                neg_feats = text_encoder(neg_prompts, device=DEVICE)  # [B, D]

            loss = la_loss(anchors, pos_feats, neg_feats, tau=tau)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            num_samples += images.size(0)

        print(f"Epoch {epoch+1}: La loss = {running_loss / num_samples:.4f}")

    import os
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(hdc_mlp.state_dict(), "checkpoints/hdc_mlp_flash_la.pth")
    print("Saved HDC+MLP (FLASH) to checkpoints/hdc_mlp_flash_la.pth")


if __name__ == "__main__":
    main()
