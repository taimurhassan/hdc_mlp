import os
import itertools

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.text_encoder import BertTextEncoder
from models.hdc_mlp import HDCMLPEncoder
from data.oct_dataset import OCTDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHOLOGY_LABELS = ["AMD", "DME", "NORMAL"]
LESION_VOCAB = ["drusen", "retinal fluid", "no lesion"]


def make_prompt(label: str, lesion: str):
    return f"The OCT sample majorly contains {lesion} and represents {label}."


def pearson_corr(v, t):
    """
    v, t: 1D tensors (already truncated to same length)
    """
    v = v - v.mean()
    t = t - t.mean()
    num = (v * t).sum()
    denom = (v.pow(2).sum().sqrt() * t.pow(2).sum().sqrt() + 1e-8)
    return num / denom


def precompute_text_features(text_encoder):
    """
    Precompute text features for all lesion-label combinations.
    Returns dict[(lesion,label)] -> feature tensor [D]
    """
    combos = list(itertools.product(LESION_VOCAB, PATHOLOGY_LABELS))
    prompts = [make_prompt(lbl, les) for (les, lbl) in combos]
    feats = text_encoder(prompts, device=DEVICE)  # [Z*C, D]
    mapping = {}
    for (les, lbl), feat in zip(combos, feats):
        mapping[(les, lbl)] = feat.detach()
    return mapping


def run_inference_for_dataset(
    dataset_name: str,
    root: str,
    text_encoder,
    hdc_mlp,
    text_feat_map,
):
    """
    Run LHDM-based inference for a single dataset (test split).
    """
    if not os.path.isdir(root):
        print(f"[{dataset_name}] Root directory not found: {root}. Skipping.")
        return

    try:
        test_dataset = OCTDataset(
            root_dir=root,
            split="test",
            img_size=224,
            augment=False
        )
    except RuntimeError as e:
        print(f"[{dataset_name}] Skipping: {e}")
        return

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4
    )

    correct = 0
    total = 0

    hdc_mlp.eval()

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc=f"Testing {dataset_name}"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            visual_feats = hdc_mlp(images)  # [B, D]

            preds = []
            for vf in visual_feats:
                best_corr = -1e9
                best_label = None
                for les in LESION_VOCAB:
                    for lbl in PATHOLOGY_LABELS:
                        tf = text_feat_map[(les, lbl)]
                        # Align lengths
                        n = min(vf.shape[0], tf.shape[0])
                        r = pearson_corr(vf[:n], tf[:n])
                        if r > best_corr:
                            best_corr = r
                            best_label = lbl
                preds.append(PATHOLOGY_LABELS.index(best_label))

            preds = torch.tensor(preds, device=DEVICE)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total if total > 0 else 0.0
    print(f"[{dataset_name}] Accuracy (LHDM correlation) = {acc:.4f}")


def main(
    zhang_root: str = "data/Zhang",
    duke_root: str = "data/Duke",
    rabbani_root: str = "data/Rabbani",
    biomisa_root: str = "data/BIOMISA",
):
    # 1. Load pre-trained encoders
    text_encoder = BertTextEncoder(
        model_name="bert-base-uncased",
        output_dim=768,
        freeze_bert=True
    )
    text_encoder.load_state_dict(
        torch.load("checkpoints/text_encoder_bert_pretrained.pth", map_location="cpu")
    )
    text_encoder.to(DEVICE)
    text_encoder.eval()

    hdc_mlp = HDCMLPEncoder(
        cnn_name="resnet18",
        hd_dim=4096,
        text_feat_dim=768,
        pretrained_backbone=False,
    )
    hdc_mlp.load_state_dict(
        torch.load("checkpoints/hdc_mlp_flash_la.pth", map_location="cpu")
    )
    hdc_mlp.to(DEVICE)
    hdc_mlp.eval()

    # 2. Precompute lesion-label text embeddings
    text_feat_map = precompute_text_features(text_encoder)

    # 3. Run over available datasets
    datasets = {
        "Zhang": zhang_root,
        "Duke": duke_root,
        "Rabbani": rabbani_root,
        "BIOMISA": biomisa_root,
    }

    for name, root in datasets.items():
        run_inference_for_dataset(
            dataset_name=name,
            root=root,
            text_encoder=text_encoder,
            hdc_mlp=hdc_mlp,
            text_feat_map=text_feat_map,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LHDM inference for FLASH-HDC OCT model.")
    parser.add_argument("--zhang_root", type=str, default="data/Zhang")
    parser.add_argument("--duke_root", type=str, default="data/Duke")
    parser.add_argument("--rabbani_root", type=str, default="data/Rabbani")
    parser.add_argument("--biomisa_root", type=str, default="data/BIOMISA")
    args = parser.parse_args()

    main(
        zhang_root=args.zhang_root,
        duke_root=args.duke_root,
        rabbani_root=args.rabbani_root,
        biomisa_root=args.biomisa_root,
    )
