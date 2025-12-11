import torch
from PIL import Image
from torchvision import transforms

from models.text_encoder import BertTextEncoder
from models.hdc_mlp import HDCMLPEncoder
from inference.lhdm_infer import make_prompt, pearson_corr, LESION_VOCAB, PATHOLOGY_LABELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Load models
text_encoder = BertTextEncoder("bert-base-uncased", 768, freeze_bert=True)
text_encoder.load_state_dict(torch.load("checkpoints/text_encoder_bert_pretrained.pth", map_location="cpu"))
text_encoder.to(DEVICE).eval()

hdc_mlp = HDCMLPEncoder("resnet18", hd_dim=4096, text_feat_dim=768, pretrained_backbone=False)
hdc_mlp.load_state_dict(torch.load("checkpoints/hdc_mlp_flash_la.pth", map_location="cpu"))
hdc_mlp.to(DEVICE).eval()

# 2. Precompute text features
prompts = []
pairs = []
for les in LESION_VOCAB:
    for lbl in PATHOLOGY_LABELS:
        prompts.append(make_prompt(lbl, les))
        pairs.append((les, lbl))

text_feats = text_encoder(prompts, device=DEVICE)

# 3. Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

img = Image.open("path/to/your/oct_image.png").convert("RGB")
img_t = transform(img).unsqueeze(0).to(DEVICE)

# 4. Visual feature
with torch.no_grad():
    v = hdc_mlp(img_t)[0]  # [D]

# 5. Find best match
best_corr = -1e9
best_label = None
for feat, (les, lbl) in zip(text_feats, pairs):
    n = min(v.shape[0], feat.shape[0])
    r = pearson_corr(v[:n], feat[:n])
    if r > best_corr:
        best_corr = r
        best_label = lbl

print("Predicted class:", best_label)
