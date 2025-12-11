# Language Assisted Learnable Hyperdimensional Computing Framework for Retinal Disease Classification

## Introduction
This repository contains the implementation of the language assisted learnable hyperdimensional computing framework for retinal disease classification. 

![TST](image.png)

The pipeline consists of three main stages:

1. **Text Encoder Pretraining** — trains a medical language encoder using clinically validated OCT prompts.  
2. **HDC Encoder Training** — trains a hyperdimensional visual encoder using OCT datasets.  
3. **Inference via Lesion-aware Hyperdimensional Matching (LHDM)** — predicts retinal disease labels by correlating visual hypervectors with text hypervectors.

This repository includes all required training scripts, inference pipeline, HDC and text encoders, La loss implementation, dataset loader, and a unified entry script (`main.py`).

---

# 1. Installation

## 1.1. Python Environment

Use **Python 3.7+**.

```bash
python -m venv venv
source venv/bin/activate     # Linux/macOS
# venv\Scripts\activate      # Windows
```

## 1.2. Install Dependencies


## 1.1. Python Environment

```
pip install torch torchvision transformers tqdm pillow
```

# 2. Dataset Structure

The framework uses four OCT datasets:

1. Zhang OCT Dataset
2. Duke OCT Dataset
3. Rabbani OCT Dataset
4. BIOMISA OCT Dataset

```
data/
  Zhang/
    train/
      AMD/
      DME/
      NORMAL/
    test/
      AMD/
      DME/
      NORMAL/

  Duke/
    train/
      AMD/
      DME/
      NORMAL/
    test/
      AMD/
      DME/
      NORMAL/

  Rabbani/
    train/
      AMD/
      DME/
      NORMAL/
    test/
      AMD/
      DME/
      NORMAL/

  BIOMISA/
    train/
      AMD/
      DME/
      NORMAL/
    test/
      AMD/
      DME/
      NORMAL/
```

If your dataset uses different folder names, modify CLASS_TO_IDX in:

```
data/oct_dataset.py
```

## 2.1. Prompts

The prompts can be saved as CSV in:

```
data/clinical_prompts_10k.csv
```

## 2.2. Prompt Structure

```
text,label
"The OCT image shows drusen and subretinal fluid suggestive of AMD.",AMD
"This OCT scan has macular edema consistent with DME.",DME
"There is no significant macular pathology.",NORMAL
...
```

# 3. Usage

Run the following script to run inference:

```
main.py
```

# 4. Training

## 4.1. Stage 1 — Text Encoder Pretraining

To train the text encoder, run:

```
python main.py --stage text_pretrain \
  --prompts_csv data/clinical_prompts_10k.csv
```

This produces:

```
checkpoints/text_encoder_bert_pretrained.pth
```

## 4.2. Stage 2 — Visual Encoder Training

To train the visual HDC encoder, run:

```
python main.py --stage train_visual \
  --zhang_root data/Zhang
```

## 4.3. Stage 3 — Inference via LHDM

To run inference across all datasets, run:

```
python main.py --stage infer \
  --zhang_root data/Zhang \
  --duke_root data/Duke \
  --rabbani_root data/Rabbani \
  --biomisa_root data/BIOMISA
```

## Citation
If you use any part of this code in your research, please cite the following paper:

```
@inproceedings{Salik2025SR,
  title   = {Language Assisted Learnable Hyperdimensional Computing Framework for Retinal Disease Classification},
  author  = {Adnan Yaqoob Salik and Shehzad Khalid and Ramsha Ahmed and Fathi Awad and Umer Hameed Shah and Taimur Hassan},
  note = {Under Review in Scientific Reports},
  year = {2025}
}
```
