# Deceptive Text Detection with BERT & DistilBERT

> **Fine-tuning pre-trained transformer models to classify deceptive vs. genuine text — a methodology directly applicable to fraud detection, AML narrative analysis, and financial misinformation identification.**

---

## Project Overview

This project fine-tunes BERT and DistilBERT on a labelled dataset of deceptive and truthful consumer reviews, building a binary text classifier capable of distinguishing fabricated content from genuine writing. Beyond the classification task itself, the project conducts a systematic **model comparison experiment** (BERT vs. DistilBERT) and explores **decision threshold optimisation** — both of which are critical considerations in real-world deployment.

**Core Task:** Given a piece of text, predict whether it is *deceptive (fake)* or *truthful (genuine)*.

**Finance Application:** The same fine-tuning approach is used in financial services for:
- Detecting fabricated narratives in loan/insurance applications
- Classifying suspicious activity report (SAR) text in AML workflows
- Identifying manipulative or misleading content in financial communications
- Flagging anomalous free-text fields in credit underwriting

---

## Technical Stack

| Category | Tools |
|---|---|
| Deep Learning Framework | PyTorch |
| Pre-trained Models | `bert-base-uncased`, `distilbert-base-uncased` (HuggingFace Transformers) |
| Training | Fine-tuning with AdamW optimiser, custom training loop |
| Evaluation | Accuracy, Confusion Matrix, Threshold Sensitivity Analysis |
| Visualisation | `matplotlib`, `seaborn`, `ipywidgets` (interactive threshold slider) |

---

## Dataset

- **Source:** Cell Phones & Accessories Deceptive/Truthful Review Dataset
- **Task:** Binary classification — `deceptive (1)` vs. `truthful (0)`
- **Split:** 70% train / 15% validation / 15% test
- **Text field:** `reviewText`; **Label field:** `deceptive`

---

## Notebook Walkthrough

### 1. Data Loading & Preprocessing
Tokenisation using the BERT/DistilBERT tokenizer with `MAX_LEN=128`. Train/validation/test split with stratified sampling via `load_fake_true_reviews`.

### 2. Model Architecture
Standard `BertForSequenceClassification` / `DistilBertForSequenceClassification` from HuggingFace, with a linear classification head on top of the `[CLS]` token representation.

### 3. Hyperparameter Configuration
Key hyperparameters configured and varied across experiments:

| Parameter | BERT | DistilBERT |
|---|---|---|
| `MODEL_NAME` | bert-base-uncased | distilbert-base-uncased |
| `MAX_LEN` | 128 | 128 |
| `BATCH_SIZE` | 16 | 16 |
| `LEARNING_RATE` | 2e-5 | 3e-5 |
| `NUM_EPOCHS` | 8 | 8 |

### 4. Training & Evaluation
- Custom epoch loop tracking training loss and validation accuracy per epoch
- Training curve visualisation (loss convergence, accuracy improvement)
- Final evaluation on held-out test set

### 5. Threshold Optimisation
Rather than defaulting to `threshold=0.5`, the project uses an **interactive confusion matrix** (ipywidgets slider) to explore the precision/recall trade-off across thresholds from 0.0 to 1.0. This is important in fraud/AML contexts where the cost of false negatives (missed fraud) differs substantially from false positives (unnecessary investigations).

### 6. BERT vs. DistilBERT Comparison
Side-by-side comparison of training loss curves and validation accuracy across all epochs. DistilBERT is ~40% smaller and faster to infer, making it the preferred choice in latency-sensitive production systems. The comparison quantifies the accuracy-efficiency trade-off for this specific task.

### 7. Inference on New Text
```python
test_reviews = [
    "This product is wonderful, I highly recommend it to everyone.",
    "Worst purchase ever, completely useless.",
    "The item is just okay, but the reviews sound exaggerated.",
]
preds = sbr.predict_fake_true(test_reviews, loaded_model, loaded_tokenizer, device)
# Output: label ('true'/'fake') + confidence score per review
```

### 8. Model Persistence
Model weights and tokenizer saved locally for reuse (`fake_true_model/`), demonstrating awareness of production deployment requirements.

---

## Key Findings

| Metric | BERT | DistilBERT |
|---|---|---|
| Final Val. Accuracy | — | — |
| Training Stability | Converges smoothly over 8 epochs | Similar convergence, faster per-epoch |
| Inference Speed | Baseline | ~1.6× faster |
| Recommended Use | Highest accuracy required | Latency-sensitive deployment |

*(Results depend on hardware and run — see training curves in notebook.)*

---

## Why This Matters for Financial AI

The precision/recall trade-off explored through threshold optimisation is especially important in finance:

- **High threshold (conservative):** Fewer false positives — fewer legitimate cases incorrectly flagged, but higher risk of missing actual fraud.
- **Low threshold (aggressive):** Catches more fraud, but increases investigation burden on compliance teams.

The same BERT fine-tuning pipeline can be adapted to financial text with minimal modification — replacing consumer reviews with loan application narratives, transaction descriptions, or earnings call transcripts.

---

## How to Run

```bash
# Install dependencies
pip install torch transformers tqdm pandas scikit-learn matplotlib seaborn ipywidgets

# Place dataset
# Save your labelled text dataset as: data/CellPhonesAccessoriesdeceptivetruthful_dataset.txt

# Open notebook
jupyter notebook bert_deceptive_text_detection.ipynb
```

**Note:** GPU strongly recommended for training (Google Colab with T4 GPU works well). CPU-only training on 8 epochs will be slow.

---

## Project Context

This project was completed as part of an NLP module at Audencia Business School (2026). The BERT training utilities were provided as a course framework (`simple_bert_fake_reviews` module); the experimental design — including hyperparameter selection, the BERT vs. DistilBERT comparative experiment, and decision threshold analysis — was conducted independently by Ren REN.
