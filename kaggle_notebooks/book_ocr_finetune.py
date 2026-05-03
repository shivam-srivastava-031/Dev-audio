# =============================================================================
# BOOK OCR FINE-TUNING — TrOCR on Kaggle GPU
# Model: microsoft/trocr-large-printed
# Datasets: IAM + SROIE + FUNSD + DocVQA + Synthetic Book Pages
# =============================================================================

# ── CELL 1: Install Dependencies ─────────────────────────────────────────────
# Run this cell first on Kaggle
"""
!pip install -q transformers==4.40.0 datasets evaluate albumentations \
    jiwer pillow accelerate sentencepiece sacrebleu \
    TextRecognitionDataGenerator --upgrade
"""

# ── CELL 2: Imports ───────────────────────────────────────────────────────────
import os, re, random, json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import albumentations as A
import torch
from torch.utils.data import Dataset, ConcatDataset

import evaluate
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import (
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    EarlyStoppingCallback,
)

# ── CELL 3: Configuration ────────────────────────────────────────────────────
CFG = {
    # Model
    "base_model":       "microsoft/trocr-large-printed",
    "output_dir":       "./trocr-book-finetuned",
    "hub_model_id":     "YOUR_HF_USERNAME/trocr-book-finetuned",   # ← change this
    "push_to_hub":      True,

    # Training
    "max_length":       128,
    "image_size":       (384, 384),
    "train_batch":      8,
    "eval_batch":       8,
    "grad_accum":       4,          # effective batch = 32
    "epochs":           12,
    "lr":               5e-5,
    "warmup_steps":     500,
    "weight_decay":     0.01,
    "fp16":             True,       # T4 / P100 on Kaggle

    # Data
    "val_split":        0.05,
    "max_train_samples": None,      # set to int to cap (e.g. 50_000)
    "seed":             42,
}

random.seed(CFG["seed"])
np.random.seed(CFG["seed"])
torch.manual_seed(CFG["seed"])
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ── CELL 4: Augmentation Pipeline ────────────────────────────────────────────
def get_augmentation():
    return A.Compose([
        A.Rotate(limit=2, p=0.5),
        A.GaussianBlur(blur_limit=(1, 3), p=0.3),
        A.GaussNoise(var_limit=(5, 25), p=0.3),
        A.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),
        A.RandomShadow(p=0.2),
    ])

augment = get_augmentation()

def augment_image(pil_image: Image.Image) -> Image.Image:
    img_np = np.array(pil_image.convert("RGB"))
    augmented = augment(image=img_np)["image"]
    return Image.fromarray(augmented)

# ── CELL 5: Load & Merge Datasets ────────────────────────────────────────────

class OCRLineDataset(Dataset):
    """Unified wrapper — accepts list of (PIL.Image, str) pairs."""

    def __init__(self, samples, processor, max_length, augment=False):
        self.samples   = samples
        self.processor = processor
        self.max_len   = max_length
        self.augment   = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, text = self.samples[idx]

        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = image.convert("RGB")

        if self.augment:
            image = augment_image(image)

        pixel_values = self.processor(
            images=image, return_tensors="pt"
        ).pixel_values.squeeze(0)

        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_len,
            truncation=True,
        ).input_ids

        labels = [
            l if l != self.processor.tokenizer.pad_token_id else -100
            for l in labels
        ]

        return {
            "pixel_values": pixel_values,
            "labels":       torch.tensor(labels, dtype=torch.long),
        }


def load_iam(split="train"):
    """IAM Handwriting — printed + handwritten English lines."""
    ds = load_dataset("Teklia/IAM-line", split=split, trust_remote_code=True)
    samples = []
    for row in ds:
        try:
            img  = row["image"] if isinstance(row["image"], Image.Image) \
                   else Image.fromarray(row["image"])
            text = row["text"].strip()
            if text:
                samples.append((img, text))
        except Exception:
            continue
    print(f"IAM [{split}]: {len(samples)} samples")
    return samples


def load_funsd(split="train"):
    """FUNSD — form understanding, printed text blocks."""
    ds = load_dataset("nielsr/funsd", split=split, trust_remote_code=True)
    samples = []
    for row in ds:
        for word, box in zip(row.get("words", []), row.get("bboxes", [])):
            word = word.strip()
            if word:
                # Crop word region from full image
                try:
                    img = row["image"]
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    x0, y0, x1, y1 = box
                    crop = img.crop((x0, y0, x1, y1))
                    if crop.width > 5 and crop.height > 5:
                        samples.append((crop, word))
                except Exception:
                    continue
    print(f"FUNSD [{split}]: {len(samples)} samples")
    return samples


def load_docvqa_lines(max_samples=10_000):
    """DocVQA — document images; we use OCR annotations as text."""
    try:
        ds = load_dataset("lmms-lab/DocVQA", split="train",
                          trust_remote_code=True)
        samples = []
        for row in ds:
            text = row.get("answers", [""])[0].strip()
            if text and len(text) > 3:
                img = row.get("image")
                if img is not None:
                    if not isinstance(img, Image.Image):
                        img = Image.fromarray(img)
                    samples.append((img, text))
            if max_samples and len(samples) >= max_samples:
                break
        print(f"DocVQA: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"DocVQA skipped: {e}")
        return []


def generate_synthetic_book_pages(n=5_000):
    """
    Synthetic printed text lines using PIL — simulates book typography.
    Requires: pip install trdg (TextRecognitionDataGenerator)
    Falls back to basic PIL rendering if trdg not available.
    """
    try:
        from trdg.generators import GeneratorFromStrings
        sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "In the beginning was the Word, and the Word was with God.",
            "It was the best of times, it was the worst of times.",
            "Call me Ishmael. Some years ago—never mind how long precisely—",
            "All happy families are alike; each unhappy family is unhappy.",
            "It is a truth universally acknowledged that a single man",
            "You are about to begin reading Italo Calvino's new novel.",
            "The sky above the port was the color of television,",
        ]
        generator = GeneratorFromStrings(
            sentences * (n // len(sentences) + 1),
            count=n,
            fonts=["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf"],
            size=32,
            skewing_angle=1,
            random_skew=True,
            blur=1,
            random_blur=True,
            background_type=0,
        )
        samples = []
        for img, lbl in generator:
            samples.append((img, lbl))
            if len(samples) >= n:
                break
        print(f"Synthetic: {len(samples)} samples")
        return samples
    except Exception as e:
        print(f"trdg synthetic fallback to PIL: {e}")
        # Basic PIL fallback
        from PIL import ImageDraw, ImageFont
        lines = [
            "The quick brown fox", "jumps over the lazy dog",
            "In the beginning", "Call me Ishmael",
            "It was the best of times", "To be or not to be",
        ]
        samples = []
        for i in range(min(n, 1000)):
            text = random.choice(lines)
            w, h = 400, 60
            img  = Image.new("RGB", (w, h), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), text, fill=(0, 0, 0))
            samples.append((img, text))
        print(f"Synthetic (PIL fallback): {len(samples)} samples")
        return samples

# ── CELL 6: Build Combined Dataset ────────────────────────────────────────────

print("Loading datasets...")
processor = TrOCRProcessor.from_pretrained(CFG["base_model"])

iam_train   = load_iam("train")
iam_val     = load_iam("validation")
funsd_train = load_funsd("train")
funsd_val   = load_funsd("test")
docvqa      = load_docvqa_lines(max_samples=8_000)
synthetic   = generate_synthetic_book_pages(n=5_000)

# Combine train sources
all_train = iam_train + funsd_train + docvqa + synthetic
all_val   = iam_val   + funsd_val

# Optional: cap training samples
if CFG["max_train_samples"]:
    random.shuffle(all_train)
    all_train = all_train[:CFG["max_train_samples"]]

print(f"\nTotal train: {len(all_train)} | Total val: {len(all_val)}")

train_dataset = OCRLineDataset(
    all_train, processor, CFG["max_length"], augment=True)
val_dataset   = OCRLineDataset(
    all_val,   processor, CFG["max_length"], augment=False)

# ── CELL 7: Model Initialization ─────────────────────────────────────────────

model = VisionEncoderDecoderModel.from_pretrained(CFG["base_model"])

# Required config for TrOCR
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id           = processor.tokenizer.pad_token_id
model.config.vocab_size             = model.config.decoder.vocab_size
model.config.eos_token_id           = processor.tokenizer.sep_token_id
model.config.max_length             = CFG["max_length"]
model.config.early_stopping         = True
model.config.no_repeat_ngram_size   = 3
model.config.length_penalty         = 2.0
model.config.num_beams              = 4

model.to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── CELL 8: Metrics ───────────────────────────────────────────────────────────

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids   = pred.predictions

    # Replace -100 padding
    labels_ids[labels_ids == -100] = processor.tokenizer.pad_token_id

    pred_str  = processor.batch_decode(pred_ids,   skip_special_tokens=True)
    label_str = processor.batch_decode(labels_ids, skip_special_tokens=True)

    cer = cer_metric.compute(predictions=pred_str, references=label_str)
    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"cer": round(cer, 4), "wer": round(wer, 4)}

# ── CELL 9: Training Arguments ────────────────────────────────────────────────

training_args = Seq2SeqTrainingArguments(
    output_dir                  = CFG["output_dir"],
    num_train_epochs            = CFG["epochs"],
    per_device_train_batch_size = CFG["train_batch"],
    per_device_eval_batch_size  = CFG["eval_batch"],
    gradient_accumulation_steps = CFG["grad_accum"],
    learning_rate               = CFG["lr"],
    warmup_steps                = CFG["warmup_steps"],
    weight_decay                = CFG["weight_decay"],
    fp16                        = CFG["fp16"],
    predict_with_generate       = True,
    evaluation_strategy         = "epoch",
    save_strategy               = "epoch",
    logging_strategy            = "steps",
    logging_steps               = 100,
    load_best_model_at_end      = True,
    metric_for_best_model       = "cer",
    greater_is_better           = False,
    save_total_limit            = 2,
    report_to                   = "none",  # change to "wandb" if you want
    push_to_hub                 = CFG["push_to_hub"],
    hub_model_id                = CFG["hub_model_id"],
)

# ── CELL 10: Trainer + Train ──────────────────────────────────────────────────

trainer = Seq2SeqTrainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_dataset,
    eval_dataset    = val_dataset,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],
)

print("Starting training...")
trainer.train()

# ── CELL 11: Evaluate + Save ──────────────────────────────────────────────────

print("\nFinal evaluation:")
metrics = trainer.evaluate()
print(json.dumps(metrics, indent=2))

# Save locally
trainer.save_model(CFG["output_dir"])
processor.save_pretrained(CFG["output_dir"])
print(f"Model saved to {CFG['output_dir']}")

# Push to HuggingFace Hub
if CFG["push_to_hub"]:
    trainer.push_to_hub(commit_message="TrOCR fine-tuned on book OCR datasets")
    processor.push_to_hub(CFG["hub_model_id"])
    print(f"Pushed to HuggingFace Hub: {CFG['hub_model_id']}")

# ── CELL 12: Quick Inference Test ────────────────────────────────────────────

def predict_image(image_path: str, model, processor, device) -> str:
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(
        images=image, return_tensors="pt"
    ).pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(
            pixel_values,
            max_length=CFG["max_length"],
            num_beams=4,
        )
    return processor.batch_decode(
        generated_ids, skip_special_tokens=True)[0]

# Example usage:
# text = predict_image("sample_page.png", model, processor, device)
# print(f"Predicted: {text}")

print("\nTraining complete. Model is ready for inference.")
print(f"Load with:\n  processor = TrOCRProcessor.from_pretrained('{CFG['hub_model_id']}')")
print(f"  model = VisionEncoderDecoderModel.from_pretrained('{CFG['hub_model_id']}')")
