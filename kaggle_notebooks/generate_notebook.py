
import json, random

def md_cell(source_lines):
    return {"cell_type": "markdown", "metadata": {}, "source": source_lines}

def code_cell(source_lines):
    return {"cell_type": "code", "metadata": {}, "source": source_lines,
            "execution_count": None, "outputs": []}

cells = [
    md_cell([
        "# Book OCR Fine-tuning: TrOCR on Scanned Book Pages\n",
        "\n",
        "**Model**: `microsoft/trocr-large-printed`  \n",
        "**Datasets**: IAM + FUNSD + Synthetic Book Lines  \n",
        "**GPU**: Kaggle T4 x2 / P100  \n",
        "**Target**: CER < 5%%, WER < 10%%\n",
        "\n",
        "### Pipeline\n",
        "1. Load and merge OCR datasets\n",
        "2. Augment images (blur, noise, rotation)\n",
        "3. Fine-tune TrOCR via Seq2SeqTrainer\n",
        "4. Evaluate CER / WER\n",
        "5. Push best model to HuggingFace Hub\n"
    ]),

    code_cell([
        "# Cell 1: Install Dependencies\n",
        "!pip install -q transformers==4.40.0 datasets evaluate albumentations \\\n",
        "    jiwer pillow accelerate sentencepiece --upgrade\n"
    ]),

    code_cell([
        "# Cell 2: Imports\n",
        "import os, re, random, json\n",
        "import numpy as np\n",
        "from PIL import Image, ImageDraw\n",
        "import albumentations as A\n",
        "import torch\n",
        "from torch.utils.data import Dataset\n",
        "import evaluate\n",
        "from datasets import load_dataset\n",
        "from transformers import (\n",
        "    TrOCRProcessor,\n",
        "    VisionEncoderDecoderModel,\n",
        "    Seq2SeqTrainer,\n",
        "    Seq2SeqTrainingArguments,\n",
        "    EarlyStoppingCallback,\n",
        ")\n",
        "print('Torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())\n"
    ]),

    md_cell(["## Configuration\n",
             "Change `hub_model_id` to your HuggingFace username before training.\n"]),

    code_cell([
        "# Cell 3: Config\n",
        "CFG = {\n",
        "    'base_model':   'microsoft/trocr-large-printed',\n",
        "    'output_dir':   './trocr-book-finetuned',\n",
        "    'hub_model_id': 'YOUR_HF_USERNAME/trocr-book-finetuned',\n",
        "    'push_to_hub':  True,\n",
        "    'max_length':   128,\n",
        "    'train_batch':  8,\n",
        "    'eval_batch':   8,\n",
        "    'grad_accum':   4,\n",
        "    'epochs':       12,\n",
        "    'lr':           5e-5,\n",
        "    'warmup_steps': 500,\n",
        "    'weight_decay': 0.01,\n",
        "    'fp16':         True,\n",
        "    'seed':         42,\n",
        "}\n",
        "random.seed(CFG['seed'])\n",
        "np.random.seed(CFG['seed'])\n",
        "torch.manual_seed(CFG['seed'])\n",
        "device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
        "print('Device:', device)\n"
    ]),

    md_cell(["## Augmentation Pipeline\n",
             "Simulates real scan conditions: blur, noise, rotation, compression.\n"]),

    code_cell([
        "# Cell 4: Augmentation\n",
        "aug = A.Compose([\n",
        "    A.Rotate(limit=2, p=0.5),\n",
        "    A.GaussianBlur(blur_limit=(1, 3), p=0.3),\n",
        "    A.GaussNoise(var_limit=(5, 25), p=0.3),\n",
        "    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),\n",
        "    A.ImageCompression(quality_lower=60, quality_upper=95, p=0.3),\n",
        "])\n",
        "\n",
        "def augment_image(img):\n",
        "    arr = np.array(img.convert('RGB'))\n",
        "    return Image.fromarray(aug(image=arr)['image'])\n",
        "\n",
        "print('Augmentation pipeline ready')\n"
    ]),

    md_cell(["## Unified Dataset Wrapper\n"]),

    code_cell([
        "# Cell 5: OCRLineDataset\n",
        "class OCRLineDataset(Dataset):\n",
        "    def __init__(self, samples, processor, max_length, do_augment=False):\n",
        "        self.samples   = samples\n",
        "        self.processor = processor\n",
        "        self.max_len   = max_length\n",
        "        self.do_aug    = do_augment\n",
        "\n",
        "    def __len__(self):\n",
        "        return len(self.samples)\n",
        "\n",
        "    def __getitem__(self, idx):\n",
        "        image, text = self.samples[idx]\n",
        "        if not isinstance(image, Image.Image):\n",
        "            image = Image.fromarray(image)\n",
        "        image = image.convert('RGB')\n",
        "        if self.do_aug:\n",
        "            image = augment_image(image)\n",
        "        pixel_values = self.processor(\n",
        "            images=image, return_tensors='pt'\n",
        "        ).pixel_values.squeeze(0)\n",
        "        labels = self.processor.tokenizer(\n",
        "            text, padding='max_length',\n",
        "            max_length=self.max_len, truncation=True,\n",
        "        ).input_ids\n",
        "        pad_id = self.processor.tokenizer.pad_token_id\n",
        "        labels = [l if l != pad_id else -100 for l in labels]\n",
        "        return {\n",
        "            'pixel_values': pixel_values,\n",
        "            'labels': torch.tensor(labels, dtype=torch.long),\n",
        "        }\n",
        "\n",
        "print('OCRLineDataset defined')\n"
    ]),

    md_cell(["## Load & Merge Datasets\n",
             "| Dataset | Approx Samples | Domain |\n",
             "|---|---|---|\n",
             "| IAM Handwriting | ~13k | Printed + handwritten lines |\n",
             "| FUNSD | ~30k word crops | Scanned form documents |\n",
             "| Synthetic (PIL) | ~5k | Book-style text lines |\n"]),

    code_cell([
        "# Cell 6: Load IAM\n",
        "processor = TrOCRProcessor.from_pretrained(CFG['base_model'])\n",
        "\n",
        "def load_iam(split):\n",
        "    ds = load_dataset('Teklia/IAM-line', split=split, trust_remote_code=True)\n",
        "    out = []\n",
        "    for r in ds:\n",
        "        try:\n",
        "            img = r['image'] if isinstance(r['image'], Image.Image) else Image.fromarray(r['image'])\n",
        "            t = r['text'].strip()\n",
        "            if t:\n",
        "                out.append((img, t))\n",
        "        except:\n",
        "            pass\n",
        "    print('IAM', split, ':', len(out))\n",
        "    return out\n",
        "\n",
        "iam_train = load_iam('train')\n",
        "iam_val   = load_iam('validation')\n"
    ]),

    code_cell([
        "# Cell 7: Load FUNSD\n",
        "def load_funsd(split):\n",
        "    ds = load_dataset('nielsr/funsd', split=split, trust_remote_code=True)\n",
        "    out = []\n",
        "    for r in ds:\n",
        "        img = r['image']\n",
        "        if not isinstance(img, Image.Image):\n",
        "            img = Image.fromarray(img)\n",
        "        for w, b in zip(r.get('words', []), r.get('bboxes', [])):\n",
        "            w = w.strip()\n",
        "            if not w:\n",
        "                continue\n",
        "            try:\n",
        "                crop = img.crop(b)\n",
        "                if crop.width > 5 and crop.height > 5:\n",
        "                    out.append((crop, w))\n",
        "            except:\n",
        "                pass\n",
        "    print('FUNSD', split, ':', len(out))\n",
        "    return out\n",
        "\n",
        "funsd_train = load_funsd('train')\n",
        "funsd_val   = load_funsd('test')\n"
    ]),

    code_cell([
        "# Cell 8: Generate Synthetic Book Lines\n",
        "BOOK_LINES = [\n",
        "    'The quick brown fox jumps over the lazy dog.',\n",
        "    'It was the best of times, it was the worst of times.',\n",
        "    'Call me Ishmael. Some years ago never mind how long.',\n",
        "    'In the beginning God created the heavens and the earth.',\n",
        "    'All happy families are alike; each unhappy family differs.',\n",
        "    'To be or not to be, that is the question.',\n",
        "    'It is a truth universally acknowledged.',\n",
        "    'Far out in the uncharted backwaters of the galaxy.',\n",
        "    'The man in black fled across the desert.',\n",
        "    'Happy families are all alike; every unhappy family is unhappy.',\n",
        "]\n",
        "\n",
        "def gen_synthetic(n=5000):\n",
        "    out = []\n",
        "    for _ in range(n):\n",
        "        text = random.choice(BOOK_LINES)\n",
        "        img = Image.new('RGB', (500, 50), (255, 255, 255))\n",
        "        ImageDraw.Draw(img).text((8, 10), text, fill=(0, 0, 0))\n",
        "        out.append((img, text))\n",
        "    print('Synthetic:', len(out))\n",
        "    return out\n",
        "\n",
        "synthetic = gen_synthetic(5000)\n"
    ]),

    code_cell([
        "# Cell 9: Build Final Datasets\n",
        "all_train = iam_train + funsd_train + synthetic\n",
        "all_val   = iam_val   + funsd_val\n",
        "\n",
        "random.shuffle(all_train)\n",
        "print('Total train:', len(all_train), '| Total val:', len(all_val))\n",
        "\n",
        "train_ds = OCRLineDataset(all_train, processor, CFG['max_length'], do_augment=True)\n",
        "val_ds   = OCRLineDataset(all_val,   processor, CFG['max_length'], do_augment=False)\n",
        "print('Datasets ready')\n"
    ]),

    md_cell(["## Model Initialization\n"]),

    code_cell([
        "# Cell 10: Load and Configure TrOCR\n",
        "model = VisionEncoderDecoderModel.from_pretrained(CFG['base_model'])\n",
        "\n",
        "model.config.decoder_start_token_id = processor.tokenizer.cls_token_id\n",
        "model.config.pad_token_id           = processor.tokenizer.pad_token_id\n",
        "model.config.vocab_size             = model.config.decoder.vocab_size\n",
        "model.config.eos_token_id           = processor.tokenizer.sep_token_id\n",
        "model.config.max_length             = CFG['max_length']\n",
        "model.config.no_repeat_ngram_size   = 3\n",
        "model.config.length_penalty         = 2.0\n",
        "model.config.num_beams              = 4\n",
        "model.to(device)\n",
        "\n",
        "total_params = sum(p.numel() for p in model.parameters())\n",
        "print('Model loaded. Parameters:', f'{total_params:,}')\n"
    ]),

    md_cell(["## CER + WER Metrics\n"]),

    code_cell([
        "# Cell 11: Metrics\n",
        "cer_metric = evaluate.load('cer')\n",
        "wer_metric = evaluate.load('wer')\n",
        "\n",
        "def compute_metrics(pred):\n",
        "    labels = pred.label_ids\n",
        "    pad_id = processor.tokenizer.pad_token_id\n",
        "    labels[labels == -100] = pad_id\n",
        "    pred_str  = processor.batch_decode(pred.predictions, skip_special_tokens=True)\n",
        "    label_str = processor.batch_decode(labels,           skip_special_tokens=True)\n",
        "    cer = cer_metric.compute(predictions=pred_str, references=label_str)\n",
        "    wer = wer_metric.compute(predictions=pred_str, references=label_str)\n",
        "    return {'cer': round(cer, 4), 'wer': round(wer, 4)}\n",
        "\n",
        "print('Metrics ready')\n"
    ]),

    md_cell(["## Training\n",
             "- Effective batch size: **32** (8 device × 4 grad accum)\n",
             "- Mixed precision: **FP16**\n",
             "- Early stopping: patience = **3 epochs**\n",
             "- Best checkpoint: lowest **CER**\n"]),

    code_cell([
        "# Cell 12: Training Arguments\n",
        "training_args = Seq2SeqTrainingArguments(\n",
        "    output_dir                  = CFG['output_dir'],\n",
        "    num_train_epochs            = CFG['epochs'],\n",
        "    per_device_train_batch_size = CFG['train_batch'],\n",
        "    per_device_eval_batch_size  = CFG['eval_batch'],\n",
        "    gradient_accumulation_steps = CFG['grad_accum'],\n",
        "    learning_rate               = CFG['lr'],\n",
        "    warmup_steps                = CFG['warmup_steps'],\n",
        "    weight_decay                = CFG['weight_decay'],\n",
        "    fp16                        = CFG['fp16'],\n",
        "    predict_with_generate       = True,\n",
        "    evaluation_strategy         = 'epoch',\n",
        "    save_strategy               = 'epoch',\n",
        "    logging_steps               = 100,\n",
        "    load_best_model_at_end      = True,\n",
        "    metric_for_best_model       = 'cer',\n",
        "    greater_is_better           = False,\n",
        "    save_total_limit            = 2,\n",
        "    report_to                   = 'none',\n",
        "    push_to_hub                 = CFG['push_to_hub'],\n",
        "    hub_model_id                = CFG['hub_model_id'],\n",
        ")\n",
        "print('Training args ready')\n"
    ]),

    code_cell([
        "# Cell 13: Run Training\n",
        "trainer = Seq2SeqTrainer(\n",
        "    model           = model,\n",
        "    args            = training_args,\n",
        "    train_dataset   = train_ds,\n",
        "    eval_dataset    = val_ds,\n",
        "    compute_metrics = compute_metrics,\n",
        "    callbacks       = [EarlyStoppingCallback(early_stopping_patience=3)],\n",
        ")\n",
        "\n",
        "print('Starting training...')\n",
        "trainer.train()\n"
    ]),

    md_cell(["## Evaluate + Save + Push\n"]),

    code_cell([
        "# Cell 14: Evaluate\n",
        "metrics = trainer.evaluate()\n",
        "print(json.dumps(metrics, indent=2))\n"
    ]),

    code_cell([
        "# Cell 15: Save locally\n",
        "trainer.save_model(CFG['output_dir'])\n",
        "processor.save_pretrained(CFG['output_dir'])\n",
        "print('Saved to', CFG['output_dir'])\n"
    ]),

    code_cell([
        "# Cell 16: Push to HuggingFace Hub\n",
        "if CFG['push_to_hub']:\n",
        "    trainer.push_to_hub(commit_message='TrOCR book OCR fine-tuned')\n",
        "    processor.push_to_hub(CFG['hub_model_id'])\n",
        "    print('Pushed to:', CFG['hub_model_id'])\n"
    ]),

    md_cell(["## Inference Test\n",
             "Use this cell to test your fine-tuned model on any image.\n"]),

    code_cell([
        "# Cell 17: Inference\n",
        "def predict(image_path):\n",
        "    img = Image.open(image_path).convert('RGB')\n",
        "    pv  = processor(images=img, return_tensors='pt').pixel_values.to(device)\n",
        "    with torch.no_grad():\n",
        "        ids = model.generate(pv, max_length=CFG['max_length'], num_beams=4)\n",
        "    return processor.batch_decode(ids, skip_special_tokens=True)[0]\n",
        "\n",
        "# Example usage:\n",
        "# print(predict('/kaggle/input/sample_page.png'))\n",
        "print('Training complete! Inference function ready.')\n",
        "print('Load your model from HuggingFace Hub with:')\n",
        "print('  TrOCRProcessor.from_pretrained(CFG[hub_model_id])')\n",
        "print('  VisionEncoderDecoderModel.from_pretrained(CFG[hub_model_id])')\n",
    ]),
]

notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        },
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "dataSources": [],
            "isGpuEnabled": True,
            "isInternetEnabled": True
        }
    },
    "cells": cells
}

out_path = r"C:\Users\shiva\.gemini\antigravity\scratch\book_to_audio\kaggle_notebooks\book_ocr_finetune.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Notebook created:", out_path)
print("Total cells:", len(cells))
