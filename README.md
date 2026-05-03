# Dev-Audio 🎧 — Book to Audiobook Converter

An end-to-end AI pipeline that converts **scanned PDF books** into high-quality **audiobooks (MP3 + M4B)** using a multi-model architecture.

## Architecture

```
Scanned PDF
    │
    ▼
[1] Layout Detection       (Nougat / DocLayout-YOLO)
    │
    ▼
[2] OCR                    (Fine-tuned TrOCR)
    │
    ▼
[3] LLM Post-processing    (Llama 3.1 — context-aware correction)
    │
    ▼
[4] Language Detection + Translation  (langdetect + NLLB-200)
    │
    ▼
[5] Semantic Structuring   (chapters, sections, paragraphs)
    │
    ▼
[6] Emotion Tagging + Speaker Segmentation
    │
    ▼
[7] Advanced TTS           (Fish Speech V1.5 / XTTS-v2)
    │
    ▼
[8] Audio Stitching + Chapter Metadata  (ffmpeg + mutagen)
    │
    ▼
Audiobook (MP3 + M4B + Chapters)
```

## Project Structure

```
Dev-audio/
├── kaggle_notebooks/
│   ├── book_ocr_finetune.ipynb     # TrOCR fine-tuning (run on Kaggle GPU)
│   └── book_ocr_finetune.py        # Same as notebook — plain Python
├── pipeline/                        # Inference pipeline (coming soon)
│   ├── pdf_to_images.py
│   ├── layout_detector.py
│   ├── ocr_engine.py
│   ├── llm_corrector.py
│   ├── lang_detector.py
│   ├── semantic_structurer.py
│   ├── emotion_tagger.py
│   ├── tts_engine.py
│   └── audio_builder.py
├── convert_book.py                  # Main CLI runner (coming soon)
├── requirements.txt
└── README.md
```

## Kaggle Notebook — TrOCR Fine-tuning

### Setup
1. Upload `kaggle_notebooks/book_ocr_finetune.ipynb` to [Kaggle](https://kaggle.com)
2. Set **GPU T4 x2** in Settings → Accelerator
3. Enable **Internet** in Settings
4. Change `hub_model_id` in the Config cell to `YOUR_HF_USERNAME/trocr-book-finetuned`
5. Run all cells

### Datasets Used
| Dataset | Source | Purpose |
|---|---|---|
| IAM Handwriting | `Teklia/IAM-line` (HuggingFace) | Printed + handwritten lines |
| FUNSD | `nielsr/funsd` (HuggingFace) | Scanned form documents |
| Synthetic Book Lines | Generated via PIL | Book-style domain adaptation |

### Training Config
- **Base model**: `microsoft/trocr-large-printed`
- **Effective batch**: 32 (8 × 4 grad accum)
- **Precision**: FP16
- **Epochs**: 12 (early stopping, patience=3)
- **Target**: CER < 5%, WER < 10%
- **Output**: Pushed to HuggingFace Hub

## Requirements

```bash
pip install transformers==4.40.0 datasets evaluate albumentations \
    jiwer pillow accelerate sentencepiece PyMuPDF pydub \
    langdetect ffmpeg-python mutagen pyloudnorm
```

## Usage (Full Pipeline — coming soon)

```bash
python convert_book.py --input my_book.pdf --output ./output/
python convert_book.py --input french_book.pdf --translate --output ./output/
```

## Models Used

| Stage | Model |
|---|---|
| Layout Detection | Nougat (`facebook/nougat-base`) + DocLayout-YOLO |
| OCR | `microsoft/trocr-large-printed` (fine-tuned) |
| LLM Correction | `meta-llama/Llama-3.1-8B-Instruct` |
| Translation | `facebook/nllb-200-distilled-600M` |
| Emotion Tagging | `j-hartmann/emotion-english-distilroberta-base` |
| TTS | Fish Speech V1.5 / XTTS-v2 |

## License

MIT License — see [LICENSE](LICENSE)
