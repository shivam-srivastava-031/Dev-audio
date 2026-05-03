# Book-to-Audio Pipeline

An end-to-end, production-grade pipeline that converts **scanned PDF books** into fully structured **audiobooks (MP3 + M4B with chapter navigation)** using a multi-model AI architecture.

---

## Architecture

```
Scanned PDF
    │
    ▼
[0] PDF → Page Images (PyMuPDF, 300 DPI)
    │
    ▼
[1] Layout Detection (DocLayout-YOLO / Nougat / LayoutLMv3)
    │   Classify: body text, headers, footers, figures — reading order preserved
    ▼
[2] OCR — Fine-tuned TrOCR (microsoft/trocr-large-printed)
    │   Per-region extraction at 300 DPI
    ▼
[3] LLM Post-processing (Llama 3.1 via Ollama)
    │   Fix OCR errors, hyphenation, noise — hallucination guard included
    ▼
[4] Language Detection + Translation (optional)
    │   langdetect → NLLB-200 (Meta, 200 languages)
    ▼
[5] Semantic Structuring (chapters → sections → paragraphs)
    │   Rule-based regex + optional LLM → structured JSON
    ▼
[6] Emotion Tagging + Speaker Segmentation
    │   RoBERTa emotion classifier + dialogue detection → voice IDs
    ▼
[7] TTS (Coqui XTTS-v2 / Fish Speech V1.5 / ElevenLabs)
    │   Per-sentence synthesis with emotion prosody & speaker voice profiles
    ▼
[8] Audio Stitching + Chapter Metadata
    │   pydub stitching → EBU R128 normalisation → ffmpeg → mutagen M4B tags
    ▼
Audiobook (MP3 + M4B + Chapters + Metadata JSON)
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
# ffmpeg must also be installed and in PATH:
# Windows: winget install ffmpeg   |   Ubuntu: sudo apt install ffmpeg
```

### 2. (Optional) Pull Llama 3.1 for OCR correction

```bash
# Install Ollama from https://ollama.ai then:
ollama pull llama3.1
```

### 3. Convert a book

```bash
# Basic (all defaults, CPU inference)
python convert_book.py --input my_book.pdf --output ./output/

# With optional translation to English
python convert_book.py --input french_book.pdf --translate --output ./output/

# Skip layout detection (simple single-column books, much faster)
python convert_book.py --input simple_book.pdf --no-layout --output ./output/

# GPU acceleration + gTTS fallback (no TTS model download)
python convert_book.py --input book.pdf --device cuda --tts-engine gtts --output ./output/

# Process only pages 1–20 for testing
python convert_book.py --input book.pdf --pages 1-20 --output ./output/
```

---

## Output Files

```
output/
├── audiobook.mp3               # Full audiobook (no chapter navigation)
├── audiobook.m4b               # M4B with chapter navigation (VLC, Apple Books)
├── chapters/
│   ├── chapter_001.mp3
│   ├── chapter_002.mp3
│   └── ...
├── metadata.json               # Book metadata (title, author, chapters, durations)
└── .pipeline_temp/             # Intermediate artifacts (safe to delete after)
    ├── layout.json
    ├── ocr_output.json
    ├── corrected_text.txt
    ├── book_structure.json
    ├── tagged_book.json
    └── tts_chunks/
        └── chapter_001/
            ├── ch001_s00001.wav
            └── ...
```

---

## Configuration

Edit `config.yaml` to configure model paths, voice profiles, and pipeline behaviour.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `layout.engine` | `doclayout` | Layout detection backend |
| `ocr.model` | `microsoft/trocr-large-printed` | Replace with your fine-tuned HF model after Kaggle training |
| `llm.backend` | `ollama` | LLM backend for OCR correction |
| `tts.engine` | `xtts` | TTS backend |
| `tts.voice_profiles` | narrator_default | Add character voice WAV files here |
| `audio.target_lufs` | `-16.0` | EBU R128 loudness standard for audiobooks |
| `translation.enabled` | `false` | Enable NLLB translation |

---

## Fine-tuning TrOCR (Stage 2)

The `kaggle_notebooks/book_ocr_finetune.ipynb` notebook trains a domain-adapted TrOCR model on real scanned book data. Run it on Kaggle (free T4 GPU):

1. Upload the notebook to [Kaggle Notebooks](https://kaggle.com/code)
2. Enable GPU accelerator (T4 x2)
3. Run all cells (~3–5 hours)
4. Push the trained model to your HuggingFace Hub
5. Update `config.yaml → ocr.model` with your HF model ID

---

## CLI Reference

```
python convert_book.py --help
```

| Flag | Default | Description |
|---|---|---|
| `--input` | *required* | Input PDF path |
| `--output` | *required* | Output directory |
| `--title` | inferred | Book title |
| `--author` | Unknown Author | Book author |
| `--cover` | none | Cover art image (JPG/PNG) |
| `--no-layout` | false | Skip layout detection |
| `--no-llm` | false | Skip LLM OCR correction |
| `--translate` | false | Enable language translation |
| `--no-emotion` | false | Skip emotion tagging |
| `--tts-engine` | xtts | TTS backend |
| `--device` | cpu | Inference device |
| `--dpi` | 300 | PDF render resolution |
| `--pages` | all | Page range e.g. `1-20` |
| `--verbose` | false | Enable debug logging |

---

## Project Structure

```
book_to_audio/
├── kaggle_notebooks/
│   ├── book_ocr_finetune.ipynb      # Stage 2: TrOCR fine-tuning (run on Kaggle)
│   └── book_ocr_finetune.py         # Python version of the notebook
├── pipeline/
│   ├── __init__.py
│   ├── pdf_to_images.py             # Stage 0: PDF → page images
│   ├── layout_detector.py           # Stage 1: DocLayout-YOLO / Nougat
│   ├── ocr_engine.py                # Stage 2: TrOCR per-region OCR
│   ├── llm_corrector.py             # Stage 3: Llama 3.1 post-processing
│   ├── lang_detector.py             # Stage 4: langdetect + NLLB translation
│   ├── semantic_structurer.py       # Stage 5: Chapter/section JSON
│   ├── emotion_tagger.py            # Stage 6: RoBERTa emotion + speaker IDs
│   ├── tts_engine.py                # Stage 7: XTTS-v2 / Fish Speech TTS
│   └── audio_builder.py             # Stage 8: stitching + M4B metadata
├── convert_book.py                  # Main CLI entrypoint
├── config.yaml                      # Model paths, TTS voice profiles, flags
├── requirements.txt
└── README.md
```
