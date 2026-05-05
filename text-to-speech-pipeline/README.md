---
title: Document to Speech
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "4.44.0"
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# Document → Speech

Convert any `.txt`, `.pdf`, or image file (`.png` / `.jpg`) into a spoken-word MP3 using **Kokoro-82M**, a lightweight StyleTTS2-based TTS model. The app also exports the cleaned, preprocessed text as a `.txt` file so you can review exactly what was synthesised.

## How it works

```
Upload file
    │
    ├─ .txt            → direct read
    ├─ .pdf (digital)  → pdfplumber text layer
    ├─ .pdf (scanned)  → EasyOCR (page rasterised → OCR)
    └─ .png / .jpg     → EasyOCR
                              │
                     Text Preprocessor
                  (clean, normalise, chunk)
                              │
                    Kokoro-82M (StyleTTS2)
                              │
                   ┌──────────┴──────────┐
                output.mp3          output.txt
```

## Outputs

| File | Description |
|------|-------------|
| `output.mp3` | Synthesised speech audio (192 kbps) |
| `output.txt` | Cleaned text chunks passed to the TTS model |

## Files

| File | Purpose |
|------|---------|
| `app.py` | Full pipeline + Gradio UI + runtime bug patch |
| `requirements.txt` | Python dependencies |
| `packages.txt` | Debian system dependencies (`espeak-ng` required by Kokoro) |
