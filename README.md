# 🎙️ Kokoro TTS — Document to Speech

> **Turn any document into natural-sounding audio** using the Kokoro-82M model (StyleTTS2, 24 kHz), powered by a Hugging Face Space backend.

![UI Preview](https://img.shields.io/badge/UI-Dark%20Glassmorphism-7c3aed?style=for-the-badge)
![Model](https://img.shields.io/badge/Model-Kokoro--82M-38bdf8?style=for-the-badge&logo=huggingface)
![Architecture](https://img.shields.io/badge/Architecture-StyleTTS2-6366f1?style=for-the-badge)
![Audio](https://img.shields.io/badge/Audio-24%20kHz-34d399?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-fbbf24?style=for-the-badge)

---

## ✨ Features

- 📄 **Multi-format support** — Upload `.txt`, `.pdf` (digital or scanned), `.png`, `.jpg`
- 🔍 **Smart OCR** — Scanned PDFs and images are processed with EasyOCR automatically
- 🎵 **High-quality audio** — 24 kHz MP3 output via Kokoro-82M (StyleTTS2 architecture)
- ⚡ **Real-time progress** — Live 4-step status tracker via Server-Sent Events (SSE)
- 📥 **Dual downloads** — Get both the generated MP3 and the extracted/processed `.txt`
- 🖥️ **Zero dependencies** — Pure HTML + CSS + JavaScript, no frameworks or build step

---

## 🛠️ How It Works

| File Format | Processing Method | Description |
|-------------|-------------------|-------------|
| `.txt` | Direct read | Raw text passed straight to the TTS model |
| `.pdf` (digital) | pdfplumber | Extracts selectable text from encoded PDFs |
| `.pdf` (scanned) | EasyOCR | Reads text from scanned page images |
| `.png` / `.jpg` | EasyOCR | OCR on screenshots or photos of documents |

### API Flow (Gradio SSE v3)

```
1. POST /gradio_api/upload         →  Upload file, get server path
2. POST /gradio_api/queue/join     →  Queue the TTS job
3. GET  /gradio_api/queue/data     →  Stream SSE events for live progress
4.      process_completed          →  Receive audio + txt FileData objects
```

---

## 🚀 Quick Start

No installation required. Just open `index.html` in any modern browser:

```bash
git clone https://github.com/shivam-srivastava-031/Dev-audio.git
cd Dev-audio
# Open index.html directly, or serve locally:
python -m http.server 8080
# → http://localhost:8080
```

---

## 📁 Project Structure

```
Dev-audio/
├── index.html   # Semantic HTML — layout, accessibility, SEO
├── style.css    # Dark glassmorphism UI with animations
└── app.js       # Gradio SSE v3 API client logic
```

---

## 🤖 Model Details

| Property | Value |
|----------|-------|
| **Model** | [Kokoro-82M](https://huggingface.co/spaces/audio8899/kokorov2) |
| **Architecture** | StyleTTS2 |
| **Parameters** | 82 million |
| **Sample Rate** | 24,000 Hz |
| **Backend** | CPU (Hugging Face Spaces) |
| **Output** | MP3 audio + processed TXT |

> ⚠️ **Note:** The model runs on CPU. Large documents or complex scanned PDFs may take a few minutes to process.

---

## 🌐 Live Demo

The frontend is deployed and calls the HF Space at:
```
https://audio8899-kokorov2.hf.space
```

---

## 🔧 Tech Stack

- **Frontend:** Vanilla HTML5 · CSS3 · JavaScript (ES2022)
- **Fonts:** Inter · Space Grotesk (Google Fonts)
- **API:** Gradio REST + SSE (Server-Sent Events)
- **Model Hosting:** Hugging Face Spaces

---

## 📄 License

MIT © [Shivam Srivastava](https://github.com/shivam-srivastava-031)
