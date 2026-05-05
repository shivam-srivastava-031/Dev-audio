"""
Document-to-Speech  —  Hugging Face Spaces (Gradio)
====================================================
Supports: .txt  |  .pdf (digital & scanned)  |  .png  .jpg  .jpeg
Model   : Kokoro-82M  (StyleTTS2-based, 24 kHz)
OCR     : EasyOCR  (CPU on free-tier HF Spaces)
"""

import os
import re
import tempfile
import textwrap
import unicodedata
import warnings
from pathlib import Path

os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
#  CRITICAL: patch gradio_client BEFORE importing gradio
#
#  Bug: gradio_client.utils._json_schema_to_python_type() calls
#       get_type(schema['additionalProperties']) where additionalProperties
#       can legally be True/False (a JSON Schema bool shorthand).
#       get_type() then does `if "const" in schema` on a bool → TypeError.
#
#  Fix: patch _json_schema_to_python_type to guard against non-dict schemas.
# ══════════════════════════════════════════════════════════════════════════════
try:
    import gradio_client.utils as _gc_utils

    _orig_j2p = _gc_utils._json_schema_to_python_type

    def _patched_j2p(schema, defs=None):
        # JSON Schema allows bool values (true = any, false = never).
        # The original code assumes schema is always a dict — guard here.
        if not isinstance(schema, dict):
            return "Any"
        # If additionalProperties is a bool, handle this branch ourselves
        # so we NEVER pass a mutated dict copy to the original — that leaked
        # a dict into Jinja2 cache keys causing "unhashable type: dict".
        if schema.get("type") == "object" and isinstance(schema.get("additionalProperties"), bool):
            props = schema.get("properties", {})
            des = [
                f"{n}: {_patched_j2p(v, defs)}"
                for n, v in props.items()
                if n != "$defs"
            ]
            if schema["additionalProperties"] is True:
                des.append("str, Any")
            return "Dict(" + ", ".join(des) + ")"
        return _orig_j2p(schema, defs)

    _gc_utils._json_schema_to_python_type = _patched_j2p

    # Also patch json_schema_to_python_type (the public wrapper)
    _orig_j2p_pub = _gc_utils.json_schema_to_python_type

    def _patched_j2p_pub(schema):
        if not isinstance(schema, dict):
            return "Any"
        return _orig_j2p_pub(schema)

    _gc_utils.json_schema_to_python_type = _patched_j2p_pub

    print("[patch] gradio_client bool-schema bug patched ✓")
except Exception as _e:
    print(f"[patch] WARNING: gradio_client patch failed ({_e})")

# ── Patch 2: gradio.networking.url_ok always returns True on HF Spaces ────────
#
#  The ValueError "When localhost is not accessible..." is thrown when
#  gradio.networking.url_ok(local_url) returns False.  On HF Spaces the
#  container's reverse-proxy intercepts the port, so a head-request from
#  inside the container fails even though the server is running fine.
#  Returning True unconditionally is safe — the server either started or
#  start_server() would have already raised.
# ─────────────────────────────────────────────────────────────────────────────
try:
    import gradio.networking as _gn
    _gn.url_ok = lambda url: True
    print("[patch] gradio.networking.url_ok patched ✓")
except Exception as _e:
    print(f"[patch] WARNING: url_ok patch failed ({_e})")

# ── Patch 3: fix Gradio 4.44 calling TemplateResponse with old Starlette API ──
#
#  Gradio 4.44 calls:
#    templates.TemplateResponse(template_name, {"request": req, ...})
#  Starlette 1.0 changed the signature to:
#    TemplateResponse(request, name, context)
#  So Starlette 1.0 receives the context dict as `request`, the template name
#  is never set, and Jinja2 gets the dict as a cache key → unhashable error.
#
#  Fix: monkey-patch Jinja2Templates.TemplateResponse to detect the old-style
#  call (first arg is a string = template name) and rewrite it to new-style.
# -----------------------------------------------------------------------------
try:
    from starlette.templating import Jinja2Templates as _J2T
    _orig_tr = _J2T.TemplateResponse

    def _compat_tr(self, *args, **kwargs):
        # Old-style: TemplateResponse(name: str, context: dict)
        if args and isinstance(args[0], str):
            name = args[0]
            ctx  = args[1] if len(args) > 1 else kwargs.pop("context", {})
            req  = ctx.get("request")
            return _orig_tr(self, req, name, ctx, *args[2:], **kwargs)
        return _orig_tr(self, *args, **kwargs)

    _J2T.TemplateResponse = _compat_tr
    print("[patch] Jinja2Templates.TemplateResponse patched for Starlette 1.0 compat ✓")
except Exception as _e:
    print(f"[patch] WARNING: TemplateResponse patch failed ({_e})")

# ── Now safe to import gradio ─────────────────────────────────────────────────
import gradio as gr
import numpy as np
import pdfplumber
import soundfile as sf
import torch
from kokoro import KPipeline
from pydub import AudioSegment

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[startup] Running on: {DEVICE}")

# ── TTS model ─────────────────────────────────────────────────────────────────
print("[startup] Loading Kokoro-82M TTS pipeline …")
_tts_pipeline = KPipeline(lang_code="a")  # 'a' = American English
SAMPLE_RATE: int = 24000  # Kokoro outputs at 24 kHz
TTS_VOICE: str = "af_heart"
TTS_SPEED: float = 1.0
print(f"[startup] TTS ready — sample rate: {SAMPLE_RATE} Hz")

# ── EasyOCR — lazy-loaded ─────────────────────────────────────────────────────
_ocr_reader = None


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        print("[ocr] Loading EasyOCR …")
        _ocr_reader = easyocr.Reader(["en"], gpu=(DEVICE.type == "cuda"), verbose=False)
        print("[ocr] EasyOCR ready.")
    return _ocr_reader


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — FILE INGESTION & TEXT EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

SUPPORTED = {".txt", ".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def extract_text(file_path: str) -> str:
    path = Path(file_path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED:
        raise gr.Error(f"Unsupported file type '{suffix}'. Upload .txt, .pdf, .png, or .jpg.")
    if suffix == ".txt":
        return _extract_txt(path)
    elif suffix == ".pdf":
        return _extract_pdf(path)
    else:
        return _extract_image(path)


def _extract_txt(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    print(f"[extract] TXT — {len(text):,} chars")
    return text


def _extract_pdf(path):
    pages_text = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for n, page in enumerate(pdf.pages, 1):
            text = page.extract_text() or ""
            if text.strip():
                pages_text.append(text)
            else:
                print(f"[extract] PDF page {n}/{total}: no text → OCR")
                img = page.to_image(resolution=200).original
                pages_text.append(" ".join(_get_ocr_reader().readtext(np.array(img), detail=0)))
    combined = "\n".join(pages_text)
    print(f"[extract] PDF — {total} pages → {len(combined):,} chars")
    return combined


def _extract_image(path):
    results = _get_ocr_reader().readtext(str(path), detail=0)
    text = " ".join(results)
    print(f"[extract] Image OCR — {len(results)} regions → {len(text):,} chars")
    return text


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — TEXT PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

MAX_CHUNK = 400

_RE_URL      = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL    = re.compile(r"\S+@\S+\.\S+")
_RE_PAGENUM  = re.compile(r"(?m)^\s*-?\s*\d+\s*-?\s*$")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_MULTI_SP = re.compile(r"[ \t]{2,}")
_RE_SPECIAL  = re.compile(r"[^\w\s.,!?\''\-:;\"()\[\]\n]")
_RE_SENT     = re.compile(r"(?<=[.!?])\s+")


def clean_text(text):
    text = unicodedata.normalize("NFKC", text)
    for pat, sub in [(_RE_URL, ""), (_RE_EMAIL, ""), (_RE_PAGENUM, ""),
                     (_RE_SPECIAL, " "), (_RE_MULTI_SP, " "), (_RE_MULTI_NL, "\n\n")]:
        text = pat.sub(sub, text)
    return text.strip()


def chunk_text(text):
    chunks, current = [], ""
    for sent in _RE_SENT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(current) + len(sent) + 1 <= MAX_CHUNK:
            current = (current + " " + sent).strip()
        else:
            if current:
                chunks.append(current)
            if len(sent) > MAX_CHUNK:
                chunks.extend(textwrap.wrap(sent, MAX_CHUNK))
                current = ""
            else:
                current = sent
    if current:
        chunks.append(current)
    print(f"[preprocess] {len(chunks)} chunks")
    return chunks


def preprocess(raw_text):
    cleaned = clean_text(raw_text)
    if not cleaned:
        raise gr.Error("No readable text found after cleaning.")
    return chunk_text(cleaned)


# ══════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — TTS SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════

SILENCE = np.zeros(int(SAMPLE_RATE * 0.40), dtype=np.float32)


def _infer(text):
    """Generate audio for a single text chunk using Kokoro-82M."""
    parts = []
    for _, _, audio in _tts_pipeline(text, voice=TTS_VOICE, speed=TTS_SPEED):
        if audio is not None and len(audio) > 0:
            if hasattr(audio, "numpy"):
                audio = audio.numpy()
            parts.append(np.asarray(audio, dtype=np.float32))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def synthesize(chunks, output_path):
    parts = []
    for i, chunk in enumerate(chunks, 1):
        print(f"[tts] {i}/{len(chunks)}: {chunk[:60]!r}")
        parts.append(_infer(chunk))
        if i < len(chunks):
            parts.append(SILENCE)
    full = np.concatenate(parts)
    wav = output_path.replace(".mp3", "_tmp.wav")
    sf.write(wav, full, SAMPLE_RATE)
    AudioSegment.from_wav(wav).export(output_path, format="mp3", bitrate="192k")
    os.remove(wav)
    print(f"[tts] Done: {len(full)/SAMPLE_RATE:.1f}s, {os.path.getsize(output_path)//1024}KB")
    return output_path


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(file_path):
    if file_path is None:
        return None, None, "⚠️ Please upload a file."
    try:
        raw    = extract_text(file_path)
        chunks = preprocess(raw)
        processed_text = "\n\n".join(chunks)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            mp3_path = f.name
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False,
                                         mode="w", encoding="utf-8") as f:
            txt_path = f.name
            f.write(processed_text)

        synthesize(chunks, mp3_path)
        size_kb  = os.path.getsize(mp3_path) / 1024
        duration = AudioSegment.from_mp3(mp3_path).duration_seconds
        return (mp3_path, txt_path,
                f"✅ Done! Duration: {duration:.1f}s  |  Size: {size_kb:.1f} KB")
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(str(exc)) from exc


# ══════════════════════════════════════════════════════════════════════════════
#  GRADIO UI
# ══════════════════════════════════════════════════════════════════════════════

_DESC = """
## 📚 Document → Speech

Upload a **text, PDF, or image** file and receive a spoken-word MP3 together with the processed text file.

| Format | Method |
|--------|--------|
| `.txt` | Direct read |
| `.pdf` (digital) | pdfplumber |
| `.pdf` (scanned) | EasyOCR |
| `.png` / `.jpg` | EasyOCR |

**Model:** Kokoro-82M (StyleTTS2, 24 kHz) · ⚠️ *CPU — large files take a few minutes*
"""

with gr.Blocks(title="Document to Speech", theme=gr.themes.Soft()) as demo:
    gr.Markdown(_DESC)
    with gr.Row():
        with gr.Column():
            file_in = gr.File(
                label="Upload file",
                file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg"],
                type="filepath",
            )
            btn = gr.Button("🔊 Convert to Speech", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Generated speech", type="filepath")
            text_out  = gr.File(label="Processed text (.txt)")
            status    = gr.Textbox(label="Status", interactive=False,
                                   placeholder="Status will appear here…")

    btn.click(fn=run_pipeline, inputs=file_in, outputs=[audio_out, text_out, status])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
