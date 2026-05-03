"""
pipeline/lang_detector.py
==========================
Stage 4 — Language Detection + Translation (Optional)

Uses:
  - langdetect (Google CLD2 Python port) for fast language identification.
  - facebook/nllb-200-distilled-600M (Meta NLLB) for translation to English
    across 200+ languages.

Translation is entirely optional and disabled by default in config.yaml.
Non-English books can still be TTS'd in their source language if a
compatible TTS voice is available.
"""

import logging
from typing import Optional

log = logging.getLogger(__name__)

# NLLB language code map — ISO 639-1 → NLLB BCP-47 script tag
# Covers the most common book languages. Extend as needed.
_LANG_MAP: dict[str, str] = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "es": "spa_Latn",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "nl": "nld_Latn",
    "ru": "rus_Cyrl",
    "zh-cn": "zho_Hans",
    "zh-tw": "zho_Hant",
    "ja": "jpn_Jpan",
    "ko": "kor_Hang",
    "ar": "arb_Arab",
    "hi": "hin_Deva",
    "bn": "ben_Beng",
    "tr": "tur_Latn",
    "pl": "pol_Latn",
    "sv": "swe_Latn",
    "no": "nob_Latn",
    "da": "dan_Latn",
    "fi": "fin_Latn",
    "el": "ell_Grek",
    "he": "heb_Hebr",
    "uk": "ukr_Cyrl",
    "cs": "ces_Latn",
    "ro": "ron_Latn",
    "hu": "hun_Latn",
    "sk": "slk_Latn",
    "bg": "bul_Cyrl",
    "hr": "hrv_Latn",
    "vi": "vie_Latn",
    "th": "tha_Thai",
    "id": "ind_Latn",
    "ms": "zsm_Latn",
    "fa": "pes_Arab",
    "ur": "urd_Arab",
    "sw": "swh_Latn",
}


class LanguageDetector:
    """Detects the primary language of a text sample using langdetect."""

    def __init__(self):
        try:
            import langdetect as _ld
            self._ld = _ld
        except ImportError as exc:
            raise ImportError("pip install langdetect") from exc

    def detect(self, text: str) -> str:
        """
        Detect the language of the provided text.

        Parameters
        ----------
        text : Sample of book text (at least a few hundred characters for accuracy).

        Returns
        -------
        ISO 639-1 language code string, e.g. "en", "fr", "de".
        Returns "unknown" on failure.
        """
        # Use the first 3000 characters as a representative sample
        sample = text[:3000].strip()
        if not sample:
            return "unknown"
        try:
            lang = self._ld.detect(sample)
            log.info("Detected language: '%s'", lang)
            return lang
        except Exception as exc:
            log.warning("Language detection failed: %s — defaulting to 'unknown'", exc)
            return "unknown"

    def detect_with_confidence(self, text: str) -> list[dict]:
        """
        Return a ranked list of (language, probability) tuples.

        Returns
        -------
        List of dicts: [{"lang": "en", "prob": 0.99}, ...]
        """
        sample = text[:3000].strip()
        if not sample:
            return []
        try:
            probabilities = self._ld.detect_langs(sample)
            return [{"lang": str(p.lang), "prob": round(p.prob, 4)} for p in probabilities]
        except Exception as exc:
            log.warning("Language probability detection failed: %s", exc)
            return []


class NLLBTranslator:
    """
    Translate text using facebook/nllb-200-distilled-600M.

    Handles very long texts by splitting into sentence-level chunks so
    that the NLLB sequence length limit (1024 tokens) is not exceeded.

    Parameters
    ----------
    model_id : HuggingFace model ID (distilled-600M or 1.3B).
    device   : "cpu" | "cuda" | "mps"
    """

    def __init__(
        self,
        model_id: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
    ):
        try:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise ImportError("pip install transformers") from exc

        log.info("Loading NLLB model '%s' on %s …", model_id, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self._model.to(device)
        self._model.eval()
        self._device = device
        log.info("NLLB model loaded")

    def _split_sentences(self, text: str, max_chars: int = 800) -> list[str]:
        """Split text into sentence-level chunks for safe NLLB translation."""
        import re
        # Split on sentence-ending punctuation
        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: list[str] = []
        current = ""
        for sent in sentences:
            if len(current) + len(sent) + 1 > max_chars and current:
                chunks.append(current.strip())
                current = sent
            else:
                current = (current + " " + sent) if current else sent
        if current.strip():
            chunks.append(current.strip())
        return chunks

    def translate(
        self,
        text: str,
        src_lang: str,
        tgt_lang: str = "eng_Latn",
    ) -> str:
        """
        Translate text from src_lang to tgt_lang using NLLB.

        Parameters
        ----------
        text     : Source text to translate.
        src_lang : NLLB BCP-47 code, e.g. "fra_Latn".
        tgt_lang : NLLB BCP-47 code, default "eng_Latn" (English).

        Returns
        -------
        Translated text string.
        """
        import torch

        self._tokenizer.src_lang = src_lang
        target_lang_id = self._tokenizer.lang_code_to_id[tgt_lang]

        chunks = self._split_sentences(text)
        translated_chunks: list[str] = []

        log.info(
            "Translating %d chunk(s): %s → %s", len(chunks), src_lang, tgt_lang
        )
        for idx, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            inputs = self._tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True, max_length=512
            ).to(self._device)
            with torch.no_grad():
                output_ids = self._model.generate(
                    **inputs,
                    forced_bos_token_id=target_lang_id,
                    max_length=1024,
                )
            translated = self._tokenizer.decode(output_ids[0], skip_special_tokens=True)
            translated_chunks.append(translated)
            log.debug("  Chunk %d/%d translated", idx + 1, len(chunks))

        return " ".join(translated_chunks)


class LanguageProcessor:
    """
    High-level interface combining detection and optional translation.

    Parameters
    ----------
    enabled     : Enable translation (if False, returns raw text unchanged).
    target_lang : ISO 639-1 target language (default "en" for English).
    nllb_model  : NLLB model ID.
    device      : "cpu" | "cuda" | "mps"
    skip_langs  : Languages to skip even if translation is enabled.
    """

    def __init__(
        self,
        enabled: bool = False,
        target_lang: str = "en",
        nllb_model: str = "facebook/nllb-200-distilled-600M",
        device: str = "cpu",
        skip_langs: Optional[list[str]] = None,
    ):
        self.enabled = enabled
        self.target_lang = target_lang
        self.skip_langs = set(skip_langs or ["en"])
        self._detector = LanguageDetector()
        self._translator: Optional[NLLBTranslator] = None

        if enabled:
            self._translator = NLLBTranslator(model_id=nllb_model, device=device)

    def process(self, text: str) -> tuple[str, str]:
        """
        Detect language and optionally translate.

        Returns
        -------
        (processed_text, detected_language_code)
        """
        lang = self._detector.detect(text)

        if not self.enabled:
            log.info("Translation disabled — passing through language '%s'", lang)
            return text, lang

        if lang in self.skip_langs or lang == "unknown":
            log.info("Skipping translation for language '%s'", lang)
            return text, lang

        nllb_src = _LANG_MAP.get(lang)
        nllb_tgt = _LANG_MAP.get(self.target_lang, "eng_Latn")

        if nllb_src is None:
            log.warning("No NLLB code for language '%s' — skipping translation", lang)
            return text, lang

        log.info("Translating from '%s' (%s) → '%s' (%s)", lang, nllb_src, self.target_lang, nllb_tgt)
        translated = self._translator.translate(text, src_lang=nllb_src, tgt_lang=nllb_tgt)
        return translated, lang


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Language detect + translate (Stage 4)")
    parser.add_argument("--input",   required=True, help="Input text file")
    parser.add_argument("--output",  required=True, help="Output text file")
    parser.add_argument("--translate", action="store_true", help="Enable translation to English")
    parser.add_argument("--model",   default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--device",  default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    processor = LanguageProcessor(
        enabled=args.translate,
        nllb_model=args.model,
        device=args.device,
    )
    result_text, detected_lang = processor.process(text)

    print(f"Detected language: {detected_lang}")

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result_text)

    print(f"Output saved → {args.output}")
