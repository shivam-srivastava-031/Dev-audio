"""
pipeline/llm_corrector.py
==========================
Stage 3 — LLM Post-Processing (Context-aware OCR correction)

Uses Llama 3.1 (via Ollama or HuggingFace Transformers) to:
  1. Fix character/word-level OCR noise (broken words, rn→m, etc.)
  2. Normalise formatting (orphaned headers, punctuation)

A Levenshtein-based hallucination guard rejects chunks where the LLM
changes more than `max_change_ratio` of the original characters.

Backends:
  - "ollama"       → Requires: ollama pull llama3.1 (recommended, CPU-friendly)
  - "transformers" → Direct HuggingFace inference (GPU recommended)
  - "skip"         → Returns raw OCR text unchanged
"""

import logging
import re
import unicodedata
from typing import List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_CORRECTION_PROMPT = """\
You are correcting OCR errors in a scanned book.
Fix ONLY clear OCR mistakes: broken words, wrong characters (e.g. "rn" instead of "m"),
extra spaces, hyphenation artifacts, and encoding issues.
Do NOT rephrase, summarise, paraphrase, add commentary, or change content.
Preserve all paragraph breaks and punctuation exactly as they appear unless clearly wrong.

OCR TEXT:
{ocr_text}

CORRECTED TEXT:"""

_FORMAT_PROMPT = """\
You are cleaning up book text. Remove only:
- Orphaned single-character lines that are clearly page numbers or artefacts
- Duplicate lines (exact repeats within 3 lines of each other)
- Stray headers/footers that break into body text (e.g. "Page 42", "Chapter 3")
Do NOT rephrase or summarise any content. Return the cleaned text only.

TEXT:
{text}

CLEANED TEXT:"""


# ---------------------------------------------------------------------------
# Levenshtein distance (pure-Python, no dependency)
# ---------------------------------------------------------------------------
def _levenshtein_ratio(s1: str, s2: str) -> float:
    """Return normalised edit distance between two strings (0.0 = identical)."""
    if not s1 and not s2:
        return 0.0
    if not s1 or not s2:
        return 1.0
    # Truncate for performance (compare first 2000 chars)
    a, b = s1[:2000], s2[:2000]
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    dist = prev[n]
    return dist / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _OllamaBackend:
    def __init__(self, model: str = "llama3.1"):
        try:
            import ollama as _ollama
        except ImportError as exc:
            raise ImportError(
                "Ollama Python client not found.\n"
                "  pip install ollama\n"
                "  Then: ollama pull llama3.1"
            ) from exc
        self._ollama = _ollama
        self.model = model
        log.info("LLM backend: Ollama model '%s'", model)

    def generate(self, prompt: str) -> str:
        response = self._ollama.generate(model=self.model, prompt=prompt, stream=False)
        return response["response"].strip()


class _TransformersBackend:
    def __init__(self, model_id: str = "meta-llama/Llama-3.1-8B-Instruct", device: str = "cpu"):
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError as exc:
            raise ImportError("pip install transformers torch") from exc

        import torch
        log.info("Loading LLM '%s' on %s (this may take a while)…", model_id, device)
        self._tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            device_map=device,
        )
        self._model.eval()
        self._device = device
        log.info("LLM loaded")

    def generate(self, prompt: str) -> str:
        import torch
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                temperature=1.0,
            )
        # Decode only the newly generated tokens (skip the prompt)
        generated = out[0][inputs["input_ids"].shape[-1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True).strip()


class _SkipBackend:
    def generate(self, prompt: str) -> str:
        # Extract the OCR text from the prompt and return it unchanged
        marker = "OCR TEXT:\n"
        if marker in prompt:
            return prompt.split(marker)[-1].replace("CORRECTED TEXT:", "").strip()
        return prompt


# ---------------------------------------------------------------------------
# LLMCorrector
# ---------------------------------------------------------------------------

class LLMCorrector:
    """
    Correct OCR text using an LLM with hallucination protection.

    Parameters
    ----------
    backend         : "ollama" | "transformers" | "skip"
    model           : Model name/ID (used by ollama or transformers backend).
    device          : "cpu" | "cuda" | "mps"
    max_chunk_tokens: Approximate token budget per correction chunk.
    max_change_ratio: Reject LLM output if edit ratio > this value.
    passes          : Number of correction passes (1 = fast, 2 = thorough).
    """

    # Roughly 4 chars per token (conservative estimate)
    _CHARS_PER_TOKEN = 4

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.1",
        device: str = "cpu",
        max_chunk_tokens: int = 512,
        max_change_ratio: float = 0.15,
        passes: int = 2,
    ):
        self.max_chunk_chars = max_chunk_tokens * self._CHARS_PER_TOKEN
        self.max_change_ratio = max_change_ratio
        self.passes = passes

        backend = backend.lower()
        if backend == "ollama":
            self._backend = _OllamaBackend(model=model)
        elif backend == "transformers":
            self._backend = _TransformersBackend(model_id=model, device=device)
        elif backend == "skip":
            self._backend = _SkipBackend()
            log.info("LLM corrector disabled (skip mode)")
        else:
            log.warning("Unknown LLM backend '%s' — falling back to skip", backend)
            self._backend = _SkipBackend()

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text at paragraph boundaries, keeping chunks under the token budget."""
        paragraphs = re.split(r"\n{2,}", text.strip())
        chunks: List[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 > self.max_chunk_chars and current:
                chunks.append(current.strip())
                current = para
            else:
                current = (current + "\n\n" + para) if current else para
        if current.strip():
            chunks.append(current.strip())
        return chunks

    # ------------------------------------------------------------------
    # Correction logic
    # ------------------------------------------------------------------

    def _correct_chunk(self, text: str, pass_num: int) -> str:
        """Run one correction pass on a single chunk."""
        prompt_template = _CORRECTION_PROMPT if pass_num == 1 else _FORMAT_PROMPT
        key = "ocr_text" if pass_num == 1 else "text"
        prompt = prompt_template.format(**{key: text})

        try:
            corrected = self._backend.generate(prompt)
        except Exception as exc:
            log.warning("LLM inference failed on chunk — returning raw text. Error: %s", exc)
            return text

        # Hallucination guard
        change_ratio = _levenshtein_ratio(text, corrected)
        if change_ratio > self.max_change_ratio:
            log.warning(
                "LLM changed %.1f%% of chunk (limit %.0f%%) — rejecting correction",
                change_ratio * 100, self.max_change_ratio * 100,
            )
            return text

        return corrected

    def correct_text(self, ocr_text: str) -> str:
        """
        Run multi-pass LLM correction on a full OCR text string.

        Parameters
        ----------
        ocr_text : Raw text extracted by TrOCR.

        Returns
        -------
        Corrected text.
        """
        if not ocr_text.strip():
            return ocr_text

        text = ocr_text
        for pass_num in range(1, self.passes + 1):
            chunks = self._split_into_chunks(text)
            log.info(
                "LLM correction pass %d/%d — %d chunk(s)",
                pass_num, self.passes, len(chunks),
            )
            corrected_chunks: List[str] = []
            for idx, chunk in enumerate(chunks):
                corrected = self._correct_chunk(chunk, pass_num=pass_num)
                corrected_chunks.append(corrected)
                log.debug("  Chunk %d/%d corrected", idx + 1, len(chunks))
            text = "\n\n".join(corrected_chunks)

        return text

    def correct_page_texts(self, page_texts: List[str]) -> List[str]:
        """
        Correct OCR text for multiple pages.

        Parameters
        ----------
        page_texts : List of raw OCR text strings (one per page).

        Returns
        -------
        List of corrected text strings.
        """
        corrected: List[str] = []
        for i, text in enumerate(page_texts):
            log.info("LLM correcting page %d/%d …", i + 1, len(page_texts))
            corrected.append(self.correct_text(text))
        return corrected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, sys

    parser = argparse.ArgumentParser(description="LLM OCR correction (Stage 3)")
    parser.add_argument("--input",   required=True, help="Path to raw OCR text file")
    parser.add_argument("--output",  required=True, help="Path to save corrected text")
    parser.add_argument("--backend", default="ollama", choices=["ollama","transformers","skip"])
    parser.add_argument("--model",   default="llama3.1")
    parser.add_argument("--passes",  type=int, default=2)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    with open(args.input, "r", encoding="utf-8") as f:
        raw_text = f.read()

    corrector = LLMCorrector(backend=args.backend, model=args.model, passes=args.passes)
    corrected = corrector.correct_text(raw_text)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(corrected)

    print(f"Correction complete → {args.output}")
