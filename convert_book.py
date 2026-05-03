"""
convert_book.py
===============
Main CLI entrypoint for the Book-to-Audio pipeline.

Runs all 8 stages in sequence:
  1. PDF → page images
  2. Layout detection
  3. OCR (TrOCR)
  4. LLM post-processing (OCR correction)
  5. Language detection + optional translation
  6. Semantic structuring (chapters/sections/paragraphs)
  7. Emotion tagging + speaker segmentation
  8. TTS synthesis
  9. Audio stitching + M4B export

Usage examples:
    # Basic (all defaults, CPU)
    python convert_book.py --input my_book.pdf --output ./output/

    # With translation
    python convert_book.py --input french_book.pdf --translate --output ./output/

    # Skip layout detection (simple single-column books)
    python convert_book.py --input simple_book.pdf --no-layout --output ./output/

    # Use GPU and a specific TTS engine
    python convert_book.py --input book.pdf --device cuda --tts-engine xtts --output ./output/
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import yaml

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> dict:
    """Load pipeline configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        log.warning("config.yaml not found — using all defaults")
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def deep_get(config: dict, *keys, default=None):
    """Safely get a nested config value."""
    d = config
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
    return d


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="convert_book",
        description="Book-to-Audiobook pipeline — converts scanned PDFs to MP3/M4B.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument("--input",  required=True, help="Input PDF file path")
    parser.add_argument("--output", required=True, help="Output directory")

    # Book metadata
    parser.add_argument("--title",  default=None, help="Book title (inferred from PDF if not given)")
    parser.add_argument("--author", default="Unknown Author", help="Book author name")
    parser.add_argument("--cover",  default=None, help="Cover art image path (JPG/PNG)")

    # Stage flags
    parser.add_argument("--no-layout",   action="store_true",  help="Skip layout detection (Stage 1)")
    parser.add_argument("--no-llm",      action="store_true",  help="Skip LLM correction (Stage 3)")
    parser.add_argument("--translate",   action="store_true",  help="Enable translation to English (Stage 4)")
    parser.add_argument("--no-emotion",  action="store_true",  help="Skip emotion tagging (Stage 6)")

    # Model / engine config
    parser.add_argument("--ocr-model",   default=None, help="TrOCR model ID or path (Stage 2)")
    parser.add_argument("--layout-engine",default=None, choices=["doclayout","nougat","skip"])
    parser.add_argument("--llm-backend", default=None, choices=["ollama","transformers","skip"])
    parser.add_argument("--llm-model",   default=None, help="LLM model name/ID (Stage 3)")
    parser.add_argument("--tts-engine",  default=None, choices=["xtts","fish_speech","elevenlabs","gtts"])
    parser.add_argument("--tts-model",   default=None, help="TTS model name/URL (Stage 7)")

    # Runtime
    parser.add_argument("--device",  default=None, help="Inference device: cpu | cuda | mps")
    parser.add_argument("--dpi",     type=int, default=None, help="PDF render DPI (default: 300)")
    parser.add_argument("--config",  default="config.yaml", help="Config YAML path")
    parser.add_argument("--pages",   default=None,
                        help="Page range to process, e.g. '1-10' (1-indexed, inclusive)")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")

    return parser


# ---------------------------------------------------------------------------
# Intermediate file paths
# ---------------------------------------------------------------------------

class PipelinePaths:
    def __init__(self, output_dir: Path, temp_dir: Path):
        self.output_dir = output_dir
        self.temp_dir = temp_dir
        # Intermediate artifacts
        self.layout_json      = temp_dir / "layout.json"
        self.ocr_json         = temp_dir / "ocr_output.json"
        self.corrected_txt    = temp_dir / "corrected_text.txt"
        self.structured_json  = temp_dir / "book_structure.json"
        self.tagged_json      = temp_dir / "tagged_book.json"
        self.tts_dir          = temp_dir / "tts_chunks"

    def setup(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_stage_1_pdf(pdf_path: str, dpi: int, page_range=None) -> list:
    """Stage 0+1 helper: PDF → images."""
    from pipeline.pdf_to_images import pdf_to_images
    log.info("═" * 60)
    log.info("Stage 0 — PDF → Images (DPI=%d)", dpi)
    images = pdf_to_images(pdf_path, dpi=dpi, page_range=page_range)
    log.info("Pages loaded: %d", len(images))
    return images


def run_stage_1_layout(images: list, engine: str, device: str, config: dict, paths: PipelinePaths) -> list:
    """Stage 1: Layout detection."""
    from pipeline.layout_detector import LayoutDetector
    log.info("═" * 60)
    log.info("Stage 1 — Layout Detection (engine=%s)", engine)
    keep_types = deep_get(config, "layout", "keep_region_types", default=None)
    weights = deep_get(config, "layout", "doclayout_weights", default=None)
    detector = LayoutDetector(engine=engine, weights_path=weights, device=device, keep_types=keep_types)
    layouts = detector.detect_all(images)
    detector.save_layouts(layouts, paths.layout_json)
    return layouts


def run_stage_2_ocr(images: list, layouts: list, ocr_model: str, device: str,
                    config: dict, paths: PipelinePaths) -> list:
    """Stage 2: OCR."""
    from pipeline.ocr_engine import TrOCREngine
    log.info("═" * 60)
    log.info("Stage 2 — OCR (model=%s)", ocr_model)
    batch_size = deep_get(config, "ocr", "batch_size", default=4)
    keep_types = deep_get(config, "layout", "keep_region_types", default=None)
    engine = TrOCREngine(model_name_or_path=ocr_model, device=device, batch_size=batch_size)
    layouts = engine.ocr_all_pages(images, layouts, keep_types=keep_types)

    # Save OCR output with text
    output = []
    for layout in layouts:
        page_data = layout.to_dict()
        for region_data, region in zip(page_data["regions"], layout.regions):
            region_data["text"] = region.text or ""
        output.append(page_data)
    with open(paths.ocr_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return layouts


def layouts_to_text(layouts: list, keep_types=None) -> str:
    """Concatenate all region text from layouts into a single string."""
    pages: list[str] = []
    for layout in layouts:
        text_regions = layout.text_regions(keep_types)
        page_text = "\n\n".join(r.text for r in text_regions if r.text)
        if page_text:
            pages.append(page_text)
    return "\n\n".join(pages)


def run_stage_3_llm(raw_text: str, backend: str, model: str,
                    device: str, config: dict, paths: PipelinePaths) -> str:
    """Stage 3: LLM correction."""
    from pipeline.llm_corrector import LLMCorrector
    log.info("═" * 60)
    log.info("Stage 3 — LLM Correction (backend=%s)", backend)
    max_change = deep_get(config, "llm", "max_change_ratio", default=0.15)
    passes = deep_get(config, "llm", "passes", default=2)
    corrector = LLMCorrector(
        backend=backend, model=model, device=device,
        max_change_ratio=max_change, passes=passes,
    )
    corrected = corrector.correct_text(raw_text)
    with open(paths.corrected_txt, "w", encoding="utf-8") as f:
        f.write(corrected)
    return corrected


def run_stage_4_lang(text: str, enabled: bool, device: str, config: dict) -> tuple[str, str]:
    """Stage 4: Language detection + translation."""
    from pipeline.lang_detector import LanguageProcessor
    log.info("═" * 60)
    log.info("Stage 4 — Language Detection (translate=%s)", enabled)
    nllb_model = deep_get(config, "translation", "nllb_model",
                          default="facebook/nllb-200-distilled-600M")
    skip_langs = deep_get(config, "translation", "skip_langs", default=["en"])
    processor = LanguageProcessor(
        enabled=enabled,
        nllb_model=nllb_model,
        device=device,
        skip_langs=skip_langs,
    )
    return processor.process(text)


def run_stage_5_structure(text: str, book_title: str, config: dict,
                          paths: PipelinePaths):
    """Stage 5: Semantic structuring."""
    from pipeline.semantic_structurer import SemanticStructurer
    log.info("═" * 60)
    log.info("Stage 5 — Semantic Structuring")
    strategy = deep_get(config, "structure", "strategy", default="hybrid")
    patterns = deep_get(config, "structure", "chapter_patterns", default=None)
    structurer = SemanticStructurer(strategy=strategy, chapter_patterns=patterns)
    book = structurer.structure(text, book_title=book_title)
    book.save(paths.structured_json)
    return book


def run_stage_6_emotion(book, device: str, config: dict, paths: PipelinePaths) -> list:
    """Stage 6: Emotion tagging."""
    from pipeline.emotion_tagger import EmotionTagger
    log.info("═" * 60)
    log.info("Stage 6 — Emotion Tagging")
    emotion_model = deep_get(config, "emotion", "model",
                             default="j-hartmann/emotion-english-distilroberta-base")
    min_conf = deep_get(config, "emotion", "min_confidence", default=0.6)
    speaker_det = deep_get(config, "emotion", "speaker_detection", default=True)
    narrator_voice = deep_get(config, "emotion", "narrator_voice", default="narrator_default")
    tagger = EmotionTagger(
        model_id=emotion_model,
        device=device,
        min_confidence=min_conf,
        speaker_detection=speaker_det,
        narrator_voice=narrator_voice,
    )
    tagged = tagger.tag_book(book)
    tagger.save_tagged_book(tagged, paths.tagged_json, book_title=book.book_title)
    return tagged


def run_stage_7_tts(tagged_book_path: Path, output_dir: Path,
                    tts_engine: str, tts_model: str, device: str,
                    language: str, config: dict) -> dict:
    """Stage 7: TTS synthesis."""
    from pipeline.tts_engine import TTSEngine
    log.info("═" * 60)
    log.info("Stage 7 — TTS Synthesis (engine=%s)", tts_engine)

    voice_profiles = deep_get(config, "tts", "voice_profiles", default=None)
    emotion_adj = deep_get(config, "tts", "emotion_adjustments", default=None)
    max_chunk = deep_get(config, "tts", "max_chunk_words", default=300)
    el_key = deep_get(config, "tts", "elevenlabs_api_key", default="")

    engine = TTSEngine(
        engine=tts_engine,
        model_or_url=tts_model,
        device=device,
        language=language,
        max_chunk_words=max_chunk,
        voice_profiles=voice_profiles,
        emotion_adjustments=emotion_adj,
        elevenlabs_api_key=el_key,
    )
    return engine.synthesize_tagged_book(tagged_book_path, output_dir)


def run_stage_8_audio(wav_by_chapter: dict, paths: PipelinePaths, config: dict,
                      book_title: str, author: str, cover: Optional[str]) -> dict:
    """Stage 8: Audio stitching + M4B."""
    from pipeline.audio_builder import AudioBuilder
    log.info("═" * 60)
    log.info("Stage 8 — Audio Stitching + M4B")

    audio_cfg = deep_get(config, "audio", default={})
    builder = AudioBuilder(
        output_dir=paths.output_dir,
        tagged_book_path=paths.tagged_json,
        tts_output_dir=paths.tts_dir,
        pause_sentence_ms=audio_cfg.get("pause_between_sentences_ms", 200),
        pause_paragraph_ms=audio_cfg.get("pause_between_paragraphs_ms", 600),
        pause_chapter_ms=audio_cfg.get("pause_between_chapters_ms", 2000),
        target_lufs=audio_cfg.get("target_lufs", -16.0),
        true_peak=audio_cfg.get("true_peak", -1.5),
        mp3_bitrate=audio_cfg.get("mp3_bitrate", "128k"),
        cover_art=cover or audio_cfg.get("cover_art"),
        book_title=book_title,
        book_author=author,
    )
    return builder.build(wav_by_chapter)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = build_parser()
    args = parser.parse_args()

    # Logging setup
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    start_time = time.time()
    config = load_config(args.config)

    # Resolve settings (CLI args take priority over config.yaml)
    device        = args.device or deep_get(config, "general", "device", default="cpu")
    dpi           = args.dpi or deep_get(config, "ocr", "dpi", default=300)
    layout_engine = ("skip" if args.no_layout else
                     args.layout_engine or deep_get(config, "layout", "engine", default="doclayout"))
    ocr_model     = args.ocr_model or deep_get(config, "ocr", "model",
                                               default="microsoft/trocr-large-printed")
    llm_backend   = ("skip" if args.no_llm else
                     args.llm_backend or deep_get(config, "llm", "backend", default="ollama"))
    llm_model     = args.llm_model or deep_get(config, "llm", "ollama_model", default="llama3.1")
    translate     = args.translate or deep_get(config, "translation", "enabled", default=False)
    tts_engine    = args.tts_engine or deep_get(config, "tts", "engine", default="xtts")
    tts_model     = args.tts_model or deep_get(config, "tts", "xtts_model",
                                               default="tts_models/multilingual/multi-dataset/xtts_v2")
    language      = deep_get(config, "tts", "language", default="en")

    # Page range
    page_range = None
    if args.pages:
        parts = args.pages.split("-")
        if len(parts) == 2:
            page_range = (int(parts[0]) - 1, int(parts[1]))  # convert to 0-indexed

    # Paths
    output_dir = Path(args.output)
    temp_dir   = output_dir / deep_get(config, "general", "temp_dir", default=".pipeline_temp")
    paths = PipelinePaths(output_dir=output_dir, temp_dir=temp_dir)
    paths.setup()

    book_title = args.title or Path(args.input).stem.replace("_", " ").title()

    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║         BOOK-TO-AUDIO PIPELINE — STARTING                ║")
    log.info("╠══════════════════════════════════════════════════════════╣")
    log.info("  Input:   %s", args.input)
    log.info("  Output:  %s", output_dir)
    log.info("  Title:   %s", book_title)
    log.info("  Device:  %s | DPI: %d | Pages: %s", device, dpi, args.pages or "all")
    log.info("╚══════════════════════════════════════════════════════════╝")

    # ── Stage 0+1: PDF → images ──────────────────────────────────────────
    images = run_stage_1_pdf(args.input, dpi=dpi, page_range=page_range)

    # ── Stage 1: Layout detection ─────────────────────────────────────────
    layouts = run_stage_1_layout(images, layout_engine, device, config, paths)

    # ── Stage 2: OCR ──────────────────────────────────────────────────────
    keep_types = deep_get(config, "layout", "keep_region_types", default=None)
    layouts = run_stage_2_ocr(images, layouts, ocr_model, device, config, paths)
    raw_text = layouts_to_text(layouts, keep_types=keep_types)
    log.info("Total OCR text: %d characters", len(raw_text))

    # ── Stage 3: LLM correction ───────────────────────────────────────────
    corrected_text = run_stage_3_llm(raw_text, llm_backend, llm_model, device, config, paths)

    # ── Stage 4: Language detection + translation ─────────────────────────
    processed_text, detected_lang = run_stage_4_lang(corrected_text, translate, device, config)
    log.info("Language: %s", detected_lang)

    # ── Stage 5: Semantic structuring ─────────────────────────────────────
    book = run_stage_5_structure(processed_text, book_title, config, paths)

    # ── Stage 6: Emotion tagging ──────────────────────────────────────────
    if not args.no_emotion:
        run_stage_6_emotion(book, device, config, paths)
    else:
        log.info("Stage 6 — Emotion tagging SKIPPED")
        # Write minimal tagged JSON for downstream stages
        _write_passthrough_tagged(book, paths.tagged_json)

    # ── Stage 7: TTS ──────────────────────────────────────────────────────
    wav_by_chapter = run_stage_7_tts(
        paths.tagged_json, paths.tts_dir,
        tts_engine, tts_model, device, language, config,
    )

    # ── Stage 8: Audio stitching ──────────────────────────────────────────
    output_files = run_stage_8_audio(
        wav_by_chapter, paths, config,
        book_title=book_title, author=args.author, cover=args.cover,
    )

    # ── Done ──────────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    log.info("╔══════════════════════════════════════════════════════════╗")
    log.info("║  PIPELINE COMPLETE — %.1f minutes                        ║", elapsed / 60)
    log.info("╠══════════════════════════════════════════════════════════╣")
    for key, val in output_files.items():
        log.info("  %-20s: %s", key, val)
    log.info("╚══════════════════════════════════════════════════════════╝")
    print("\nDone! Output files:")
    for key, val in output_files.items():
        print(f"  {key}: {val}")


def _write_passthrough_tagged(book, tagged_json_path: Path) -> None:
    """Write a minimal tagged_book.json without emotion data (Stage 6 skipped)."""
    chapters = []
    for ch in book.chapters:
        sections = []
        for sec in ch.sections:
            paras = []
            for para in sec.paragraphs:
                sentences = [{"text": s, "emotion": "neutral", "emotion_confidence": 1.0,
                              "is_dialogue": False, "speaker_id": "narrator_default"}
                             for s in para.text.split(". ") if s.strip()]
                paras.append({"paragraph_emotion": "neutral", "sentences": sentences})
            sections.append({"heading": sec.heading, "paragraphs": paras})
        chapters.append({
            "chapter_number": ch.number,
            "chapter_title": ch.title,
            "sections": sections,
        })
    data = {"book_title": book.book_title, "character_voice_map": {}, "chapters": chapters}
    tagged_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(tagged_json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
