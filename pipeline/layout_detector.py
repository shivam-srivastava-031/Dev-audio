"""
pipeline/layout_detector.py
============================
Stage 1 — Layout Detection

Detects and classifies regions on each page image (headers, footers,
body text, figures, tables) so that OCR only processes clean text blocks
in the correct reading order.

Supported backends:
  - "doclayout"   → DocLayout-YOLO (general books, fast)
  - "nougat"      → Meta Nougat (academic / scientific PDFs)
  - "layoutlmv3"  → LayoutLMv3 (region classification on detected boxes)
  - "skip"        → Pass the full image as a single body region (no detection)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from PIL import Image

log = logging.getLogger(__name__)

# Region types we consider as useful text for OCR
TEXT_REGION_TYPES = {"body", "title", "chapter_title", "caption", "text", "plain text"}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Region:
    """A detected layout region on a page."""
    region_type: str                     # "body" | "title" | "header" | "footer" | "figure" | ...
    bbox: Tuple[int, int, int, int]      # (x1, y1, x2, y2) in pixels
    confidence: float = 1.0
    text: Optional[str] = None           # populated later by OCR

    def to_dict(self) -> dict:
        return {
            "type": self.region_type,
            "bbox": list(self.bbox),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class PageLayout:
    """All detected regions for a single page."""
    page_number: int                     # 1-indexed
    image_size: Tuple[int, int]          # (width, height) in pixels
    regions: List[Region] = field(default_factory=list)

    def text_regions(self, keep_types: Optional[List[str]] = None) -> List[Region]:
        """Return only regions suitable for OCR."""
        allowed = set(keep_types) if keep_types else TEXT_REGION_TYPES
        return [r for r in self.regions if r.region_type.lower() in allowed]

    def to_dict(self) -> dict:
        return {
            "page": self.page_number,
            "image_size": {"width": self.image_size[0], "height": self.image_size[1]},
            "regions": [r.to_dict() for r in self.regions],
        }


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

class _SkipDetector:
    """No-op detector — treats each full page as a single body region."""

    def detect(self, image: Image.Image, page_number: int) -> PageLayout:
        w, h = image.size
        log.debug("Page %d: skip mode — returning full-page body region", page_number)
        return PageLayout(
            page_number=page_number,
            image_size=(w, h),
            regions=[Region("body", (0, 0, w, h), confidence=1.0)],
        )


class _DocLayoutDetector:
    """
    DocLayout-YOLO backend for fast bounding-box region detection.
    Requires: pip install doclayout-yolo ultralytics
    Weights are auto-downloaded from HuggingFace if not cached locally.
    """

    def __init__(self, weights_path: Optional[str] = None):
        try:
            from doclayout_yolo import YOLOv10
        except ImportError as exc:
            raise ImportError(
                "DocLayout-YOLO not installed.\n"
                "  pip install doclayout-yolo ultralytics"
            ) from exc

        if weights_path and Path(weights_path).exists():
            log.info("Loading DocLayout-YOLO weights from %s", weights_path)
            self._model = YOLOv10(weights_path)
        else:
            log.info("Downloading DocLayout-YOLO weights from HuggingFace...")
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id="juliozhao/DocLayout-YOLO-DocStructBench",
                filename="doclayout_yolo_docstructbench_imgsz1024.pt",
            )
            self._model = YOLOv10(path)
        log.info("DocLayout-YOLO model ready")

    # DocLayout-YOLO class names → normalised region type mapping
    _CLASS_MAP: Dict[str, str] = {
        "title": "chapter_title",
        "plain text": "body",
        "abandon": "footer",           # usually page numbers / headers to skip
        "figure": "figure",
        "figure_caption": "caption",
        "table": "table",
        "table_caption": "caption",
        "table_footnote": "footer",
        "isolate_formula": "figure",
        "formula_caption": "caption",
    }

    def detect(self, image: Image.Image, page_number: int) -> PageLayout:
        w, h = image.size
        results = self._model.predict(image, imgsz=1024, conf=0.25, verbose=False)
        regions: List[Region] = []
        if results and len(results[0].boxes):
            boxes = results[0].boxes
            for box in boxes:
                cls_name = self._model.names[int(box.cls)]
                region_type = self._CLASS_MAP.get(cls_name, cls_name)
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                regions.append(Region(region_type, (x1, y1, x2, y2), confidence=conf))
        else:
            # Fallback: full page
            regions = [Region("body", (0, 0, w, h), confidence=1.0)]

        # Sort regions top-to-bottom, left-to-right (reading order)
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
        log.debug("Page %d: %d regions detected", page_number, len(regions))
        return PageLayout(page_number, (w, h), regions)


class _NougatDetector:
    """
    Meta Nougat backend — OCR-free document understanding for academic PDFs.
    Returns the full page text as a single region (Nougat handles layout internally).
    Note: Nougat performs OCR itself; in this pipeline its output text is used as
    the OCR result and the Region will carry pre-filled text.
    """

    def __init__(self, model_id: str = "facebook/nougat-base", device: str = "cpu"):
        try:
            from transformers import NougatProcessor, VisionEncoderDecoderModel
        except ImportError as exc:
            raise ImportError("transformers required: pip install transformers") from exc

        log.info("Loading Nougat model '%s' on %s …", model_id, device)
        self._processor = NougatProcessor.from_pretrained(model_id)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_id)
        self._device = device
        self._model.to(device)
        self._model.eval()
        log.info("Nougat model ready")

    def detect(self, image: Image.Image, page_number: int) -> PageLayout:
        import torch
        w, h = image.size
        pixel_values = self._processor(image, return_tensors="pt").pixel_values.to(self._device)
        with torch.no_grad():
            outputs = self._model.generate(
                pixel_values,
                min_length=1,
                max_new_tokens=2048,
                bad_words_ids=[[self._processor.tokenizer.unk_token_id]],
            )
        page_text = self._processor.batch_decode(outputs, skip_special_tokens=True)[0]
        page_text = self._processor.post_process_generation(page_text, fix_markdown=False)

        region = Region("body", (0, 0, w, h), confidence=1.0, text=page_text)
        log.debug("Page %d: Nougat extracted %d chars", page_number, len(page_text))
        return PageLayout(page_number, (w, h), regions=[region])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class LayoutDetector:
    """
    Unified layout detection interface.

    Parameters
    ----------
    engine          : One of "doclayout", "nougat", "layoutlmv3", "skip".
    weights_path    : Optional path to pre-downloaded model weights.
    device          : "cpu" | "cuda" | "mps"
    keep_types      : Region types to retain for OCR (others are discarded).
    """

    def __init__(
        self,
        engine: str = "doclayout",
        weights_path: Optional[str] = None,
        device: str = "cpu",
        keep_types: Optional[List[str]] = None,
    ):
        self.engine = engine.lower()
        self.keep_types = keep_types or list(TEXT_REGION_TYPES)

        if self.engine == "skip":
            self._backend = _SkipDetector()
        elif self.engine == "doclayout":
            self._backend = _DocLayoutDetector(weights_path=weights_path)
        elif self.engine == "nougat":
            self._backend = _NougatDetector(device=device)
        else:
            log.warning("Unknown layout engine '%s' — falling back to 'skip'", engine)
            self._backend = _SkipDetector()

    def detect_page(self, image: Image.Image, page_number: int = 1) -> PageLayout:
        """Detect layout regions on a single page image."""
        return self._backend.detect(image, page_number)

    def detect_all(self, images: List[Image.Image]) -> List[PageLayout]:
        """Detect layout on a list of page images (1-indexed page numbers)."""
        layouts: List[PageLayout] = []
        for idx, img in enumerate(images):
            page_num = idx + 1
            layout = self.detect_page(img, page_number=page_num)
            layouts.append(layout)
            log.info(
                "Layout page %d/%d — %d text regions",
                page_num, len(images),
                len(layout.text_regions(self.keep_types)),
            )
        return layouts

    def save_layouts(self, layouts: List[PageLayout], output_path: str | os.PathLike) -> None:
        """Save all page layouts to a JSON file for inspection or downstream use."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        data = [l.to_dict() for l in layouts]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        log.info("Layout data saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pipeline.pdf_to_images import pdf_to_images

    parser = argparse.ArgumentParser(description="Run layout detection on a PDF.")
    parser.add_argument("--pdf",    required=True)
    parser.add_argument("--engine", default="doclayout", choices=["doclayout","nougat","skip"])
    parser.add_argument("--dpi",    type=int, default=300)
    parser.add_argument("--out",    default="layout_output.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    images = pdf_to_images(args.pdf, dpi=args.dpi)
    detector = LayoutDetector(engine=args.engine)
    layouts = detector.detect_all(images)
    detector.save_layouts(layouts, args.out)
    print(f"Layout detection complete — {len(layouts)} pages → {args.out}")
