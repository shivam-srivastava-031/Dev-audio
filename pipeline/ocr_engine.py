"""
pipeline/ocr_engine.py
=======================
Stage 2 — OCR using (fine-tuned) TrOCR

Crops each detected layout region from the page image and runs
microsoft/trocr-large-printed (or your fine-tuned checkpoint) to
extract the text content.

Key design decisions:
- Processes one region at a time to handle variable region sizes.
- Batches multiple regions per forward pass for efficiency.
- Falls back to base model if fine-tuned checkpoint is unavailable.
- Regions that already have text (e.g. from Nougat) are passed through.
"""

import logging
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image

from pipeline.layout_detector import PageLayout, Region

log = logging.getLogger(__name__)


class TrOCREngine:
    """
    OCR engine wrapping microsoft/trocr-large-printed (or a fine-tuned
    checkpoint thereof).

    Parameters
    ----------
    model_name_or_path : HuggingFace model ID or local directory.
    device             : "cpu" | "cuda" | "mps"
    batch_size         : Number of region images to process per forward pass.
    max_new_tokens     : Maximum generated tokens per region (controls output length).
    """

    def __init__(
        self,
        model_name_or_path: str = "microsoft/trocr-large-printed",
        device: str = "cpu",
        batch_size: int = 4,
        max_new_tokens: int = 512,
    ):
        try:
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            import torch
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required:\n"
                "  pip install transformers torch"
            ) from exc

        import torch
        self._torch = torch
        self.device = device
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens

        log.info("Loading TrOCR model '%s' on %s …", model_name_or_path, device)
        self._processor = TrOCRProcessor.from_pretrained(model_name_or_path)
        self._model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)
        self._model.to(device)
        self._model.eval()
        log.info("TrOCR model loaded — ready for inference")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crop_region(self, page_image: Image.Image, region: Region) -> Image.Image:
        """Crop a region from the page image, with a small safety margin."""
        x1, y1, x2, y2 = region.bbox
        w, h = page_image.size
        # Add 4px padding, clamped to image bounds
        x1 = max(0, x1 - 4)
        y1 = max(0, y1 - 4)
        x2 = min(w, x2 + 4)
        y2 = min(h, y2 + 4)
        cropped = page_image.crop((x1, y1, x2, y2)).convert("RGB")
        return cropped

    def _run_inference(self, images: List[Image.Image]) -> List[str]:
        """Run TrOCR inference on a batch of PIL images."""
        pixel_values = self._processor(
            images=images,
            return_tensors="pt",
        ).pixel_values.to(self.device)

        with self._torch.no_grad():
            generated_ids = self._model.generate(
                pixel_values,
                max_new_tokens=self.max_new_tokens,
            )

        texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ocr_region(self, page_image: Image.Image, region: Region) -> str:
        """Extract text from a single layout region."""
        # If Nougat already populated text, skip OCR
        if region.text:
            return region.text

        crop = self._crop_region(page_image, region)
        if crop.width < 10 or crop.height < 10:
            log.debug("Skipping tiny region (too small for OCR)")
            return ""

        texts = self._run_inference([crop])
        return texts[0] if texts else ""

    def ocr_page(
        self,
        page_image: Image.Image,
        layout: PageLayout,
        keep_types: Optional[List[str]] = None,
    ) -> PageLayout:
        """
        Run OCR on all text regions of a PageLayout.
        Populates region.text in-place and returns the modified layout.

        Parameters
        ----------
        page_image  : The full-page PIL Image.
        layout      : PageLayout with detected regions.
        keep_types  : Region types to process (others are skipped).

        Returns
        -------
        The updated PageLayout with region.text populated.
        """
        text_regions = layout.text_regions(keep_types)
        log.info(
            "Page %d: OCR on %d region(s)",
            layout.page_number, len(text_regions),
        )

        # Process in batches for efficiency
        for i in range(0, len(text_regions), self.batch_size):
            batch_regions = text_regions[i : i + self.batch_size]

            # Filter regions that already have text (Nougat pass-through)
            to_infer = [r for r in batch_regions if not r.text]
            pre_filled = [r for r in batch_regions if r.text]

            if pre_filled:
                log.debug(
                    "  %d region(s) already have text (Nougat pass-through)",
                    len(pre_filled),
                )

            if to_infer:
                crops = [self._crop_region(page_image, r) for r in to_infer]
                # Skip degenerate crops
                valid_pairs = [(r, c) for r, c in zip(to_infer, crops)
                               if c.width >= 10 and c.height >= 10]
                if valid_pairs:
                    valid_regions, valid_crops = zip(*valid_pairs)
                    texts = self._run_inference(list(valid_crops))
                    for region, text in zip(valid_regions, texts):
                        region.text = text
                        log.debug(
                            "  Region [%s] → %d chars: %s…",
                            region.region_type, len(text),
                            text[:60].replace("\n", " "),
                        )

        return layout

    def ocr_all_pages(
        self,
        page_images: List[Image.Image],
        layouts: List[PageLayout],
        keep_types: Optional[List[str]] = None,
    ) -> List[PageLayout]:
        """
        Run OCR across all pages.

        Parameters
        ----------
        page_images : List of full-page PIL Images.
        layouts     : Corresponding list of PageLayout objects.
        keep_types  : Region types to include.

        Returns
        -------
        Updated list of PageLayout objects with text populated.
        """
        if len(page_images) != len(layouts):
            raise ValueError(
                f"Mismatch: {len(page_images)} images vs {len(layouts)} layouts"
            )

        for img, layout in zip(page_images, layouts):
            self.ocr_page(img, layout, keep_types=keep_types)
            log.info(
                "OCR complete — page %d/%d",
                layout.page_number, len(layouts),
            )

        return layouts


# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, json
    from pipeline.pdf_to_images import pdf_to_images
    from pipeline.layout_detector import LayoutDetector

    parser = argparse.ArgumentParser(description="Run OCR on a PDF (Stage 2).")
    parser.add_argument("--pdf",    required=True, help="Input PDF path")
    parser.add_argument("--model",  default="microsoft/trocr-large-printed",
                        help="TrOCR model ID or local path")
    parser.add_argument("--engine", default="doclayout", choices=["doclayout","nougat","skip"])
    parser.add_argument("--dpi",    type=int, default=300)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out",    default="ocr_output.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    images = pdf_to_images(args.pdf, dpi=args.dpi)
    detector = LayoutDetector(engine=args.engine, device=args.device)
    layouts = detector.detect_all(images)

    ocr = TrOCREngine(model_name_or_path=args.model, device=args.device)
    layouts = ocr.ocr_all_pages(images, layouts)

    # Save combined output
    output = [l.to_dict() for l in layouts]
    # Add text fields
    for page_data, layout in zip(output, layouts):
        for region_data, region in zip(page_data["regions"], layout.regions):
            region_data["text"] = region.text or ""

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"OCR complete — {len(layouts)} pages → {args.out}")
