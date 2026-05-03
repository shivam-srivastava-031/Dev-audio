"""
pipeline/pdf_to_images.py
=========================
Stage 0 helper — Convert a PDF file into per-page PIL Images at a
target DPI for downstream layout detection and OCR.

Usage (standalone):
    python pipeline/pdf_to_images.py --pdf my_book.pdf --dpi 300 --out page_images/
"""

import argparse
import logging
import os
from pathlib import Path
from typing import List, Optional

from PIL import Image

log = logging.getLogger(__name__)


def pdf_to_images(
    pdf_path: str | os.PathLike,
    dpi: int = 300,
    output_dir: Optional[str | os.PathLike] = None,
    page_range: Optional[tuple[int, int]] = None,
) -> List[Image.Image]:
    """
    Convert every page of a PDF to a PIL Image at the given DPI.

    Parameters
    ----------
    pdf_path    : Path to the PDF file.
    dpi         : Render resolution (default: 300 DPI — recommended for OCR).
    output_dir  : If given, save pages as PNG files to this directory.
    page_range  : Tuple (start, end) for zero-indexed page selection. None → all pages.

    Returns
    -------
    List of PIL.Image.Image objects (one per page).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError(
            "PyMuPDF is required: pip install PyMuPDF"
        ) from exc

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    log.info("Opened PDF '%s' — %d pages, rendering at %d DPI", pdf_path.name, total_pages, dpi)

    start_page, end_page = 0, total_pages
    if page_range is not None:
        start_page = max(0, page_range[0])
        end_page = min(total_pages, page_range[1])

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Compute zoom matrix: PyMuPDF default is 72 DPI
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)

    images: List[Image.Image] = []
    for page_num in range(start_page, end_page):
        page = doc.load_page(page_num)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

        if output_dir is not None:
            save_path = output_dir / f"page_{page_num + 1:04d}.png"
            img.save(save_path, format="PNG")
            log.debug("Saved page %d → %s", page_num + 1, save_path)

    doc.close()
    log.info("Converted %d pages from '%s'", len(images), pdf_path.name)
    return images


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a PDF to per-page PNG images.")
    parser.add_argument("--pdf",  required=True, help="Path to input PDF file")
    parser.add_argument("--dpi",  type=int, default=300, help="Render DPI (default: 300)")
    parser.add_argument("--out",  required=True, help="Output directory for page images")
    parser.add_argument("--start", type=int, default=0, help="Start page index (0-indexed)")
    parser.add_argument("--end",   type=int, default=None, help="End page index (exclusive)")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = _parse_args()
    page_range = (args.start, args.end) if args.end is not None else None
    images = pdf_to_images(args.pdf, dpi=args.dpi, output_dir=args.out, page_range=page_range)
    print(f"Done — {len(images)} page images saved to: {args.out}")
