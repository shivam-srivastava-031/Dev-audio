"""
pipeline/semantic_structurer.py
================================
Stage 5 — Semantic Structuring

Converts flat corrected text into a structured JSON representation with:
  - Book title
  - Chapters (with numbers and titles)
  - Sections (optional sub-headings)
  - Paragraphs

Strategy: "hybrid" (default)
  1. Rule-based: Detect chapters via configurable regex patterns.
  2. LLM-assisted: For ambiguous heading detection (optional).

The output JSON is the canonical intermediate format used by Stage 6
(emotion tagging) and Stage 7 (TTS).
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field, asdict
from typing import List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Paragraph:
    text: str
    index: int                   # position within the section


@dataclass
class Section:
    heading: Optional[str]
    paragraphs: List[Paragraph] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "heading": self.heading,
            "paragraphs": [p.text for p in self.paragraphs],
        }


@dataclass
class Chapter:
    number: int
    title: str
    sections: List[Section] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "number": self.number,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
        }


@dataclass
class BookStructure:
    book_title: str
    chapters: List[Chapter] = field(default_factory=list)
    language: str = "en"

    def to_dict(self) -> dict:
        return {
            "book_title": self.book_title,
            "language": self.language,
            "chapters": [c.to_dict() for c in self.chapters],
        }

    def save(self, path: str | os.PathLike) -> None:
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        log.info("Book structure saved → %s", path)

    @classmethod
    def load(cls, path: str | os.PathLike) -> "BookStructure":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        book = cls(book_title=data.get("book_title", "Unknown"), language=data.get("language", "en"))
        for ch_data in data.get("chapters", []):
            chapter = Chapter(number=ch_data["number"], title=ch_data["title"])
            for sec_data in ch_data.get("sections", []):
                section = Section(heading=sec_data.get("heading"))
                for i, para_text in enumerate(sec_data.get("paragraphs", [])):
                    section.paragraphs.append(Paragraph(text=para_text, index=i))
                chapter.sections.append(section)
            book.chapters.append(chapter)
        return book

    def plain_text_chapters(self) -> List[str]:
        """Return a list of plain text strings, one per chapter."""
        texts: List[str] = []
        for chapter in self.chapters:
            parts = [chapter.title]
            for section in chapter.sections:
                if section.heading:
                    parts.append(section.heading)
                for para in section.paragraphs:
                    parts.append(para.text)
            texts.append("\n\n".join(parts))
        return texts


# ---------------------------------------------------------------------------
# Default chapter heading patterns
# ---------------------------------------------------------------------------

DEFAULT_CHAPTER_PATTERNS = [
    r"^(Chapter|CHAPTER|Chapitre|Kapitel)\s+(\d+|[IVXLCDM]+)[\s:—\-]*(.*)$",
    r"^(Part|PART|Teil|Partie)\s+(\d+|[IVXLCDM]+)[\s:—\-]*(.*)$",
    r"^(Prologue|PROLOGUE|Epilogue|EPILOGUE|Preface|PREFACE|Introduction|INTRODUCTION)$",
    r"^\d+\.\s+.{3,60}$",     # e.g. "1. The Dark Forest"
]

# Section heading: shorter all-caps or title-case lines that are NOT chapters
_SECTION_PATTERN = re.compile(r"^[A-Z][A-Za-z\s]{2,50}$")


# ---------------------------------------------------------------------------
# Rule-based structurer
# ---------------------------------------------------------------------------

class RuleBasedStructurer:
    """
    Parses flat text into BookStructure using configurable regex patterns.

    Parameters
    ----------
    chapter_patterns : List of regex strings that match chapter heading lines.
    title_from_first  : If True, use the first non-empty line as the book title.
    """

    def __init__(
        self,
        chapter_patterns: Optional[List[str]] = None,
        title_from_first: bool = True,
    ):
        patterns = chapter_patterns or DEFAULT_CHAPTER_PATTERNS
        self._chapter_re = [re.compile(p, re.MULTILINE) for p in patterns]
        self.title_from_first = title_from_first

    def _is_chapter_heading(self, line: str) -> bool:
        for pattern in self._chapter_re:
            if pattern.match(line.strip()):
                return True
        return False

    def _extract_chapter_title(self, line: str) -> str:
        """Normalise the chapter heading line into a clean title."""
        line = line.strip()
        # Remove leading numbering like "Chapter 1 — " or "1. "
        line = re.sub(r"^(Chapter|CHAPTER|Part|PART)\s+\d+\s*[:\-—]?\s*", "", line)
        line = re.sub(r"^\d+\.\s+", "", line)
        return line or "Chapter"

    def structure(self, text: str, book_title: str = "Unknown Book") -> BookStructure:
        """
        Parse flat text into a BookStructure.

        Parameters
        ----------
        text       : The full corrected book text.
        book_title : Book title (from metadata or filename).

        Returns
        -------
        BookStructure
        """
        lines = text.splitlines()
        book = BookStructure(book_title=book_title)

        # Attempt to infer title from first meaningful line
        if self.title_from_first:
            for line in lines:
                stripped = line.strip()
                if stripped and not self._is_chapter_heading(stripped):
                    book.book_title = stripped
                    log.info("Inferred book title: '%s'", book.book_title)
                    break

        current_chapter: Optional[Chapter] = None
        current_section: Optional[Section] = None
        current_paragraph_lines: List[str] = []
        chapter_count = 0

        def _flush_paragraph():
            nonlocal current_paragraph_lines
            text_block = " ".join(current_paragraph_lines).strip()
            if text_block and current_section is not None:
                idx = len(current_section.paragraphs)
                current_section.paragraphs.append(Paragraph(text=text_block, index=idx))
            current_paragraph_lines = []

        def _flush_section():
            nonlocal current_section
            _flush_paragraph()
            if current_section is not None and current_chapter is not None:
                current_chapter.sections.append(current_section)
            current_section = None

        def _flush_chapter():
            nonlocal current_chapter
            _flush_section()
            if current_chapter is not None:
                book.chapters.append(current_chapter)
            current_chapter = None

        for line in lines:
            stripped = line.strip()

            if not stripped:
                # Blank line = paragraph break
                _flush_paragraph()
                continue

            if self._is_chapter_heading(stripped):
                _flush_chapter()
                chapter_count += 1
                title = self._extract_chapter_title(stripped)
                current_chapter = Chapter(number=chapter_count, title=title or stripped)
                current_section = Section(heading=None)  # default unnamed section
                log.debug("Chapter %d: %s", chapter_count, title)
                continue

            # If no chapter yet, create a default one
            if current_chapter is None:
                chapter_count += 1
                current_chapter = Chapter(number=chapter_count, title="Opening")
                current_section = Section(heading=None)

            # Section heading detection (short, title-case lines between paragraphs)
            if (
                not current_paragraph_lines
                and len(stripped) < 80
                and _SECTION_PATTERN.match(stripped)
                and not stripped.endswith(".")
            ):
                _flush_section()
                current_section = Section(heading=stripped)
                continue

            current_paragraph_lines.append(stripped)

        # Flush remaining content
        _flush_chapter()

        if not book.chapters:
            # No chapter markers found — treat entire text as one chapter
            log.warning("No chapter headings found — treating entire text as Chapter 1")
            chapter = Chapter(number=1, title="Full Text")
            section = Section(heading=None)
            paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
            for i, para_text in enumerate(paragraphs):
                section.paragraphs.append(Paragraph(text=para_text, index=i))
            chapter.sections.append(section)
            book.chapters.append(chapter)

        log.info(
            "Structured book: %d chapters, %d total sections",
            len(book.chapters),
            sum(len(c.sections) for c in book.chapters),
        )
        return book


# ---------------------------------------------------------------------------
# Public facade
# ---------------------------------------------------------------------------

class SemanticStructurer:
    """
    Semantic structuring facade.

    Parameters
    ----------
    strategy        : "rules" | "hybrid" | "llm"
    chapter_patterns: Custom chapter heading regex patterns.
    llm_backend     : LLM backend for hybrid mode (optional).
    """

    def __init__(
        self,
        strategy: str = "hybrid",
        chapter_patterns: Optional[List[str]] = None,
        llm_backend=None,
    ):
        self.strategy = strategy.lower()
        self._rules = RuleBasedStructurer(chapter_patterns=chapter_patterns)
        self._llm = llm_backend  # optional LLMCorrector instance for hybrid mode

    def structure(self, text: str, book_title: str = "Unknown Book") -> BookStructure:
        """
        Parse text into a BookStructure.

        Parameters
        ----------
        text       : Full corrected book text.
        book_title : Book title hint.

        Returns
        -------
        BookStructure
        """
        book = self._rules.structure(text, book_title=book_title)

        # In hybrid mode, if very few chapters were found (≤1), ask LLM
        if self.strategy == "hybrid" and self._llm and len(book.chapters) <= 1 and len(text) > 5000:
            log.info("Hybrid mode: few chapters detected — attempting LLM chapter identification")
            book = self._llm_structure(text, book_title)

        return book

    def _llm_structure(self, text: str, book_title: str) -> BookStructure:
        """Ask LLM to identify chapter break points from the first portion of text."""
        sample = text[:4000]
        prompt = (
            f"The following is the beginning of a book titled '{book_title}'.\n"
            "List the chapter headings you can identify, one per line, in the format:\n"
            "CHAPTER: <heading text>\n\n"
            f"TEXT:\n{sample}\n\nCHAPTER LIST:"
        )
        try:
            response = self._llm._backend.generate(prompt)
            lines = [l.strip() for l in response.splitlines() if l.startswith("CHAPTER:")]
            patterns = [re.escape(l.replace("CHAPTER:", "").strip()) for l in lines]
            if patterns:
                log.info("LLM identified %d chapter patterns", len(patterns))
                self._rules = RuleBasedStructurer(chapter_patterns=patterns)
        except Exception as exc:
            log.warning("LLM chapter detection failed: %s", exc)

        return self._rules.structure(text, book_title=book_title)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Semantic structuring (Stage 5)")
    parser.add_argument("--input",  required=True, help="Corrected text file")
    parser.add_argument("--output", required=True, help="Output JSON file")
    parser.add_argument("--title",  default="Unknown Book", help="Book title")
    parser.add_argument("--strategy", default="hybrid", choices=["rules","hybrid"])
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    structurer = SemanticStructurer(strategy=args.strategy)
    book = structurer.structure(text, book_title=args.title)
    book.save(args.output)

    print(f"Structured: {len(book.chapters)} chapters → {args.output}")
