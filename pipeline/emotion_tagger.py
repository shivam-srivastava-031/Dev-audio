"""
pipeline/emotion_tagger.py
===========================
Stage 6 — Emotion Tagging + Speaker Segmentation

Enriches the structured book text with:
  1. Sentence-level emotion tags (joy, anger, sadness, fear, disgust,
     surprise, neutral) using j-hartmann/emotion-english-distilroberta-base.
  2. Speaker segmentation: identifies dialogue turns and assigns consistent
     voice IDs to recurring character names.

Output: an enriched version of the BookStructure paragraphs as a list of
"tagged sentences" per paragraph, ready for TTS in Stage 7.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TaggedSentence:
    """A sentence with emotion and speaker metadata."""
    text: str
    emotion: str = "neutral"
    emotion_confidence: float = 1.0
    is_dialogue: bool = False
    speaker_id: str = "narrator_default"

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "emotion": self.emotion,
            "emotion_confidence": round(self.emotion_confidence, 4),
            "is_dialogue": self.is_dialogue,
            "speaker_id": self.speaker_id,
        }


@dataclass
class TaggedParagraph:
    sentences: List[TaggedSentence] = field(default_factory=list)
    paragraph_emotion: str = "neutral"   # dominant emotion for the whole paragraph

    def to_dict(self) -> dict:
        return {
            "paragraph_emotion": self.paragraph_emotion,
            "sentences": [s.to_dict() for s in self.sentences],
        }

    def plain_text(self) -> str:
        return " ".join(s.text for s in self.sentences)


# ---------------------------------------------------------------------------
# Sentence splitter (regex, no external NLP lib required)
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"\'«])|(?<=\.\")\s+|(?<=\!\")\ s+|(?<=\?\")\s+'
)


def split_sentences(text: str) -> List[str]:
    """Split a paragraph into sentences using punctuation heuristics."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    # Merge very short fragments (< 5 chars) with the next sentence
    merged: List[str] = []
    for part in parts:
        if merged and len(part) < 5:
            merged[-1] = merged[-1] + " " + part
        else:
            merged.append(part.strip())
    return [s for s in merged if s]


# ---------------------------------------------------------------------------
# Emotion classifier
# ---------------------------------------------------------------------------

class EmotionClassifier:
    """
    Sentence-level emotion classification using RoBERTa.

    Model: j-hartmann/emotion-english-distilroberta-base
    Labels: anger, disgust, fear, joy, neutral, sadness, surprise
    """

    _LABEL_ALIASES = {
        "anger": "anger",
        "disgust": "disgust",
        "fear": "fear",
        "joy": "joy",
        "neutral": "neutral",
        "sadness": "sadness",
        "surprise": "surprise",
    }

    def __init__(
        self,
        model_id: str = "j-hartmann/emotion-english-distilroberta-base",
        device: str = "cpu",
        batch_size: int = 16,
        min_confidence: float = 0.6,
    ):
        try:
            from transformers import pipeline as hf_pipeline
        except ImportError as exc:
            raise ImportError("pip install transformers") from exc

        log.info("Loading emotion model '%s' on %s …", model_id, device)
        self._pipe = hf_pipeline(
            "text-classification",
            model=model_id,
            device=0 if device == "cuda" else -1,
            top_k=1,
        )
        self.batch_size = batch_size
        self.min_confidence = min_confidence
        log.info("Emotion model ready")

    def classify_batch(self, sentences: List[str]) -> List[Tuple[str, float]]:
        """
        Classify a list of sentences.

        Returns
        -------
        List of (emotion_label, confidence) tuples.
        """
        if not sentences:
            return []

        # Truncate very long sentences to avoid OOM
        truncated = [s[:512] for s in sentences]
        results = self._pipe(truncated, batch_size=self.batch_size)

        output: List[Tuple[str, float]] = []
        for result in results:
            # pipeline with top_k=1 returns [[{label, score}]]
            if isinstance(result, list):
                item = result[0]
            else:
                item = result
            label = self._LABEL_ALIASES.get(item["label"].lower(), "neutral")
            score = float(item["score"])
            if score < self.min_confidence:
                label = "neutral"
            output.append((label, score))
        return output


# ---------------------------------------------------------------------------
# Speaker / dialogue detector
# ---------------------------------------------------------------------------

# Dialogue verbs used to attribute quotes
_DIALOGUE_VERBS = {
    "said", "says", "asked", "answered", "replied", "whispered", "shouted",
    "yelled", "cried", "called", "muttered", "murmured", "exclaimed",
    "remarked", "responded", "growled", "hissed", "snapped", "breathed",
}

# Match quoted dialogue: "..." or '...'
_QUOTE_RE = re.compile(r'[""«](.+?)[""»]|\'(.+?)\'', re.DOTALL)


def detect_dialogue(sentence: str) -> Tuple[bool, Optional[str]]:
    """
    Detect if a sentence contains dialogue.

    Returns
    -------
    (is_dialogue, speaker_name_or_None)
    """
    has_quote = bool(_QUOTE_RE.search(sentence))
    if not has_quote:
        return False, None

    # Try to extract speaker name from attribution pattern
    # e.g. '"Get out!" she screamed.' or '"Hello," said John.'
    attribution_match = re.search(
        r'[""»,\.!?]\s+([A-Z][a-z]+)\s+(?:' + "|".join(_DIALOGUE_VERBS) + r')',
        sentence,
    )
    if attribution_match:
        return True, attribution_match.group(1)

    return True, None


class SpeakerTracker:
    """
    Assigns consistent voice IDs to characters detected in dialogue.

    Characters are assigned IDs like "character_1", "character_2", etc.
    The narrator always gets "narrator_default".
    """

    def __init__(self, narrator_voice: str = "narrator_default"):
        self.narrator_voice = narrator_voice
        self._char_voices: Dict[str, str] = {}
        self._char_counter = 0

    def get_voice_id(self, speaker_name: Optional[str], is_dialogue: bool) -> str:
        if not is_dialogue:
            return self.narrator_voice
        if speaker_name is None:
            return "character_unknown"
        if speaker_name not in self._char_voices:
            self._char_counter += 1
            self._char_voices[speaker_name] = f"character_{self._char_counter}"
            log.debug("Assigned voice ID '%s' to character '%s'",
                      self._char_voices[speaker_name], speaker_name)
        return self._char_voices[speaker_name]

    @property
    def character_map(self) -> Dict[str, str]:
        return dict(self._char_voices)


# ---------------------------------------------------------------------------
# Main emotion tagger
# ---------------------------------------------------------------------------

class EmotionTagger:
    """
    Full emotion tagging + speaker segmentation for a BookStructure.

    Parameters
    ----------
    model_id        : HuggingFace model ID for emotion classification.
    device          : "cpu" | "cuda" | "mps"
    batch_size      : Sentences per inference batch.
    min_confidence  : Minimum confidence to assign a non-neutral emotion.
    speaker_detection: Enable dialogue speaker tracking.
    narrator_voice  : Default narrator voice ID.
    """

    def __init__(
        self,
        model_id: str = "j-hartmann/emotion-english-distilroberta-base",
        device: str = "cpu",
        batch_size: int = 16,
        min_confidence: float = 0.6,
        speaker_detection: bool = True,
        narrator_voice: str = "narrator_default",
    ):
        self._classifier = EmotionClassifier(
            model_id=model_id,
            device=device,
            batch_size=batch_size,
            min_confidence=min_confidence,
        )
        self.speaker_detection = speaker_detection
        self._tracker = SpeakerTracker(narrator_voice=narrator_voice)

    def tag_paragraph(self, paragraph_text: str) -> TaggedParagraph:
        """Tag all sentences in a single paragraph."""
        sentences = split_sentences(paragraph_text)
        if not sentences:
            return TaggedParagraph()

        # Classify emotions in batch
        emotion_results = self._classifier.classify_batch(sentences)

        tagged_sentences: List[TaggedSentence] = []
        for sent, (emotion, conf) in zip(sentences, emotion_results):
            is_dialogue, speaker_name = (
                detect_dialogue(sent) if self.speaker_detection else (False, None)
            )
            voice_id = self._tracker.get_voice_id(speaker_name, is_dialogue)

            tagged_sentences.append(TaggedSentence(
                text=sent,
                emotion=emotion,
                emotion_confidence=conf,
                is_dialogue=is_dialogue,
                speaker_id=voice_id,
            ))

        # Compute dominant paragraph emotion (most frequent non-neutral)
        emotion_counts: Dict[str, int] = {}
        for ts in tagged_sentences:
            emotion_counts[ts.emotion] = emotion_counts.get(ts.emotion, 0) + 1
        dominant = max(emotion_counts, key=emotion_counts.get)

        return TaggedParagraph(sentences=tagged_sentences, paragraph_emotion=dominant)

    def tag_book(self, book_structure) -> List[Dict]:
        """
        Tag an entire BookStructure (from Stage 5).

        Parameters
        ----------
        book_structure : BookStructure instance from semantic_structurer.

        Returns
        -------
        List of dicts, one per chapter, containing tagged paragraph data.
        Also attaches character map metadata.
        """
        tagged_book = []

        for chapter in book_structure.chapters:
            tagged_chapter = {
                "chapter_number": chapter.number,
                "chapter_title": chapter.title,
                "sections": [],
            }
            for section in chapter.sections:
                tagged_section = {
                    "heading": section.heading,
                    "paragraphs": [],
                }
                for para in section.paragraphs:
                    tagged_para = self.tag_paragraph(para.text)
                    tagged_section["paragraphs"].append(tagged_para.to_dict())
                tagged_chapter["sections"].append(tagged_section)
            tagged_book.append(tagged_chapter)
            log.info(
                "Tagged chapter %d: %s",
                chapter.number, chapter.title,
            )

        return tagged_book

    def save_tagged_book(
        self,
        tagged_book: List[Dict],
        output_path: str | os.PathLike,
        book_title: str = "Unknown Book",
    ) -> None:
        """Save the tagged book to JSON."""
        output = {
            "book_title": book_title,
            "character_voice_map": self._tracker.character_map,
            "chapters": tagged_book,
        }
        output_path = os.fspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        log.info("Tagged book saved → %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    from pipeline.semantic_structurer import BookStructure

    parser = argparse.ArgumentParser(description="Emotion tagging (Stage 6)")
    parser.add_argument("--input",  required=True, help="Structured book JSON (Stage 5 output)")
    parser.add_argument("--output", required=True, help="Tagged book JSON output path")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    book = BookStructure.load(args.input)
    tagger = EmotionTagger(device=args.device, batch_size=args.batch_size)
    tagged = tagger.tag_book(book)
    tagger.save_tagged_book(tagged, args.output, book_title=book.book_title)

    print(f"Emotion tagging complete → {args.output}")
