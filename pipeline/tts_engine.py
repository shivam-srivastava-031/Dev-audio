"""
pipeline/tts_engine.py
=======================
Stage 7 — Text-to-Speech Generation

Converts tagged sentence data (from Stage 6) into WAV audio files,
applying emotion-driven prosody adjustments and per-character voice
profiles.

Supported engines:
  - "xtts"        → Coqui XTTS-v2 (multi-speaker, voice cloning)
  - "fish_speech"  → Fish Speech V1.5 via local HTTP server
  - "elevenlabs"  → ElevenLabs API (highest quality, paid)
  - "gtts"         → gTTS (Google TTS, no GPU, free, limited quality — fallback)
"""

import logging
import os
import re
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: split text into TTS-safe chunks (≤ max_words words)
# ---------------------------------------------------------------------------

def chunk_text(text: str, max_words: int = 300) -> List[str]:
    """Split text into chunks of at most max_words words at sentence boundaries."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks: List[str] = []
    current_words: List[str] = []

    for sent in sentences:
        words = sent.split()
        if len(current_words) + len(words) > max_words and current_words:
            chunks.append(" ".join(current_words))
            current_words = words
        else:
            current_words.extend(words)

    if current_words:
        chunks.append(" ".join(current_words))

    return [c for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------

class _XTTSBackend:
    """
    Coqui XTTS-v2 backend using the TTS Python library.
    Supports voice cloning via a reference speaker WAV.
    """

    def __init__(self, model_name: str, device: str = "cpu"):
        try:
            from TTS.api import TTS
        except ImportError as exc:
            raise ImportError(
                "Coqui TTS not installed.\n  pip install TTS"
            ) from exc

        log.info("Loading XTTS model '%s' on %s …", model_name, device)
        self._tts = TTS(model_name=model_name)
        # Move to device
        if device == "cuda":
            self._tts = self._tts.to("cuda")
        log.info("XTTS model ready")

    def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ) -> None:
        kwargs: dict = {
            "text": text,
            "file_path": output_path,
            "language": language,
            "speed": speed,
        }
        if speaker_wav and Path(speaker_wav).exists():
            kwargs["speaker_wav"] = speaker_wav
        else:
            # XTTS-v2 built-in speaker (first in list)
            kwargs["speaker"] = "Claribel Dervla"
        self._tts.tts_to_file(**kwargs)


class _FishSpeechBackend:
    """Fish Speech V1.5 via local HTTP REST API."""

    def __init__(self, server_url: str = "http://localhost:8080"):
        try:
            import requests
            self._requests = requests
        except ImportError as exc:
            raise ImportError("pip install requests") from exc
        self._url = server_url.rstrip("/")
        log.info("Fish Speech backend: %s", self._url)

    def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ) -> None:
        payload = {"text": text, "language": language, "speed": speed}
        if speaker_wav:
            payload["reference_audio"] = speaker_wav

        response = self._requests.post(
            f"{self._url}/v1/tts",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)


class _ElevenLabsBackend:
    """ElevenLabs API backend."""

    def __init__(self, api_key: str, default_voice_id: str = "21m00Tcm4TlvDq8ikWAM"):
        try:
            import requests
            self._requests = requests
        except ImportError as exc:
            raise ImportError("pip install requests") from exc

        self._api_key = api_key
        self._default_voice_id = default_voice_id
        self._base_url = "https://api.elevenlabs.io/v1"
        log.info("ElevenLabs backend configured")

    def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ) -> None:
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": "eleven_multilingual_v2",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.75,
                "speaking_rate": speed,
            },
        }
        response = self._requests.post(
            f"{self._base_url}/text-to-speech/{self._default_voice_id}",
            headers=headers,
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(response.content)


class _GTTSBackend:
    """Google TTS fallback (no GPU, free, limited prosody)."""

    def __init__(self):
        try:
            from gtts import gTTS
            self._gTTS = gTTS
        except ImportError as exc:
            raise ImportError("pip install gTTS") from exc
        log.info("gTTS fallback backend loaded")

    def synthesize(
        self,
        text: str,
        output_path: str,
        speaker_wav: Optional[str] = None,
        language: str = "en",
        speed: float = 1.0,
    ) -> None:
        tts = self._gTTS(text=text, lang=language[:2], slow=(speed < 0.9))
        # gTTS saves as MP3; convert to WAV via pydub
        mp3_path = output_path.replace(".wav", "_tmp.mp3")
        tts.save(mp3_path)
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(output_path, format="wav")
            os.remove(mp3_path)
        except Exception:
            # If pydub unavailable, keep the MP3
            os.rename(mp3_path, output_path.replace(".wav", ".mp3"))


# ---------------------------------------------------------------------------
# Emotion → prosody mapping
# ---------------------------------------------------------------------------

DEFAULT_EMOTION_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "joy":     {"speed": 1.1,  "pitch_shift": 1.5},
    "anger":   {"speed": 1.2,  "pitch_shift": -1.0},
    "sadness": {"speed": 0.9,  "pitch_shift": -1.5},
    "fear":    {"speed": 1.15, "pitch_shift": 0.5},
    "disgust": {"speed": 1.05, "pitch_shift": -0.5},
    "surprise":{"speed": 1.1,  "pitch_shift": 1.0},
    "neutral": {"speed": 1.0,  "pitch_shift": 0.0},
}


# ---------------------------------------------------------------------------
# TTSEngine (main class)
# ---------------------------------------------------------------------------

class TTSEngine:
    """
    High-level TTS engine.

    Parameters
    ----------
    engine              : "xtts" | "fish_speech" | "elevenlabs" | "gtts"
    model_or_url        : XTTS model name, Fish Speech server URL, or ElevenLabs voice ID.
    device              : "cpu" | "cuda"
    language            : Default synthesis language (ISO 639-1 code).
    max_chunk_words     : Max words per synthesis chunk.
    voice_profiles      : Dict mapping voice_id → {"speaker_wav", "speed", "pitch_shift"}.
    emotion_adjustments : Dict mapping emotion → {"speed", "pitch_shift"}.
    elevenlabs_api_key  : API key (only for elevenlabs engine).
    """

    def __init__(
        self,
        engine: str = "xtts",
        model_or_url: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        device: str = "cpu",
        language: str = "en",
        max_chunk_words: int = 300,
        voice_profiles: Optional[Dict] = None,
        emotion_adjustments: Optional[Dict] = None,
        elevenlabs_api_key: str = "",
    ):
        self.language = language
        self.max_chunk_words = max_chunk_words
        self.voice_profiles = voice_profiles or {
            "narrator_default": {"speaker_wav": None, "speed": 1.0}
        }
        self.emotion_adjustments = emotion_adjustments or DEFAULT_EMOTION_ADJUSTMENTS

        engine = engine.lower()
        if engine == "xtts":
            self._backend = _XTTSBackend(model_name=model_or_url, device=device)
        elif engine == "fish_speech":
            self._backend = _FishSpeechBackend(server_url=model_or_url)
        elif engine == "elevenlabs":
            self._backend = _ElevenLabsBackend(api_key=elevenlabs_api_key)
        elif engine == "gtts":
            self._backend = _GTTSBackend()
        else:
            log.warning("Unknown TTS engine '%s' — falling back to gTTS", engine)
            self._backend = _GTTSBackend()

    # ------------------------------------------------------------------
    # Prosody helpers
    # ------------------------------------------------------------------

    def _get_voice_params(self, voice_id: str, emotion: str) -> Dict:
        """Merge voice profile settings with emotion adjustments."""
        profile = self.voice_profiles.get(voice_id, {})
        adj = self.emotion_adjustments.get(emotion, self.emotion_adjustments["neutral"])

        base_speed = profile.get("speed", 1.0)
        emotion_speed = adj.get("speed", 1.0)
        final_speed = round(base_speed * emotion_speed, 3)

        return {
            "speaker_wav": profile.get("speaker_wav"),
            "speed": final_speed,
        }

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesize_sentence(
        self,
        sentence: dict,  # TaggedSentence.to_dict()
        output_path: str,
    ) -> Optional[str]:
        """
        Synthesize a single tagged sentence to a WAV file.

        Returns the output_path on success, None on failure.
        """
        text = sentence.get("text", "").strip()
        if not text:
            return None

        emotion = sentence.get("emotion", "neutral")
        voice_id = sentence.get("speaker_id", "narrator_default")
        params = self._get_voice_params(voice_id, emotion)

        try:
            self._backend.synthesize(
                text=text,
                output_path=output_path,
                speaker_wav=params.get("speaker_wav"),
                language=self.language,
                speed=params["speed"],
            )
            return output_path
        except Exception as exc:
            log.error("TTS failed for sentence '%s…': %s", text[:40], exc)
            return None

    def synthesize_chapter(
        self,
        chapter_data: dict,
        output_dir: str | os.PathLike,
        chapter_index: int = 1,
    ) -> List[str]:
        """
        Synthesize all tagged sentences in a chapter.

        Parameters
        ----------
        chapter_data  : Chapter dict from emotion_tagger output.
        output_dir    : Directory to save per-sentence WAV files.
        chapter_index : Chapter number for file naming.

        Returns
        -------
        List of paths to generated WAV files (in order).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        wav_paths: List[str] = []
        sentence_counter = 0

        for section in chapter_data.get("sections", []):
            for para in section.get("paragraphs", []):
                for sent in para.get("sentences", []):
                    sentence_counter += 1
                    wav_name = f"ch{chapter_index:03d}_s{sentence_counter:05d}.wav"
                    wav_path = str(output_dir / wav_name)

                    result = self.synthesize_sentence(sent, wav_path)
                    if result:
                        wav_paths.append(result)
                        log.debug("  [%d] %s", sentence_counter, sent["text"][:60])
                    else:
                        log.warning("  Sentence %d skipped (synthesis failed)", sentence_counter)

        log.info(
            "Chapter %d: %d sentences → %d WAV files",
            chapter_index, sentence_counter, len(wav_paths),
        )
        return wav_paths

    def synthesize_tagged_book(
        self,
        tagged_book_path: str | os.PathLike,
        output_dir: str | os.PathLike,
    ) -> Dict[int, List[str]]:
        """
        Synthesize an entire tagged book JSON (from Stage 6).

        Parameters
        ----------
        tagged_book_path : Path to tagged_book.json from emotion_tagger.
        output_dir       : Root output directory.

        Returns
        -------
        Dict mapping chapter_number → list of WAV file paths.
        """
        import json
        output_dir = Path(output_dir)

        with open(tagged_book_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chapters = data.get("chapters", [])
        result: Dict[int, List[str]] = {}

        for chapter in chapters:
            ch_num = chapter.get("chapter_number", 1)
            ch_dir = output_dir / "chunks" / f"chapter_{ch_num:03d}"
            wav_paths = self.synthesize_chapter(chapter, ch_dir, chapter_index=ch_num)
            result[ch_num] = wav_paths

        log.info(
            "TTS complete: %d chapters, %d total WAV files",
            len(result),
            sum(len(v) for v in result.values()),
        )
        return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TTS synthesis (Stage 7)")
    parser.add_argument("--input",   required=True, help="Tagged book JSON (Stage 6 output)")
    parser.add_argument("--output",  required=True, help="Output directory")
    parser.add_argument("--engine",  default="xtts", choices=["xtts","fish_speech","elevenlabs","gtts"])
    parser.add_argument("--device",  default="cpu")
    parser.add_argument("--language", default="en")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    engine = TTSEngine(engine=args.engine, device=args.device, language=args.language)
    result = engine.synthesize_tagged_book(args.input, args.output)
    total_wavs = sum(len(v) for v in result.values())
    print(f"TTS complete — {len(result)} chapters, {total_wavs} WAV files → {args.output}")
