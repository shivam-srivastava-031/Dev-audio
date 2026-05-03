"""
pipeline/audio_builder.py
==========================
Stage 8 — Audio Stitching + Chapter Metadata

Combines all per-sentence WAV files into:
  1. Per-chapter MP3 files
  2. Full audiobook MP3 (all chapters concatenated)
  3. M4B audiobook with proper chapter navigation markers

Also applies:
  - EBU R128 loudness normalisation (-16 LUFS standard for audiobooks)
  - Configurable silence padding between sentences, paragraphs, chapters
  - ID3 / M4B metadata tags (title, author, narrator, cover art)

Dependencies:
  - pydub         (audio manipulation)
  - pyloudnorm    (EBU R128 loudness measurement + normalisation)
  - ffmpeg        (MP3 + M4B encoding, must be in PATH)
  - mutagen       (ID3 tag writing for MP3; M4B chapter tags)
"""

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _load_wav(path: str):
    """Load a WAV file as a pydub AudioSegment."""
    from pydub import AudioSegment
    return AudioSegment.from_wav(path)


def _silence(ms: int):
    """Create a silent AudioSegment of given duration."""
    from pydub import AudioSegment
    return AudioSegment.silent(duration=ms)


def _normalize_lufs(segment, target_lufs: float = -16.0, true_peak: float = -1.5):
    """
    Apply EBU R128 integrated loudness normalisation.

    Parameters
    ----------
    segment     : pydub AudioSegment.
    target_lufs : Target integrated loudness (dB LUFS), default -16.
    true_peak   : True peak ceiling (dBTP), default -1.5.

    Returns
    -------
    Normalised AudioSegment.
    """
    try:
        import numpy as np
        import pyloudnorm as pyln
        from pydub import AudioSegment

        # Convert to numpy float32 for pyloudnorm
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        samples = samples / (2 ** (segment.sample_width * 8 - 1))  # normalise to [-1, 1]
        if segment.channels == 2:
            samples = samples.reshape((-1, 2))

        meter = pyln.Meter(segment.frame_rate)
        loudness = meter.integrated_loudness(samples)

        if abs(loudness - target_lufs) < 0.5:
            log.debug("Already at target loudness (%.1f LUFS) — skipping normalisation", loudness)
            return segment

        gain_db = target_lufs - loudness
        log.debug("Normalising: %.1f LUFS → %.1f LUFS (gain %.1f dB)", loudness, target_lufs, gain_db)
        return segment.apply_gain(gain_db)

    except Exception as exc:
        log.warning("Loudness normalisation failed: %s — returning unchanged audio", exc)
        return segment


# ---------------------------------------------------------------------------
# Chapter stitching
# ---------------------------------------------------------------------------

class ChapterStitcher:
    """
    Stitch per-sentence WAV files for one chapter into a single AudioSegment.

    Parameters
    ----------
    pause_sentence_ms   : Silence between sentences (ms).
    pause_paragraph_ms  : Silence between paragraphs (ms).
    target_lufs         : Target loudness (EBU R128).
    true_peak           : True peak ceiling.
    """

    def __init__(
        self,
        pause_sentence_ms: int = 200,
        pause_paragraph_ms: int = 600,
        target_lufs: float = -16.0,
        true_peak: float = -1.5,
    ):
        self.pause_sentence_ms = pause_sentence_ms
        self.pause_paragraph_ms = pause_paragraph_ms
        self.target_lufs = target_lufs
        self.true_peak = true_peak

    def stitch(self, wav_paths: List[str]) -> object:
        """
        Stitch a list of WAV file paths into one AudioSegment.

        The file naming convention (from tts_engine.py) uses:
            ch{NNN}_s{NNNNN}.wav
        Files are loaded in sorted order (already in TTS sentence order).
        """
        from pydub import AudioSegment

        if not wav_paths:
            log.warning("No WAV files to stitch — returning empty segment")
            return AudioSegment.silent(duration=0)

        sorted_paths = sorted(wav_paths, key=lambda p: Path(p).name)
        combined = AudioSegment.empty()

        for i, wav_path in enumerate(sorted_paths):
            if not Path(wav_path).exists():
                log.warning("WAV not found, skipping: %s", wav_path)
                continue
            seg = _load_wav(wav_path)
            combined += seg

            # Add inter-sentence pause (shorter than paragraph pause)
            combined += _silence(self.pause_sentence_ms)

        log.info("Stitched %d segments — duration: %.1f s", len(sorted_paths),
                 len(combined) / 1000)
        return combined

    def stitch_chapter_from_tagged(
        self,
        chapter_data: dict,
        wav_dir: str | os.PathLike,
        chapter_index: int,
    ) -> object:
        """
        Stitch chapter audio using tagged paragraph structure to insert
        correct pauses between paragraphs.

        Parameters
        ----------
        chapter_data  : Chapter dict from tagged_book.json.
        wav_dir       : Directory containing the chapter's WAV chunks.
        chapter_index : Chapter number for file naming.
        """
        from pydub import AudioSegment

        wav_dir = Path(wav_dir)
        combined = AudioSegment.empty()
        sentence_counter = 0

        for section in chapter_data.get("sections", []):
            for para_idx, para in enumerate(section.get("paragraphs", [])):
                # Paragraph pause (except before the very first paragraph)
                if sentence_counter > 0:
                    combined += _silence(self.pause_paragraph_ms)

                for sent in para.get("sentences", []):
                    sentence_counter += 1
                    wav_name = f"ch{chapter_index:03d}_s{sentence_counter:05d}.wav"
                    wav_path = wav_dir / wav_name
                    if wav_path.exists():
                        combined += _load_wav(str(wav_path))
                        combined += _silence(self.pause_sentence_ms)
                    else:
                        log.warning("Missing chunk: %s", wav_path)

        log.info(
            "Chapter %d stitched — %d sentences, %.1f s",
            chapter_index, sentence_counter, len(combined) / 1000,
        )
        return combined


# ---------------------------------------------------------------------------
# AudioBuilder (main class)
# ---------------------------------------------------------------------------

class AudioBuilder:
    """
    Build the final audiobook from TTS chunks.

    Parameters
    ----------
    output_dir          : Root output directory.
    tagged_book_path    : Path to tagged_book.json (for pause logic).
    tts_output_dir      : Directory containing per-chapter WAV chunks.
    pause_sentence_ms   : Silence between sentences (ms).
    pause_paragraph_ms  : Silence between paragraphs (ms).
    pause_chapter_ms    : Silence between chapters (ms).
    target_lufs         : EBU R128 target loudness.
    true_peak           : True peak ceiling (dBTP).
    mp3_bitrate         : MP3 encoding bitrate (e.g. "128k").
    cover_art           : Path to cover art image (optional).
    book_title          : Book title for metadata tags.
    book_author         : Book author for metadata tags.
    """

    def __init__(
        self,
        output_dir: str | os.PathLike,
        tagged_book_path: str | os.PathLike,
        tts_output_dir: str | os.PathLike,
        pause_sentence_ms: int = 200,
        pause_paragraph_ms: int = 600,
        pause_chapter_ms: int = 2000,
        target_lufs: float = -16.0,
        true_peak: float = -1.5,
        mp3_bitrate: str = "128k",
        cover_art: Optional[str] = None,
        book_title: str = "Audiobook",
        book_author: str = "Unknown Author",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tagged_book_path = Path(tagged_book_path)
        self.tts_output_dir = Path(tts_output_dir)
        self.pause_chapter_ms = pause_chapter_ms
        self.mp3_bitrate = mp3_bitrate
        self.cover_art = cover_art
        self.book_title = book_title
        self.book_author = book_author
        self.target_lufs = target_lufs
        self.true_peak = true_peak

        self._stitcher = ChapterStitcher(
            pause_sentence_ms=pause_sentence_ms,
            pause_paragraph_ms=pause_paragraph_ms,
            target_lufs=target_lufs,
            true_peak=true_peak,
        )

    def _export_mp3(self, segment, output_path: Path, tags: dict) -> None:
        """Export an AudioSegment to MP3 with ID3 tags."""
        from pydub import AudioSegment
        segment.export(
            str(output_path),
            format="mp3",
            bitrate=self.mp3_bitrate,
            tags=tags,
        )
        log.info("Exported MP3: %s (%.1f s)", output_path.name, len(segment) / 1000)

    def _check_ffmpeg(self) -> bool:
        """Verify ffmpeg is available in PATH."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True, check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            log.warning("ffmpeg not found in PATH — M4B creation will be skipped")
            return False

    def build(self, wav_paths_by_chapter: Dict[int, List[str]]) -> dict:
        """
        Build the full audiobook from per-chapter WAV file lists.

        Parameters
        ----------
        wav_paths_by_chapter : Dict mapping chapter_number → list of WAV paths.

        Returns
        -------
        Dict with paths to all output files.
        """
        # Load tagged book for paragraph-aware stitching
        with open(self.tagged_book_path, "r", encoding="utf-8") as f:
            tagged_book = json.load(f)
        chapters_data = {c["chapter_number"]: c for c in tagged_book.get("chapters", [])}

        chapters_dir = self.output_dir / "chapters"
        chapters_dir.mkdir(exist_ok=True)

        chapter_mp3_paths: List[Path] = []
        chapter_durations_ms: List[int] = []
        full_book = None

        from pydub import AudioSegment

        for ch_num in sorted(wav_paths_by_chapter.keys()):
            wav_paths = wav_paths_by_chapter[ch_num]
            chapter_data = chapters_data.get(ch_num, {})
            ch_wav_dir = self.tts_output_dir / "chunks" / f"chapter_{ch_num:03d}"

            log.info("Building chapter %d …", ch_num)

            # Stitch with paragraph-aware pauses
            if chapter_data:
                chapter_audio = self._stitcher.stitch_chapter_from_tagged(
                    chapter_data, ch_wav_dir, ch_num
                )
            else:
                chapter_audio = self._stitcher.stitch(wav_paths)

            # Loudness normalisation
            chapter_audio = _normalize_lufs(chapter_audio, self.target_lufs, self.true_peak)
            chapter_durations_ms.append(len(chapter_audio))

            # Export per-chapter MP3
            ch_title = chapter_data.get("chapter_title", f"Chapter {ch_num}")
            ch_mp3 = chapters_dir / f"chapter_{ch_num:03d}.mp3"
            self._export_mp3(chapter_audio, ch_mp3, tags={
                "title": ch_title,
                "artist": self.book_author,
                "album": self.book_title,
                "track": str(ch_num),
            })
            chapter_mp3_paths.append(ch_mp3)

            # Append to full book
            if full_book is None:
                full_book = chapter_audio
            else:
                full_book += _silence(self.pause_chapter_ms)
                full_book += chapter_audio

        if full_book is None:
            log.error("No audio data produced — check TTS output")
            return {}

        # Export full MP3
        full_mp3 = self.output_dir / "audiobook.mp3"
        self._export_mp3(full_book, full_mp3, tags={
            "title": self.book_title,
            "artist": self.book_author,
            "album": self.book_title,
        })

        # Build M4B
        m4b_path = None
        if self._check_ffmpeg():
            m4b_path = self._build_m4b(
                chapter_mp3_paths,
                chapters_data,
                chapter_durations_ms,
            )

        # Save metadata JSON
        metadata = {
            "book_title": self.book_title,
            "book_author": self.book_author,
            "chapters": [
                {
                    "number": ch_num,
                    "title": chapters_data.get(ch_num, {}).get("chapter_title", f"Chapter {ch_num}"),
                    "duration_seconds": round(chapter_durations_ms[i] / 1000, 2),
                }
                for i, ch_num in enumerate(sorted(wav_paths_by_chapter.keys()))
            ],
        }
        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        output_files = {
            "audiobook_mp3": str(full_mp3),
            "chapter_mp3s": [str(p) for p in chapter_mp3_paths],
            "metadata_json": str(meta_path),
        }
        if m4b_path:
            output_files["audiobook_m4b"] = str(m4b_path)

        log.info("AudioBuilder complete — output: %s", self.output_dir)
        return output_files

    def _build_m4b(
        self,
        chapter_mp3_paths: List[Path],
        chapters_data: dict,
        chapter_durations_ms: List[int],
    ) -> Optional[Path]:
        """
        Create an M4B audiobook with chapter markers using ffmpeg.

        Steps:
        1. Concatenate chapter MP3s via ffmpeg concat demuxer
        2. Convert to AAC (M4B container)
        3. Inject chapter metadata via ffmpeg metadata file
        """
        m4b_path = self.output_dir / "audiobook.m4b"
        temp_dir = Path(tempfile.mkdtemp())

        try:
            # 1. Write concat list
            concat_list = temp_dir / "concat.txt"
            with open(concat_list, "w", encoding="utf-8") as f:
                for mp3 in chapter_mp3_paths:
                    f.write(f"file '{mp3.resolve()}'\n")

            # 2. Concatenate to single AAC
            concat_aac = temp_dir / "full_book.aac"
            subprocess.run([
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-c:a", "aac", "-b:a", "128k",
                str(concat_aac),
            ], check=True, capture_output=True)

            # 3. Build ffmpeg chapter metadata file
            chapter_meta = temp_dir / "chapters.txt"
            lines = [
                ";FFMETADATA1",
                f"title={self.book_title}",
                f"artist={self.book_author}",
                "",
            ]
            offset_ms = 0
            sorted_ch_nums = sorted(chapters_data.keys())
            for i, ch_num in enumerate(sorted_ch_nums):
                ch_title = chapters_data[ch_num].get("chapter_title", f"Chapter {ch_num}")
                start_ms = offset_ms
                dur = chapter_durations_ms[i] if i < len(chapter_durations_ms) else 0
                # Add chapter gap
                if i > 0:
                    start_ms += self.pause_chapter_ms
                end_ms = start_ms + dur
                offset_ms = end_ms

                lines += [
                    "[CHAPTER]",
                    "TIMEBASE=1/1000",
                    f"START={start_ms}",
                    f"END={end_ms}",
                    f"title={ch_title}",
                    "",
                ]
            with open(chapter_meta, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))

            # 4. Inject metadata + optional cover art
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-i", str(concat_aac),
                "-i", str(chapter_meta),
                "-map_metadata", "1",
                "-c:a", "copy",
            ]
            if self.cover_art and Path(self.cover_art).exists():
                ffmpeg_cmd += ["-i", self.cover_art, "-map", "0:a", "-map", "2:v",
                               "-disposition:v", "attached_pic"]
            ffmpeg_cmd.append(str(m4b_path))

            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            log.info("M4B created: %s", m4b_path)
            return m4b_path

        except subprocess.CalledProcessError as exc:
            log.error("ffmpeg M4B creation failed: %s\nstderr: %s", exc, exc.stderr.decode())
            return None
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Audio stitching + M4B creation (Stage 8)")
    parser.add_argument("--tagged-book",  required=True, help="tagged_book.json from Stage 6")
    parser.add_argument("--tts-dir",      required=True, help="TTS WAV output directory from Stage 7")
    parser.add_argument("--output",       required=True, help="Output directory")
    parser.add_argument("--title",        default="Audiobook")
    parser.add_argument("--author",       default="Unknown Author")
    parser.add_argument("--cover",        default=None)
    parser.add_argument("--lufs",         type=float, default=-16.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # Discover WAV chunks by chapter
    tts_dir = Path(args.tts_dir)
    wav_by_chapter: Dict[int, List[str]] = {}
    chunks_dir = tts_dir / "chunks"
    if chunks_dir.exists():
        for ch_dir in sorted(chunks_dir.iterdir()):
            if ch_dir.is_dir() and ch_dir.name.startswith("chapter_"):
                ch_num = int(ch_dir.name.split("_")[-1])
                wavs = sorted(str(p) for p in ch_dir.glob("*.wav"))
                if wavs:
                    wav_by_chapter[ch_num] = wavs

    if not wav_by_chapter:
        print("No WAV chunks found in", chunks_dir)
        raise SystemExit(1)

    builder = AudioBuilder(
        output_dir=args.output,
        tagged_book_path=args.tagged_book,
        tts_output_dir=args.tts_dir,
        target_lufs=args.lufs,
        book_title=args.title,
        book_author=args.author,
        cover_art=args.cover,
    )
    files = builder.build(wav_by_chapter)
    print("Audiobook built:")
    for key, val in files.items():
        print(f"  {key}: {val}")
