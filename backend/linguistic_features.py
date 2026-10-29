"""
linguistic_features.py — Transcript-derived linguistic biomarker extraction.

Extracts lexical diversity, filler-word frequency, repetition, speaking rate,
and other language metrics from a transcribed voice recording.

All metrics are derived solely from the audio transcript and duration.
No demographic or clinical metadata is used.
"""

from __future__ import annotations

import io
import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger(__name__)

# Filler / hesitation words associated with cognitive load and word-finding difficulty.
FILLER_WORDS = {
    "um", "uh", "er", "ah", "hmm", "mm", "hm",
    "like", "you know", "i mean", "well", "so",
    "basically", "literally", "right",
}

# Minimum transcript word count before metrics are considered reliable.
MIN_RELIABLE_WORDS = 5


def transcribe_audio(audio_array, sample_rate: int) -> dict:
    """
    Run speech-to-text on a mono float32 NumPy audio array at `sample_rate` Hz.

    Returns a dict with keys:
      transcript (str)         — raw recognised text (empty string on failure)
      confidence_score (float) — approximate confidence [0.0 – 1.0]
      engine (str)             — engine name used
      status (str)             — 'ok' | 'unclear' | 'unavailable' | 'error'
      notes (str)              — human-readable explanation
    """
    try:
        import soundfile as sf
        import speech_recognition as sr_lib
    except ImportError as exc:
        return _failed_result(f"Required library not installed: {exc}")

    try:
        bio = io.BytesIO()
        sf.write(bio, audio_array, sample_rate, format="WAV")
        bio.seek(0)

        recogniser = sr_lib.Recognizer()
        recogniser.operation_timeout = 1.5
        with sr_lib.AudioFile(bio) as source:
            audio_data = recogniser.record(source)

        import socket
        old_timeout = socket.getdefaulttimeout()
        try:
            socket.setdefaulttimeout(3.0)
            text = recogniser.recognize_google(audio_data)
            return {
                "transcript": text,
                "confidence_score": 0.88,
                "engine": "Google Speech Recognition",
                "status": "ok",
                "notes": "Transcribed via online Google Speech API.",
            }
        except sr_lib.UnknownValueError:
            return {
                "transcript": "",
                "confidence_score": 0.15,
                "engine": "Google Speech Recognition",
                "status": "unclear",
                "notes": "Speech was too quiet, unclear, or absent.",
            }
        except Exception as net_err:
            logger.info("Speech recognition network fallback: %s", net_err)
            return {
                "transcript": "",
                "confidence_score": 0.0,
                "engine": "Speech Recognition (Offline)",
                "status": "unavailable",
                "notes": "Online transcription service unavailable. Linguistic metrics set to zero.",
            }
        finally:
            socket.setdefaulttimeout(old_timeout)

    except Exception as exc:
        logger.warning("Transcription failed: %s", exc)
        return _failed_result(str(exc))


def _failed_result(reason: str) -> dict:
    return {
        "transcript": "",
        "confidence_score": 0.0,
        "engine": "None",
        "status": "error",
        "notes": reason,
    }


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokenizer (strips punctuation)."""
    return re.findall(r"\b[a-z']+\b", text.lower())


def extract_linguistic_features(
    transcript: str,
    duration_seconds: float,
    confidence_score: float = 0.0,
    transcription_status: str = "ok",
) -> dict:
    """
    Compute linguistic / lexical diversity features from a transcript string.

    Parameters
    ----------
    transcript:           Raw recognised text.
    duration_seconds:     Total recording length in seconds.
    confidence_score:     ASR confidence (0 – 1).
    transcription_status: 'ok' | 'unclear' | 'unavailable' | 'error'

    Returns
    -------
    dict of float features — all NaN-free (zero-filled on transcription failure).
    """

    # ── Zero-fill when transcription was not successful ──────────────────────
    if transcription_status not in ("ok",) or not transcript.strip():
        return _zero_linguistic_features(confidence_score, transcription_status)

    tokens = _tokenize(transcript)
    total_words = len(tokens)

    if total_words == 0:
        return _zero_linguistic_features(confidence_score, transcription_status)

    # ── Vocabulary diversity ──────────────────────────────────────────────────
    unique_words = len(set(tokens))

    # Root Type-Token Ratio (RTTR) — less sensitive to text length than raw TTR.
    import math
    rttr = unique_words / math.sqrt(total_words) if total_words > 0 else 0.0

    # Simple TTR (retained for model compatibility, but RTTR is preferred).
    ttr = unique_words / total_words if total_words > 0 else 0.0

    # ── Filler words ─────────────────────────────────────────────────────────
    # Count single-word fillers.
    filler_count = sum(1 for tok in tokens if tok in FILLER_WORDS)
    filler_ratio = filler_count / total_words

    # ── Word repetition (immediate perseveration) ────────────────────────────
    # Count pairs of consecutive identical tokens.
    repeated_pairs = sum(
        1 for i in range(1, len(tokens)) if tokens[i] == tokens[i - 1]
    )
    repeated_ratio = repeated_pairs / max(total_words - 1, 1)

    # ── Speaking rate ─────────────────────────────────────────────────────────
    speaking_rate_wpm = (
        (total_words / duration_seconds) * 60.0
        if duration_seconds > 0
        else 0.0
    )

    # ── Utterance / sentence complexity ──────────────────────────────────────
    # Simple sentence split on punctuation.
    sentences = [s.strip() for s in re.split(r"[.!?]+", transcript) if s.strip()]
    num_sentences = max(len(sentences), 1)
    avg_words_per_sentence = total_words / num_sentences

    # ── Reliability flag ─────────────────────────────────────────────────────
    is_reliable = int(total_words >= MIN_RELIABLE_WORDS)

    return {
        # Word counts
        "ling_total_words": float(total_words),
        "ling_unique_words": float(unique_words),
        # Lexical diversity
        "ling_ttr": round(ttr, 4),
        "ling_rttr": round(rttr, 4),
        # Filler words
        "ling_filler_count": float(filler_count),
        "ling_filler_ratio": round(filler_ratio, 4),
        # Repetition
        "ling_repeated_pairs": float(repeated_pairs),
        "ling_repeated_ratio": round(repeated_ratio, 4),
        # Rate & complexity
        "ling_speaking_rate_wpm": round(speaking_rate_wpm, 2),
        "ling_avg_words_per_sentence": round(avg_words_per_sentence, 2),
        "ling_num_sentences": float(num_sentences),
        # Meta
        "ling_transcription_confidence": round(confidence_score, 4),
        "ling_is_reliable": float(is_reliable),
    }


def _zero_linguistic_features(confidence: float, status: str) -> dict:
    """Return all-zero linguistic feature dict when transcription failed."""
    return {
        "ling_total_words": 0.0,
        "ling_unique_words": 0.0,
        "ling_ttr": 0.0,
        "ling_rttr": 0.0,
        "ling_filler_count": 0.0,
        "ling_filler_ratio": 0.0,
        "ling_repeated_pairs": 0.0,
        "ling_repeated_ratio": 0.0,
        "ling_speaking_rate_wpm": 0.0,
        "ling_avg_words_per_sentence": 0.0,
        "ling_num_sentences": 0.0,
        "ling_transcription_confidence": round(confidence, 4),
        "ling_is_reliable": 0.0,
    }


LINGUISTIC_FEATURE_NAMES: list[str] = list(_zero_linguistic_features(0.0, "").keys())
