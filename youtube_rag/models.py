"""Data structures passed through the YouTube RAG pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class TranscriptSegment:
    text: str
    start_seconds: float
    end_seconds: float


@dataclass(frozen=True)
class PreparedTranscript:
    video_id: str
    language: str
    segments: list[TranscriptSegment]
