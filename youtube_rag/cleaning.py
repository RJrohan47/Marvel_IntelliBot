"""Stage 2: clean caption artifacts without losing timestamps."""

from __future__ import annotations

import html
import re

from youtube_rag.models import PreparedTranscript, TranscriptSegment


class TranscriptCleaner:
    def clean(self, transcript: PreparedTranscript) -> PreparedTranscript:
        cleaned_segments = []
        for segment in transcript.segments:
            text = html.unescape(segment.text)
            text = re.sub(r"\[(?:music|applause|laughter|noise|inaudible).*?\]", " ", text, flags=re.I)
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                cleaned_segments.append(TranscriptSegment(text, segment.start_seconds, segment.end_seconds))
        if not cleaned_segments:
            raise RuntimeError("The transcript is empty after cleaning.")
        return PreparedTranscript(transcript.video_id, transcript.language, cleaned_segments)
