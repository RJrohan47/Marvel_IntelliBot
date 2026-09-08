"""Stage 1: load captions and ensure the transcript is in English."""

from __future__ import annotations

import re
from urllib.parse import parse_qs, urlparse

from youtube_transcript_api import NoTranscriptFound, TranscriptsDisabled, VideoUnavailable, YouTubeTranscriptApi

from youtube_rag.models import PreparedTranscript, TranscriptSegment


ENGLISH_CODES = ("en", "en-US", "en-GB")


def extract_video_id(video_or_url: str) -> str:
    value = video_or_url.strip()
    parsed = urlparse(value)
    host = parsed.netloc.lower().split(":")[0]
    if host in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if not video_id:
            match = re.match(r"/(?:shorts|embed|live)/([^/?#]+)", parsed.path)
            video_id = match.group(1) if match else ""
    elif not parsed.scheme and not parsed.netloc:
        video_id = value
    else:
        video_id = ""
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("Enter a valid YouTube video URL or 11-character video ID.")
    return video_id


class YouTubeTranscriptLoader:
    """Loads English captions, or asks YouTube for an English translation."""

    def load(self, video_id: str, preferred_languages: list[str] | None = None) -> PreparedTranscript:
        try:
            transcripts = YouTubeTranscriptApi().list(video_id)
            try:
                selected = transcripts.find_transcript(ENGLISH_CODES)
                fetched = selected.fetch()
            except NoTranscriptFound:
                selected = self._select_translatable(transcripts, preferred_languages)
                if selected is None:
                    raise RuntimeError("No English or translatable transcript was found for this video.")
                fetched = selected.translate("en").fetch()
        except TranscriptsDisabled as exc:
            raise RuntimeError("Captions are disabled for this video.") from exc
        except NoTranscriptFound as exc:
            raise RuntimeError("No usable transcript was found for this video.") from exc
        except VideoUnavailable as exc:
            raise RuntimeError("This video is unavailable or private.") from exc
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Could not fetch or translate this video's transcript: {exc}") from exc

        segments = [
            TranscriptSegment(str(item.text), float(item.start), float(item.start) + float(item.duration))
            for item in fetched
            if str(item.text).strip()
        ]
        if not segments:
            raise RuntimeError("The video transcript is empty.")
        return PreparedTranscript(video_id=video_id, language="en", segments=segments)

    @staticmethod
    def _select_translatable(transcripts, preferred_languages: list[str] | None):
        candidates = list(transcripts)
        for language in preferred_languages or []:
            match = next((item for item in candidates if item.language_code == language), None)
            if match and match.is_translatable:
                return match
        return next((item for item in candidates if item.is_translatable), None)
