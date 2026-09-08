"""Stage 3: sentence-aware splitting with character overlap and timestamps."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from youtube_rag.models import PreparedTranscript


@dataclass(frozen=True)
class _TimedSpan:
    start_character: int
    end_character: int
    start_seconds: float
    end_seconds: float


class SentenceAwareTranscriptSplitter:
    """Prefer sentence boundaries, then use progressively smaller separators.

    ``chunk_overlap`` is measured in characters. A sentence boundary is used
    whenever possible; a very long single sentence may still need a word split.
    """

    def __init__(self, chunk_size: int, chunk_overlap: int) -> None:
        if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            add_start_index=True,
            keep_separator="end",
            separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", " ", ""],
        )

    def split(self, transcript: PreparedTranscript) -> list[Document]:
        text, spans = self._assemble_text_and_spans(transcript)
        split_documents = self.splitter.split_documents(
            [Document(page_content=text, metadata={"video_id": transcript.video_id, "language": transcript.language})]
        )
        starts = [span.start_character for span in spans]
        chunks = []
        for chunk_index, document in enumerate(split_documents):
            start = int(document.metadata.pop("start_index"))
            end = start + len(document.page_content)
            metadata = {
                **document.metadata,
                "chunk_index": chunk_index,
                "start_seconds": self._timestamp_for(start, starts, spans, is_end=False),
                "end_seconds": self._timestamp_for(max(start, end - 1), starts, spans, is_end=True),
            }
            chunks.append(Document(page_content=document.page_content, metadata=metadata))
        return chunks

    @staticmethod
    def _assemble_text_and_spans(transcript: PreparedTranscript) -> tuple[str, list[_TimedSpan]]:
        parts, spans = [], []
        cursor = 0
        for segment in transcript.segments:
            if parts:
                parts.append(" ")
                cursor += 1
            start = cursor
            parts.append(segment.text)
            cursor += len(segment.text)
            spans.append(_TimedSpan(start, cursor, segment.start_seconds, segment.end_seconds))
        return "".join(parts), spans

    @staticmethod
    def _timestamp_for(position: int, starts: list[int], spans: list[_TimedSpan], is_end: bool) -> float:
        span_index = max(0, bisect_right(starts, position) - 1)
        span = spans[span_index]
        return span.end_seconds if is_end else span.start_seconds
