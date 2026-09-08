import unittest

from youtube_rag.cleaning import TranscriptCleaner
from youtube_rag.generation import format_timestamp
from youtube_rag.loader import extract_video_id
from youtube_rag.models import PreparedTranscript, TranscriptSegment
from youtube_rag.retrieval import MMRTranscriptRetriever
from youtube_rag.splitter import SentenceAwareTranscriptSplitter


class VideoIdExtractionTests(unittest.TestCase):
    def test_accepts_watch_shorts_and_raw_id_formats(self):
        self.assertEqual(extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("https://www.youtube.com/shorts/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id("dQw4w9WgXcQ"), "dQw4w9WgXcQ")

    def test_rejects_input_that_is_not_a_valid_id_or_url(self):
        with self.assertRaises(ValueError):
            extract_video_id("too-short")
        with self.assertRaises(ValueError):
            extract_video_id("https://example.com/watch?v=dQw4w9WgXcQ")


class TranscriptCleanerTests(unittest.TestCase):
    def test_strips_bracketed_noise_and_unescapes_html(self):
        transcript = PreparedTranscript(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[
                TranscriptSegment("  Hello   world  ", 0, 1),
                TranscriptSegment("[Music] Rock &amp; roll", 1, 2),
            ],
        )
        cleaned = TranscriptCleaner().clean(transcript)
        self.assertEqual(cleaned.segments[0].text, "Hello world")
        self.assertEqual(cleaned.segments[1].text, "Rock & roll")

    def test_raises_when_nothing_is_left_after_cleaning(self):
        transcript = PreparedTranscript(
            video_id="dQw4w9WgXcQ",
            language="en",
            segments=[TranscriptSegment("[music]", 0, 1), TranscriptSegment("[Applause]", 1, 2)],
        )
        with self.assertRaises(RuntimeError):
            TranscriptCleaner().clean(transcript)


class SplitterValidationTests(unittest.TestCase):
    def test_rejects_overlap_greater_than_or_equal_to_chunk_size(self):
        with self.assertRaises(ValueError):
            SentenceAwareTranscriptSplitter(chunk_size=100, chunk_overlap=100)

    def test_rejects_non_positive_chunk_size(self):
        with self.assertRaises(ValueError):
            SentenceAwareTranscriptSplitter(chunk_size=0, chunk_overlap=0)


class RetrieverValidationTests(unittest.TestCase):
    def test_rejects_non_positive_top_k(self):
        with self.assertRaises(ValueError):
            MMRTranscriptRetriever(vector_store=None, top_k=0, fetch_k=10)

    def test_rejects_fetch_k_smaller_than_top_k(self):
        with self.assertRaises(ValueError):
            MMRTranscriptRetriever(vector_store=None, top_k=10, fetch_k=5)


class TimestampFormattingTests(unittest.TestCase):
    def test_formats_minutes_and_seconds_under_an_hour(self):
        self.assertEqual(format_timestamp(0), "0:00")
        self.assertEqual(format_timestamp(65), "1:05")

    def test_formats_hours_when_over_an_hour(self):
        self.assertEqual(format_timestamp(3661), "1:01:01")


if __name__ == "__main__":
    unittest.main()