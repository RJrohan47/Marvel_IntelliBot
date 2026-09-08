import unittest

from langchain_core.messages import HumanMessage

from marvel_workflow import route_initial_request
from youtube_rag.loader import extract_video_id
from youtube_rag.models import PreparedTranscript, TranscriptSegment
from youtube_rag.splitter import SentenceAwareTranscriptSplitter


class RoutingRegressionTests(unittest.TestCase):
    """Protect the boundary between Marvel's general agent and video RAG."""

    def test_general_agent_is_used_without_video_context(self):
        self.assertEqual(route_initial_request({"messages": [HumanMessage(content="What's the weather?")]}), "general_agent")
        self.assertEqual(route_initial_request({"messages": [], "active_video_id": None}), "general_agent")

    def test_video_rag_is_used_only_with_an_active_video(self):
        self.assertEqual(
            route_initial_request({"messages": [HumanMessage(content="Summarize it")], "active_video_id": "dQw4w9WgXcQ"}),
            "youtube_rag",
        )

    def test_youtube_url_handling_and_timestamp_chunking(self):
        self.assertEqual(extract_video_id("https://youtu.be/dQw4w9WgXcQ"), "dQw4w9WgXcQ")
        chunks = SentenceAwareTranscriptSplitter(chunk_size=20, chunk_overlap=5).split(
            PreparedTranscript(
                video_id="dQw4w9WgXcQ",
                language="en",
                segments=[TranscriptSegment("first.", 0, 2), TranscriptSegment("second.", 2, 4)],
            )
        )
        self.assertEqual(len(chunks), 1)
        self.assertEqual((chunks[0].metadata["start_seconds"], chunks[0].metadata["end_seconds"]), (0, 4))


if __name__ == "__main__":
    unittest.main()
