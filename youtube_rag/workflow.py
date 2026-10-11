"""LangGraph subgraph for answering questions about an indexed YouTube video."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from youtube_rag.service import generate_video_answer, retrieve_video_chunks


class YouTubeRagState(TypedDict):
    """Private state used only while the YouTube RAG subgraph is running."""

    question: str
    video_id: str
    retrieved_chunks: list[Document]
    answer: str


def retrieve_transcript_evidence(state: YouTubeRagState) -> dict:
    """Find relevant chunks in the attached video's existing FAISS index."""
    return {
        "retrieved_chunks": retrieve_video_chunks(state["question"], state["video_id"])
    }


def generate_grounded_answer(state: YouTubeRagState) -> dict:
    """Use the configured answer model to answer from retrieved evidence."""
    return {
        "answer": generate_video_answer(
            state["question"], state["video_id"], state["retrieved_chunks"]
        )
    }


def build_youtube_rag_graph():
    """Build the private-state RAG subgraph used by Marvel's parent graph."""
    graph = StateGraph(YouTubeRagState)
    graph.add_node("retrieve_transcript_evidence", retrieve_transcript_evidence)
    graph.add_node("generate_grounded_answer", generate_grounded_answer)
    graph.add_edge(START, "retrieve_transcript_evidence")
    graph.add_edge("retrieve_transcript_evidence", "generate_grounded_answer")
    graph.add_edge("generate_grounded_answer", END)
    return graph.compile()


youtube_rag_graph = build_youtube_rag_graph()
