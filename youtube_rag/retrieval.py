"""Stage 5: retrieve diverse, relevant transcript chunks with MMR."""

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class MMRTranscriptRetriever:
    def __init__(self, vector_store: FAISS, top_k: int, fetch_k: int) -> None:
        if top_k < 1 or fetch_k < top_k:
            raise ValueError("fetch_k must be greater than or equal to top_k, and top_k must be positive.")
        self.vector_store = vector_store
        self.top_k = top_k
        self.fetch_k = fetch_k

    def retrieve(self, question: str) -> list[Document]:
        return self.vector_store.max_marginal_relevance_search(question, k=self.top_k, fetch_k=self.fetch_k)
