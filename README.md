# Marvel Chatbot

Marvel is a Streamlit chatbot powered by LangGraph and Gemini. It supports general chat with web-search, calculator, and weather tools, plus timestamp-grounded Q&A over an attached YouTube video.

## Workflow

```text
START
  └─ active video attached?
       ├─ yes → youtube_rag → END
       │          (retrieve transcript evidence → grounded Gemini answer)
       └─ no  → general_agent → tool needed?
                    ├─ yes → general_tools → general_agent
                    └─ no  → END
```

For review, `marvel_workflow.py` defines state, nodes, routes, and the graph;
`marvel_tools.py` owns general tools; `youtube_rag/` owns video ingestion and
retrieval; and `langgraph_tool_backend.py` only wires SQLite persistence.

## Setup

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run frontend.py
```

Set `GOOGLE_API_KEY` in `.env`. The weather tool additionally needs `WEATHERSTACK_API_KEY`.

## YouTube RAG

1. Start a new or existing chat.
2. In the sidebar, paste a YouTube watch, share, shorts, embed URL, or video ID.
3. Select **Attach and index video**.
4. Ask questions normally. While the video is attached, answers are based only on its retrieved transcript passages and include clickable timestamp citations.

Indexes are cached under `data/youtube_indexes/` and metadata is stored in `marvel_chatbot.db`. Both are local application data and are excluded from Git.

Videos with disabled/unavailable captions cannot be indexed. Remove video context in the sidebar to return a chat to the general-purpose assistant.

### RAG pipeline

The code follows one explicit data flow:

```text
YouTubeTranscriptLoader
  → TranscriptCleaner
  → SentenceAwareTranscriptSplitter
  → FaissVideoIndex (Gemini embeddings)
  → MMRTranscriptRetriever (top 5)
  → GroundedAnswerGenerator
```

Each stage lives in its matching module inside `youtube_rag/`. The loader prefers
English captions and otherwise uses YouTube's transcript translation to English.
The splitter prefers sentence boundaries and preserves timestamp metadata on every
LangChain `Document`. The final answer is generated only from the five MMR-selected
chunks and includes YouTube timestamp links.
