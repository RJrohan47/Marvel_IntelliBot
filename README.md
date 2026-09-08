# Marvel IntelliBot

Marvel is a Streamlit chatbot I built with LangGraph and Gemini. It's a general assistant with tool use (web search, a calculator, live weather), and it can switch into a grounded YouTube Q&A mode — attach a video and ask questions about it, and it'll only answer from what's actually in the transcript, with timestamp links back to where it found the answer.

## Features

- General chat backed by Gemini, with tool calling for web search, a calculator, and live weather
- YouTube video Q&A: paste a URL or video ID, ask questions grounded only in that video's transcript
- Every answer includes clickable timestamp citations pointing back to the source clip
- If the transcript doesn't contain the answer, it says so instead of guessing
- The transcript is treated as untrusted data in the prompt, not as instructions — text inside a video's captions can't hijack the assistant
- Chat history persists across restarts (SQLite via LangGraph's checkpointer), and past threads are switchable from the sidebar
- A video's transcript is embedded and indexed once; later questions reuse the cached FAISS index instead of rebuilding it
- New chats get an auto-generated title from the first message

## How it works

Marvel's routing is an explicit LangGraph state machine. I wanted whether a turn goes to the general agent or to video RAG to be a plain, testable decision the graph makes up front, not something buried inside one big node:

```text
START
  └─ active video attached?
       ├─ yes → youtube_rag → END
       │          (retrieve transcript evidence → grounded Gemini answer)
       └─ no  → general_agent → tool needed?
                    ├─ yes → general_tools → general_agent
                    └─ no  → END
```

- `marvel_workflow.py` — graph state, nodes, routing
- `marvel_tools.py` — the general-purpose tools (search, calculator, weather)
- `youtube_rag/` — video ingestion, indexing, retrieval
- `langgraph_tool_backend.py` — wires the graph to a SQLite checkpointer
- `thread_manager.py` — sidebar chat history (titles, ordering)

### The YouTube RAG pipeline

When a video is attached for the first time, it goes through six stages, each in its own module so I could test and tweak them independently:

```text
YouTubeTranscriptLoader        → fetch captions (or auto-translate to English)
       ↓
TranscriptCleaner              → strip [music]/[applause] artifacts, normalize whitespace
       ↓
SentenceAwareTranscriptSplitter→ chunk on sentence boundaries, keep timestamps per chunk
       ↓
FaissVideoIndex                → embed chunks (Gemini embeddings), persist a per-video FAISS index
       ↓
MMRTranscriptRetriever         → top-k retrieval via Maximal Marginal Relevance
       ↓
GroundedAnswerGenerator        → answer from retrieved evidence only, with timestamp citations
```

Indexes are cached to disk under `data/youtube_indexes/`, so asking about a video I've already indexed is instant instead of re-embedding it. Videos with disabled or missing captions can't be indexed.

## Project structure

```text
Marvel_IntelliBot/
├── frontend.py                  # Streamlit UI: chat window, sidebar, thread switching
├── langgraph_tool_backend.py    # Entry point: wires the graph to SQLite persistence
├── marvel_workflow.py           # Graph state, nodes, routing (general agent <-> video RAG)
├── marvel_tools.py              # General tools: web search, calculator, weather
├── thread_manager.py            # Chat sidebar metadata (titles, ordering) in SQLite
├── requirements.txt
├── .env.example
├── tests/
│   ├── test_routing.py          # Routing, video-ID parsing, chunk/timestamp tests
│   └── test_youtube_rag.py      # Cleaner, splitter, retriever, and timestamp formatting tests
└── youtube_rag/
    ├── config.py                # Tunable pipeline settings (chunk size, k, etc.)
    ├── loader.py                # Transcript fetching + video-ID extraction
    ├── cleaning.py               # Caption cleanup
    ├── splitter.py               # Sentence-aware, timestamp-preserving chunking
    ├── indexing.py               # Embeddings + FAISS index build/load
    ├── retrieval.py              # MMR-based retrieval
    ├── generation.py             # Grounded answer generation + timestamp citations
    └── service.py                # Orchestration + thread-to-video persistence
```

## Tech stack

- **UI:** Streamlit
- **Orchestration:** LangGraph — explicit state graph, conditional routing, `ToolNode`
- **LLM:** Google Gemini for chat and embeddings, via `langchain-google-genai`
- **Vector store:** FAISS, local and per-video, cached to disk
- **Persistence:** SQLite — chat state through `langgraph-checkpoint-sqlite`, thread/video metadata through a small custom schema
- **Transcript ingestion:** `youtube-transcript-api`
- **Tools:** DuckDuckGo Search (`ddgs`), Weatherstack API

## Getting started

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (`GOOGLE_API_KEY`)
- *(Optional)* A [Weatherstack](https://weatherstack.com/) API key for the weather tool

### Installation

```bash
git clone https://github.com/RJrohan47/Marvel_IntelliBot.git
cd Marvel_IntelliBot

python -m venv .venv

# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env   # Windows: Copy-Item .env.example .env
```

Then set your key(s) in `.env`:

| Variable | Required | Description |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Used for both chat and embeddings. `GEMINI_API_KEY` also works as an alias. |
| `WEATHERSTACK_API_KEY` | No | Enables the weather tool; without it, weather queries return a "not configured" message. |
| `GEMINI_EMBED_MODEL` | No | Embedding model, default `models/gemini-embedding-001`. |
| `YOUTUBE_RAG_CHUNK_SIZE` | No | Transcript chunk size in characters, default `1200`. |
| `YOUTUBE_RAG_CHUNK_OVERLAP` | No | Chunk overlap in characters, default `200`. |
| `YOUTUBE_RAG_RETRIEVAL_K` | No | Chunks returned per question, default `5`. |
| `YOUTUBE_RAG_MMR_FETCH_K` | No | Candidate pool size for MMR, default `20`. |

### Run it

```bash
streamlit run frontend.py
```

Opens at `http://localhost:8501`.

## Usage

**General chat** — just start typing. Marvel calls web search, the calculator, or the weather tool automatically when a question needs one.

**YouTube video Q&A:**
1. Start a new or existing chat.
2. In the sidebar, paste a YouTube watch, share, shorts, or embed URL (or a raw 11-character video ID).
3. Click **Attach and index video** — Marvel fetches captions and builds a searchable index (only on first use; cached after that).
4. Ask questions normally. While a video is attached, answers come only from its transcript and include clickable timestamp links.
5. Click **Remove video context** to go back to the general assistant.

## Testing

```bash
python -m pytest tests/
```

`tests/test_routing.py` covers the routing decision between the general agent and video RAG, YouTube URL/ID parsing, and chunk/timestamp correctness after splitting. `tests/test_youtube_rag.py` covers the transcript cleaner, the splitter's and retriever's input validation, and timestamp formatting. None of these need an API key or network access to run.

Tests run automatically on every push and pull request via GitHub Actions (`.github/workflows/tests.yml`).

## Known limitations

- Single local SQLite database — no multi-user auth or cloud deployment yet
- English-only transcripts (or auto-translated to English)
- No live demo deployed yet

## License

MIT — see [LICENSE](LICENSE).

## Author

**Rohan Jain**
- Portfolio: [rohanjainportfolio.netlify.app](https://rohanjainportfolio.netlify.app)
- LinkedIn: [linkedin.com/in/rohanjain47](https://linkedin.com/in/rohanjain47)
- GitHub: [@RJrohan47](https://github.com/RJrohan47)