# Marvel IntelliBot

Marvel is a Streamlit chatbot I built with LangGraph and Gemini. It's a general assistant with tool use (web search, a calculator, live weather), and it can switch into a grounded YouTube Q&A mode — attach a video and ask questions about it, and it'll only answer from what's actually in the transcript, with timestamp links back to where it found the answer.

## Features

- General chat backed by Gemini, with tool calling for web search, a calculator, live weather, and expense tracking
- YouTube video Q&A: paste a URL or video ID, ask questions grounded only in that video's transcript
- Every answer includes clickable timestamp citations pointing back to the source clip
- If the transcript doesn't contain the answer, it says so instead of guessing
- The transcript is treated as untrusted data in the prompt, not as instructions — text inside a video's captions can't hijack the assistant
- Chat history persists across restarts (SQLite via LangGraph's checkpointer), and past threads are switchable from the sidebar
- Short-term memory (STM) builds a bounded model context from a rolling summary and recent complete turns
- Long-term memory (LTM) saves selected user preferences and ongoing context in PostgreSQL for reuse across threads
- Inactive chat threads are automatically removed after 45 days without activity
- Expense tracking tools come from a separate local FastMCP server connected to the chatbot over stdio
- A video's transcript is embedded and indexed once; later questions reuse the cached FAISS index instead of rebuilding it
- New chats get an auto-generated title from the first message

## How it works

Marvel's routing is an explicit LangGraph state machine. I wanted whether a turn goes to the general agent or to video RAG to be a plain, testable decision the graph makes up front, not something buried inside one big node:

```text
START
  └─ active video attached?
       ├─ yes → youtube_rag → END
       │          (retrieve transcript evidence → grounded Gemini answer)
       └─ no  → Gemini general_agent → tool needed?
                    ├─ yes → general_tools → Gemini general_agent
                    └─ no  → END
```

Gemini is the primary model for normal chat, tool selection, and YouTube RAG generation. Groq-hosted OpenAI GPT-OSS is a fallback when a provider availability error occurs. Web search retries once and returns a controlled error to Gemini if the search service remains unavailable.

### Short-term memory (STM)

The general assistant uses STM to keep each model request within a configurable, approximate token budget while preserving thread history in the SQLite checkpointer:

1. STM estimates the size of the system prompt, tool schemas, saved summary, and conversation messages.
2. If the request is over budget, it keeps recent complete turns and summarizes older turns in batches, starting with the oldest.
3. The model receives the system prompt, the saved summary, and the recent turns. The summary and its message-count cutoff are saved in the thread state for later turns.

The full message list is still retained in the latest checkpoint; STM currently trims the model input only. Summarization is performed only when compaction is needed. If summarization fails, the workflow uses the last saved summary and recent turns, and instructs the assistant not to guess about missing context.

Token counts use LangChain's approximate counter, not the provider's exact tokenizer. The budget is a soft application limit and includes tool schemas in the main assistant request. Defaults can be adjusted with these environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `STM_CONTEXT_TOKEN_BUDGET` | `24000` | Approximate maximum size for the assembled model input |
| `STM_RECENT_TOKEN_BUDGET` | `8000` | Maximum approximate size of recent turns kept intact |
| `STM_SUMMARY_BATCH_TOKEN_BUDGET` | `8000` | Approximate size of each older batch sent for summarization |
| `STM_SUMMARY_TARGET_TOKEN_BUDGET` | `1500` | Approximate target size for the rolling summary |

STM emits estimated token counts and processing times through Python's `INFO` logger. Configure the app's logging level to `INFO` to see those metrics.

The YouTube Q&A route is separate: it uses the current question and attached video's retrieved transcript evidence rather than the general chat history.

### Long-term memory (LTM)

The app keeps a small user profile in LangGraph's PostgreSQL `PostgresStore`, separate from the SQLite checkpointer that stores thread history. It uses the shared namespace `("users", user_id, "user_memories")`; the default development user ID is `user_1`, so the same profile is available in each of that user's threads.

Before answering a new user turn, the app reads the existing profile and asks the configured model for structured memory updates from the latest user message. It saves only explicit, reusable details such as preferences and ongoing projects, then includes the updated profile in the current model context. The extractor does not use assistant or tool messages as new memory sources. Because extraction happens before response generation, a newly saved preference can affect the reply to the same turn. This adds a model call before the answer. If extraction or a per-turn store operation fails, the assistant continues without newly loaded memory.

Memory text is included in STM's approximate context-size calculation for general chat. The YouTube Q&A path can use saved preferences for response style only; transcript evidence remains the source for video facts. Semantic vector search is not enabled, so the initial store reads a bounded set of items from the user's namespace and does not require pgvector.

`LTM_USER_ID` is a development identity, not user authentication. A multi-user deployment must derive a stable user ID from authenticated identity instead of sharing `user_1`.
Deleting or expiring a chat removes that thread's checkpointed history; it does not delete the user's cross-thread LTM profile.

### Expense Tracker MCP server

Expense tracking is provided by a separate local project, `Expense-tracker-MCP-server`, built with **FastMCP**. The server exposes expense operations as MCP tools, including adding, updating, removing, and listing expenses, plus totals, averages, and summaries. It also exposes the available categories as the `expense://categories` MCP resource. The server stores expense records in its own local SQLite database (`expenses.db`); this is separate from both the chatbot's thread-history database and its PostgreSQL LTM store. The expense tracker is the source of truth for exact transaction amounts and dates.

At startup, `mcp_client.py` launches the FastMCP server as a local subprocess over MCP's **stdio** transport. `langchain-mcp-adapters` discovers the server's tools, and the chatbot adapts them for LangGraph's tool node alongside its other general-chat tools. This keeps the expense service as an independently runnable MCP server while letting the chatbot select and call its tools during a conversation. It is a local integration; the server is not hosted remotely.

The chatbot repository does not contain the separate MCP server project. The current client configuration expects it at `D:\MCP_server\Expense-tracker-MCP-server`, with its own Windows virtual environment at `.venv\Scripts\python.exe`. Install the MCP server's dependencies in that environment and update `SERVER_DIR` and `SERVER_PYTHON` in `mcp_client.py` if it lives elsewhere or you use another operating system. If the local server cannot start, the chatbot can still run, but expense tools are unavailable.

### Thread retention

The app deletes an entire thread after 45 days without activity, including its LangGraph checkpoints, thread metadata, and video association. Cleanup runs when the app starts and checks at most once per day per process. Active threads are not currently compacted in the checkpoint database, so their stored message history can continue to grow even though STM limits what is sent to the model.

### Key modules

- `marvel_workflow.py` — graph state, nodes, routing
- `marvel_tools.py` — the general-purpose tools (search, calculator, weather)
- `short_term_memory.py` — token estimation, recent-turn trimming, and batched summarization
- `long_term_memory.py` — user-scoped memory extraction, upserts, and context formatting
- `model_fallback.py` — Gemini and optional Groq model configuration and provider fallback
- `mcp_client.py` — starts the local FastMCP Expense Tracker server over stdio and adapts its tools for the graph
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
├── marvel_tools.py              # General tools and Expense Tracker MCP tools
├── short_term_memory.py         # Token estimation and summary-based model context
├── long_term_memory.py          # User-scoped profile shared across threads
├── model_fallback.py            # Primary and fallback chat model setup
├── mcp_client.py                # Expense Tracker MCP integration
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
- **LLMs:** Google Gemini for chat and embeddings, with OpenAI GPT-OSS 20B hosted on Groq available as a fallback
- **Vector store:** FAISS, local and per-video, cached to disk
- **Persistence:** SQLite for thread checkpoints and metadata; PostgreSQL for cross-thread LTM
- **STM:** approximate token counting, complete-turn trimming, and rolling summaries for older conversation turns
- **Transcript ingestion:** `youtube-transcript-api`
- **Tools:** DuckDuckGo Search (`ddgs`), Weatherstack API, and a local FastMCP Expense Tracker server connected over stdio

## Getting started

### Prerequisites

- Python 3.11+
- A [Google AI Studio](https://aistudio.google.com/apikey) API key (`GOOGLE_API_KEY`)
- PostgreSQL with a database and login role for long-term memory
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
| `GROQ_API_KEY` | No | Enables the Groq-hosted model as a fallback when Gemini encounters a provider availability error. |
| `GROQ_CHAT_MODEL` | No | Groq model ID, default `openai/gpt-oss-20b`. Used only as fallback. |
| `GEMINI_CHAT_MODEL` | No | Gemini chat model, default `gemini-3.6-flash`. |
| `POSTGRES_URI` | Yes | PostgreSQL connection string for LTM; URL-encode special characters in the password. |
| `LTM_USER_ID` | No | Stable user namespace ID, default `user_1` for local development. |
| `WEATHERSTACK_API_KEY` | No | Enables the weather tool; without it, weather queries return a "not configured" message. |
| `GEMINI_EMBED_MODEL` | No | Embedding model, default `models/gemini-embedding-001`. |
| `STM_CONTEXT_TOKEN_BUDGET` | No | STM soft input budget, default `24000` approximate tokens. |
| `STM_RECENT_TOKEN_BUDGET` | No | STM recent-turn budget, default `8000` approximate tokens. |
| `STM_SUMMARY_BATCH_TOKEN_BUDGET` | No | Maximum older-message batch size for summarization, default `8000` approximate tokens. |
| `STM_SUMMARY_TARGET_TOKEN_BUDGET` | No | Target rolling-summary size, default `1500` approximate tokens. |
| `YOUTUBE_RAG_CHUNK_SIZE` | No | Transcript chunk size in characters, default `1200`. |
| `YOUTUBE_RAG_CHUNK_OVERLAP` | No | Chunk overlap in characters, default `200`. |
| `YOUTUBE_RAG_RETRIEVAL_K` | No | Chunks returned per question, default `5`. |
| `YOUTUBE_RAG_MMR_FETCH_K` | No | Candidate pool size for MMR, default `20`. |

Expense tracking requires the separate FastMCP server described above. Its code, dependencies, virtual environment, and `expenses.db` are managed in that MCP server project. `mcp_client.py` currently points to a local Windows installation at `D:\MCP_server\Expense-tracker-MCP-server`; update `SERVER_DIR` and `SERVER_PYTHON` in that file for your environment. If the server is unavailable, the chatbot can still start, but expense tools will not be available.

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

- Local SQLite checkpoints and PostgreSQL LTM — no multi-user auth or cloud deployment yet
- STM keeps the full message list in checkpoints; it reduces model input size but does not compact active-thread storage
- STM token counts are estimates, and summaries may omit details; exact expense facts should be retrieved from the expense tracker tools
- LTM currently uses one configured user ID and has no authentication or per-user account management
- English-only transcripts (or auto-translated to English)
- No live demo deployed yet

## License

MIT — see [LICENSE](LICENSE).

## Author

**Rohan Jain**
- Portfolio: [rohanjainportfolio.netlify.app](https://rohanjainportfolio.netlify.app)
- LinkedIn: [linkedin.com/in/rohanjain47](https://linkedin.com/in/rohanjain47)
- GitHub: [@RJrohan47](https://github.com/RJrohan47)
