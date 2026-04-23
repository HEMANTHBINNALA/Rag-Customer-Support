# RAG Customer Support Assistant

> A production-ready Retrieval-Augmented Generation (RAG) system for customer support, built with **LangGraph**, **ChromaDB**, and **sentence-transformers**. Features a full Human-in-the-Loop (HITL) escalation pathway and a sleek Streamlit chat interface.

---

## 🏗️ Project Structure

```
Rag Project Innomatics/
│
├── ingestion/
│   ├── pdf_loader.py       # PDF text extraction (PyMuPDF)
│   ├── text_cleaner.py     # Noise removal
│   └── chunker.py          # Recursive text splitting
│
├── embeddings/
│   ├── embed_model.py      # MiniLM-L6-v2 sentence-transformers
│   └── vector_store.py     # ChromaDB CRUD operations
│
├── retrieval/
│   └── retriever.py        # Similarity search + context formatting
│
├── generation/
│   ├── prompt_builder.py   # Anti-hallucination prompt engineering
│   └── llm_client.py       # Unified Groq / Gemini LLM client
│
├── graph/
│   ├── state.py            # GraphState TypedDict
│   ├── nodes.py            # 7 LangGraph node functions
│   ├── edges.py            # Conditional routing logic
│   └── workflow.py         # Graph assembly + run_query()
│
├── hitl/
│   └── escalation.py       # HITL logging + resolution tracking
│
├── config.py               # Centralized settings (loaded from .env)
├── ingest_pipeline.py      # One-shot ingestion CLI
├── main.py                 # Interactive terminal chat
├── app.py                  # Streamlit web UI
├── requirements.txt
└── .env.example
```

---

## ⚡ Quick Start

### 1. Clone & Install

```bash
cd "Rag Project Innomatics"
pip install -r requirements.txt
```

### 2. Configure API Keys

```bash
copy .env.example .env
# Edit .env with your Groq or Gemini API key
```

Get a **free Groq API key** at: https://console.groq.com

### 3. Ingest Your PDF

```bash
python ingest_pipeline.py --pdf path/to/your_document.pdf
```

Add `--clear` to wipe the DB first:

```bash
python ingest_pipeline.py --pdf path/to/your_document.pdf --clear
```

### 4. Run the App

**Streamlit UI (recommended):**
```bash
streamlit run app.py
```

**Terminal CLI:**
```bash
python main.py
```

---

## 🔧 Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `groq` | `"groq"` or `"gemini"` |
| `GROQ_API_KEY` | — | Your Groq API key |
| `GOOGLE_API_KEY` | — | Your Google Gemini API key |
| `GROQ_MODEL` | `llama3-8b-8192` | Groq model name |
| `GEMINI_MODEL` | `gemini-1.5-flash` | Gemini model name |
| `CHUNK_SIZE` | `500` | Characters per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K_RESULTS` | `4` | Chunks to retrieve per query |
| `CONFIDENCE_THRESHOLD` | `0.35` | Min cosine similarity to answer |
| `CHROMA_DB_PATH` | `./chroma_db` | ChromaDB persistence directory |

---

## 🧠 Architecture

### LangGraph Workflow

```
START
  └──→ retrieve_node        (embed query → search ChromaDB)
             └──→ evaluate_node     (check confidence + sensitive keywords)
                      │
                      ├── [low confidence / sensitive]  ──→ escalate_node ──→ END
                      │
                      └── [confident]
                               └──→ generate_node    (LLM answer)
                                         └──→ validate_node  (check LLM uncertainty)
                                                  │
                                                  ├── [uncertain] ──→ escalate_node ──→ END
                                                  │
                                                  └── [confident] ──→ format_node ──→ END
```

### HITL Escalation Triggers

| Trigger | Condition |
|---|---|
| Low retrieval confidence | cosine similarity < 0.35 |
| No results | ChromaDB returns 0 chunks |
| Sensitive keyword | billing, refund, fraud, legal, cancel… |
| LLM uncertainty | "I don't know", "not in the document"… |

Escalation events are logged as JSON files in `./escalation_logs/`.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph 0.1+ |
| Vector DB | ChromaDB 0.5+ |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| LLM (primary) | Groq llama3-8b-8192 |
| LLM (fallback) | Google Gemini 1.5 Flash |
| PDF Parsing | PyMuPDF (fitz) |
| UI | Streamlit |
| Config | python-dotenv |

---

## 📊 Streamlit UI Features

- **Chat interface** with user/assistant message bubbles
- **PDF upload** directly in the sidebar → triggers full ingestion pipeline
- **Confidence score badge** on every assistant response (green/yellow/red)
- **Debug panel** — shows retrieved chunks, scores, and source pages per answer
- **Escalation log viewer** — shows latest escalation events with status
- **Dark glassmorphism design** with animated gradients

---

## 📁 Sample Workflow

```bash
# 1. Ingest your company FAQ PDF
python ingest_pipeline.py --pdf docs/company_faq.pdf

# 2. Launch the Streamlit UI
streamlit run app.py

# 3. Ask a question in the browser
# "What is your return policy?"
# → Retrieves relevant chunks → LLM answers → Cites page numbers

# 4. Ask a sensitive question
# "I want a refund for my order"
# → Escalation triggered → Human agent notified → JSON log created
```

---

## 🔮 Future Extensions

- **Multi-document support** — ingest multiple PDFs into separate collections
- **Async FastAPI backend** — decouple UI from processing for concurrent users
- **Celery + Redis task queue** — replace file-based escalation with real queuing
- **OCR support** — handle scanned PDFs via Tesseract
- **Evaluation pipeline** — RAGAS metrics for faithfulness, relevance, recall

---

*Built as part of the Innomatics RAG Project — demonstrating production-grade RAG system design with LangGraph orchestration.*
