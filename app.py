"""
app.py — Streamlit web interface for the RAG Customer Support Assistant.

Features:
  - Chat-style message interface with avatars
  - Sidebar: PDF upload + live ingestion progress
  - Sidebar: ChromaDB status + chunk count
  - Sidebar: Escalation log viewer
  - Debug panel: retrieved chunk details per answer
  - Confidence score badge on every response
  - Session-based conversation history

Run with:
    streamlit run app.py
"""

import streamlit as st
import tempfile
import os
import time
from pathlib import Path

# ── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="RAG Support Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark gradient background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        min-height: 100vh;
    }

    /* Chat message containers */
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 18px 18px 4px 18px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .assistant-bubble {
        background: rgba(255, 255, 255, 0.08);
        backdrop-filter: blur(12px);
        color: #e8e8f0;
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 85%;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        font-size: 0.95rem;
        line-height: 1.6;
    }

    .escalated-bubble {
        background: linear-gradient(135deg, rgba(255, 152, 0, 0.15), rgba(255, 87, 34, 0.1));
        border: 1px solid rgba(255, 152, 0, 0.4);
        border-radius: 18px 18px 18px 4px;
        padding: 14px 18px;
        margin: 8px 0;
        max-width: 85%;
        color: #ffd180;
        font-size: 0.95rem;
        line-height: 1.6;
    }

    /* Confidence badge */
    .conf-badge-high {
        display: inline-block;
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 8px;
        vertical-align: middle;
    }

    .conf-badge-mid {
        display: inline-block;
        background: linear-gradient(135deg, #f7971e, #ffd200);
        color: #333;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 8px;
        vertical-align: middle;
    }

    .conf-badge-low {
        display: inline-block;
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 2px 10px;
        border-radius: 20px;
        margin-left: 8px;
        vertical-align: middle;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: rgba(15, 12, 41, 0.95) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* Status pill */
    .status-pill-ready {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    .status-pill-empty {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }

    /* Header */
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }

    .main-header h1 {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
    }

    .main-header p {
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.07) !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 12px 16px !important;
        font-family: 'Inter', sans-serif !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2) !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4) !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
        color: rgba(255,255,255,0.7) !important;
    }

    /* Divider */
    hr {
        border-color: rgba(255,255,255,0.08) !important;
    }

    /* Scrollable chat area */
    .chat-container {
        max-height: 60vh;
        overflow-y: auto;
        padding: 0 8px;
        scroll-behavior: smooth;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Imports (after st setup) ─────────────────────────────────────────────────
from embeddings.vector_store import get_chunk_count
from hitl.escalation import get_all_escalations


# ── Session State ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []       # [{role, content, meta}]
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "last_processed_query" not in st.session_state:
    st.session_state.last_processed_query = ""


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<h2 style='color:white; font-weight:700; margin-bottom:4px;'>⚙️ Control Panel</h2>",
        unsafe_allow_html=True,
    )
    st.divider()

    # ── ChromaDB Status Card ──────────────────────────────────────────────────
    chunk_count = get_chunk_count()
    if chunk_count > 0:
        st.markdown(
            f"""
            <div style="
                background: linear-gradient(135deg, rgba(17,153,142,0.25), rgba(56,239,125,0.15));
                border: 1px solid rgba(56,239,125,0.4);
                border-radius: 12px;
                padding: 14px 16px;
                margin-bottom: 4px;
            ">
                <div style="font-size:0.7rem; color:rgba(255,255,255,0.5); letter-spacing:1px; margin-bottom:4px;">KNOWLEDGE BASE</div>
                <div style="font-size:1.05rem; font-weight:700; color:#38ef7d;">✅ Ready to Answer</div>
                <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:4px;">{chunk_count:,} chunks indexed in ChromaDB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="
                background: linear-gradient(135deg, rgba(255,65,108,0.2), rgba(255,75,43,0.1));
                border: 1px solid rgba(255,65,108,0.4);
                border-radius: 12px;
                padding: 14px 16px;
                margin-bottom: 4px;
            ">
                <div style="font-size:0.7rem; color:rgba(255,255,255,0.5); letter-spacing:1px; margin-bottom:4px;">KNOWLEDGE BASE</div>
                <div style="font-size:1.05rem; font-weight:700; color:#ff6b6b;">⚠️ Empty — No Data Yet</div>
                <div style="font-size:0.8rem; color:rgba(255,255,255,0.6); margin-top:4px;">Upload a file below to get started</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Upload Section Header ─────────────────────────────────────────────────
    st.markdown(
        """
        <div style="margin-bottom: 10px;">
            <div style="font-size:0.7rem; color:rgba(255,255,255,0.5); letter-spacing:1px; margin-bottom:6px;">STEP 1 — UPLOAD YOUR DOCUMENT</div>
            <div style="font-size:0.85rem; color:rgba(255,255,255,0.8); line-height:1.5;">
                Drop a <strong style="color:#a78bfa;">PDF</strong> or <strong style="color:#a78bfa;">CSV</strong> file to
                build the knowledge base. Up to <strong style="color:#38ef7d;">1 GB</strong> supported.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Accepted formats info
    st.markdown(
        """
        <div style="
            display: flex; gap: 8px; margin-bottom: 10px; flex-wrap: wrap;
        ">
            <span style="background:rgba(167,139,250,0.15); border:1px solid rgba(167,139,250,0.3);
                         border-radius:6px; padding:3px 10px; font-size:0.75rem; color:#a78bfa;">
                📄 PDF — any text-based document
            </span>
            <span style="background:rgba(56,239,125,0.1); border:1px solid rgba(56,239,125,0.3);
                         border-radius:6px; padding:3px 10px; font-size:0.75rem; color:#38ef7d;">
                📊 CSV — Q&amp;A dataset (up to 1 GB)
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Upload PDF or CSV (max 1 GB)",
        type=["pdf", "csv"],
        label_visibility="visible",
        help=(
            "PDF: any readable text-based PDF (user manuals, FAQs, policies).\n\n"
            "CSV: single_qna.csv (columns: Question + Answer) or "
            "multi_questions.csv (paired with multi_answers.csv)."
        ),
    )

    st.divider()

    # ── Step 2: Options ───────────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem; color:rgba(255,255,255,0.5); letter-spacing:1px; margin-bottom:8px;'>STEP 2 — CONFIGURE OPTIONS</div>",
        unsafe_allow_html=True,
    )

    clear_existing = st.checkbox(
        "🗑️ Replace existing knowledge base",
        value=False,
        help=(
            "When checked: completely wipes the current ChromaDB data before ingesting the new file.\n\n"
            "When unchecked: new content is ADDED on top of existing data (safe for incremental updates)."
        ),
    )
    if clear_existing:
        st.warning("⚠️ This will permanently delete all existing indexed data before ingesting the new file.", icon="🗑️")

    # CSV-specific options
    max_rows = 5000
    category_filter = None
    answers_path = None

    if uploaded_file is not None and uploaded_file.name.endswith(".csv"):
        with st.expander("⚙️ CSV Advanced Options", expanded=True):
            st.caption("These settings only apply to CSV files.")
            max_rows = st.number_input(
                "Maximum rows to ingest",
                min_value=100,
                max_value=50000,
                value=5000,
                step=500,
                help=(
                    "Your CSV may contain millions of rows. "
                    "This caps how many are loaded. "
                    "5,000 rows ≈ 2–3 min ingestion time."
                ),
            )
            st.caption(f"Will ingest up to **{int(max_rows):,}** Q&A rows.")

            category_filter = st.text_input(
                "Filter by Category (optional)",
                placeholder="e.g.  Automotive   or   Electronics",
                help=(
                    "Only ingest rows matching this category name.\n"
                    "Leave blank to ingest ALL categories."
                ),
            ) or None
            if category_filter:
                st.info(f"Only rows where Category = **{category_filter}** will be ingested.")

            if "question" in uploaded_file.name.lower() and "multi" in uploaded_file.name.lower():
                st.markdown("---")
                st.markdown("**Multi-turn Q&A detected** — also provide the answers file:")
                answers_path = st.text_input(
                    "Full path to multi_answers.csv",
                    placeholder=r"C:\Users\LENOVO\Downloads\archive\multi_answers.csv",
                    help="Required when ingesting multi_questions.csv. The two files are joined on QuestionID.",
                ) or None
                if not answers_path:
                    st.warning("Provide the answers CSV path to enable multi-turn ingestion.")

    st.divider()

    # ── Step 3: Ingest Button ─────────────────────────────────────────────────
    st.markdown(
        "<div style='font-size:0.7rem; color:rgba(255,255,255,0.5); letter-spacing:1px; margin-bottom:8px;'>STEP 3 — START INGESTION</div>",
        unsafe_allow_html=True,
    )

    if uploaded_file is None:
        st.markdown(
            "<div style='color:rgba(255,255,255,0.35); font-size:0.82rem; text-align:center; padding:10px 0;'>← Upload a file first to enable ingestion</div>",
            unsafe_allow_html=True,
        )
    else:
        # File info preview
        file_size_mb = round(uploaded_file.size / (1024 * 1024), 1)
        file_type = "PDF" if uploaded_file.name.endswith(".pdf") else "CSV"
        st.markdown(
            f"""
            <div style="
                background: rgba(102,126,234,0.12);
                border: 1px solid rgba(102,126,234,0.3);
                border-radius: 10px;
                padding: 10px 14px;
                margin-bottom: 10px;
                font-size: 0.82rem;
                color: rgba(255,255,255,0.8);
            ">
                <strong>📁 {uploaded_file.name}</strong><br>
                <span style="color:rgba(255,255,255,0.5);">{file_type} · {file_size_mb} MB</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🚀 Start Ingestion", use_container_width=True, type="primary"):
            import tempfile, os
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            with st.spinner(f"Ingesting {uploaded_file.name}... this may take a few minutes for large files."):
                try:
                    from ingest_pipeline import run_ingestion
                    if suffix == ".pdf":
                        run_ingestion(pdf_path=tmp_path, clear=clear_existing)
                    else:
                        run_ingestion(
                            csv_path=tmp_path,
                            answers_path=answers_path,
                            clear=clear_existing,
                            max_rows=int(max_rows),
                            category_filter=category_filter,
                        )
                    os.unlink(tmp_path)
                    new_count = get_chunk_count()
                    st.success(
                        f"✅ **{uploaded_file.name}** ingested!\n\n"
                        f"Knowledge base now has **{new_count:,} chunks** ready for queries."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"❌ Ingestion failed: {exc}")
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)


    st.divider()

    # ── Debug Mode Toggle ─────────────────────────────────────────────────────
    st.session_state.debug_mode = st.toggle(
        "🔍 Show retrieval debug info",
        value=st.session_state.debug_mode,
    )

    st.divider()

    # ── Escalation Log Viewer ─────────────────────────────────────────────────
    st.markdown(
        "<p style='color:rgba(255,255,255,0.6); font-size:0.8rem; margin-bottom:8px;'>ESCALATION LOG</p>",
        unsafe_allow_html=True,
    )

    escalations = get_all_escalations()
    if escalations:
        for esc in escalations[:5]:  # Show latest 5
            status_icon = "✅" if esc.get("status") == "resolved" else "🔴"
            with st.expander(
                f"{status_icon} {esc['timestamp'][:16]} — {esc['query'][:40]}..."
            ):
                st.markdown(
                    f"**Reason:** {esc['escalation_reason']}",
                    unsafe_allow_html=False,
                )
                st.caption(f"Event ID: {esc['event_id']} | Status: {esc['status']}")
    else:
        st.caption("No escalations yet.")

    st.divider()

    # ── Clear Chat ────────────────────────────────────────────────────────────
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT AREA
# ══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 RAG Support Assistant</h1>
        <p>Powered by LangGraph · ChromaDB · sentence-transformers · Groq/Gemini</p>
    </div>
    """,
    unsafe_allow_html=True,
)
st.divider()

# ── Render Conversation History ───────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    meta = msg.get("meta", {})

    if role == "user":
        st.markdown(
            f'<div class="user-bubble">👤 {content}</div>',
            unsafe_allow_html=True,
        )
    else:
        # Choose bubble class
        is_escalated = "escalated to a human" in content.lower()
        bubble_class = "escalated-bubble" if is_escalated else "assistant-bubble"

        # Confidence badge
        score = meta.get("confidence_score", 0.0)
        if score >= 0.65:
            badge = f'<span class="conf-badge-high">⚡ {score:.2f}</span>'
        elif score >= 0.35:
            badge = f'<span class="conf-badge-mid">⚡ {score:.2f}</span>'
        else:
            badge = f'<span class="conf-badge-low">⚡ {score:.2f}</span>'

        st.markdown(
            f"""
            <div class="{bubble_class}">
                <div style="margin-bottom:6px;">
                    <span style="font-size:0.8rem;opacity:0.6;">🤖 Assistant</span>
                    {badge if score > 0 else ""}
                </div>
                {content.replace(chr(10), "<br>")}
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Debug panel
        if st.session_state.debug_mode and meta.get("retrieved_chunks"):
            with st.expander("🔍 Retrieval Details", expanded=False):
                chunks = meta["retrieved_chunks"]
                st.caption(
                    f"Confidence: {score:.4f} | "
                    f"Chunks: {len(chunks)} | "
                    f"Escalated: {meta.get('needs_escalation', False)}"
                )
                for i, c in enumerate(chunks, 1):
                    st.markdown(
                        f"**[{i}]** `score={c['score']:.4f}` | "
                        f"*{c['source_doc']}* p.{c['page_number']}"
                    )
                    st.code(c["text"][:300], language=None)

# ── Input Area ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.text_input(
        "Your question",
        placeholder="Ask anything about the documentation...",
        label_visibility="collapsed",
        key="user_input_field",
    )

with col2:
    send_clicked = st.button("Send ➤", use_container_width=True)

# ── Process Query ─────────────────────────────────────────────────────────────
if (send_clicked or user_input) and user_input.strip() and user_input.strip() != st.session_state.last_processed_query:
    query = user_input.strip()
    st.session_state.last_processed_query = query  # prevent re-processing on rerun

    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": query})

    # Check KB is populated
    if get_chunk_count() == 0:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": (
                    "⚠️ The knowledge base is empty. Please upload and ingest a PDF "
                    "using the sidebar before asking questions."
                ),
                "meta": {},
            }
        )
        st.rerun()

    # Run RAG graph
    with st.spinner("🔍 Retrieving & generating answer..."):
        try:
            from graph.workflow import run_query
            final_state = run_query(query)
            answer = final_state.get("final_answer", "No answer generated.")
            meta = {
                "confidence_score": final_state.get("confidence_score", 0.0),
                "needs_escalation": final_state.get("needs_escalation", False),
                "escalation_reason": final_state.get("escalation_reason", ""),
                "retrieved_chunks": final_state.get("retrieved_chunks", []),
            }
        except Exception as exc:
            answer = f"⚠️ System error: {exc}"
            meta = {}

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "meta": meta}
    )
    st.rerun()

# ── Empty State ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    if chunk_count > 0:
        hint = "Knowledge base is ready — ask your first question below! 👇"
    else:
        hint = "Start by uploading a PDF or CSV in the sidebar, then ask your first question."
    st.markdown(
        f"""
        <div style="text-align:center; padding: 60px 0; opacity: 0.45;">
            <div style="font-size: 4rem;">💬</div>
            <p style="color:rgba(255,255,255,0.7); font-size:1rem; margin-top:12px;">
                {hint}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
