"""
Streamlit UI for the RAG Document Q&A Chatbot
----------------------------------------------
Upload a PDF, Excel, or image file and chat with its contents.
"""

import streamlit as st
from rag_chatbot import RAGChatbot

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Chat Bot",
    page_icon="🤖",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .stApp { background-color: #0f1117; color: #e0e0e0; }

        section[data-testid="stSidebar"] {
            background-color: #1a1d27;
            border-right: 1px solid #2e3250;
        }

        .chat-user {
            background: #1e3a5f;
            border-radius: 12px 12px 2px 12px;
            padding: 10px 14px;
            margin: 6px 0;
            max-width: 75%;
            margin-left: auto;
            font-size: 0.95rem;
        }
        .chat-bot {
            background: #1e2535;
            border: 1px solid #2e3250;
            border-radius: 12px 12px 12px 2px;
            padding: 10px 14px;
            margin: 6px 0;
            max-width: 80%;
            font-size: 0.95rem;
        }
        .chat-label {
            font-size: 0.72rem;
            color: #7a8499;
            margin-bottom: 3px;
        }

        [data-testid="stFileUploader"] {
            border: 2px dashed #2e3a5e !important;
            border-radius: 10px;
            padding: 10px;
        }

        .stTextInput > div > div > input {
            background-color: #1a1d27 !important;
            color: #e0e0e0 !important;
            border: 1px solid #2e3250 !important;
            border-radius: 8px;
        }

        .stButton > button {
            background: linear-gradient(135deg, #1e5cff, #7b2fff);
            color: white;
            border: none;
            border-radius: 8px;
            padding: 8px 20px;
            font-weight: 600;
        }
        .stButton > button:hover { opacity: 0.88; }

        .streamlit-expanderHeader { font-size: 0.82rem; color: #7a8499; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "bot" not in st.session_state:
    try:
        st.session_state.bot = RAGChatbot()
    except Exception as e:
        st.session_state.bot = None
        st.session_state._bot_error = str(e)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "ingested_files" not in st.session_state:
    st.session_state.ingested_files = []

# Key counter trick: incrementing this key resets the text_input widget,
# effectively clearing it after every send without triggering re-submission.
if "input_key" not in st.session_state:
    st.session_state.input_key = 0

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📂 Upload Documents")
    st.markdown(
        "<small style='color:#7a8499'>Supported: PDF · Excel · JPG / PNG · TXT</small>",
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        label="Drop files here",
        type=["pdf", "xlsx", "xls", "jpg", "jpeg", "png", "txt"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        new_files = [
            f for f in uploaded_files
            if f.name not in st.session_state.ingested_files
        ]
        if new_files:
            with st.spinner("Processing files…"):
                for uf in new_files:
                    try:
                        n_chunks = st.session_state.bot.ingest_bytes(uf.read(), uf.name)
                        st.session_state.ingested_files.append(uf.name)
                        st.success(f"✅ **{uf.name}** — {n_chunks} chunks indexed")
                    except Exception as e:
                        st.error(f"❌ {uf.name}: {e}")

    if st.session_state.ingested_files:
        st.markdown("---")
        st.markdown("**Indexed files**")
        for fname in st.session_state.ingested_files:
            ext = fname.rsplit(".", 1)[-1].upper()
            icon = {"PDF": "📄", "XLSX": "📊", "XLS": "📊",
                    "JPG": "🖼️", "JPEG": "🖼️", "PNG": "🖼️", "TXT": "📝"}.get(ext, "📁")
            st.markdown(f"{icon} `{fname}`")

    st.markdown("---")
    if st.button("🗑️ Clear all & start over", use_container_width=True):
        st.session_state.bot.reset()
        st.session_state.ingested_files = []
        st.session_state.chat_history = []
        st.session_state.input_key += 1
        st.rerun()

    st.markdown("---")
    st.markdown(
        "<small style='color:#4a5280'>Document Chat Bot · Powered by Llama 3.1 via Groq</small>",
        unsafe_allow_html=True,
    )

# ── Main panel ────────────────────────────────────────────────────────────────
st.markdown("## 🤖 Document Chat Bot")
st.markdown(
    "<p style='color:#7a8499; margin-top:-10px'>Upload a document on the left, then ask anything about it.</p>",
    unsafe_allow_html=True,
)

# Show a clear error if the bot failed to initialise (e.g. missing API key)
if st.session_state.bot is None or not st.session_state.bot.is_ready():
    st.error(
        "**API key not found.** "
        "Go to your Streamlit Cloud app → **Settings → Secrets** and add:\n\n"
        "```toml\nGROQ_API_KEY = \"gsk_...\"\n```\n\n"
        "Get a free key at [console.groq.com](https://console.groq.com) (no credit card), then click **Reboot app**.",
        icon="🔑",
    )
    st.stop()

# ── Chat history display ──────────────────────────────────────────────────────
with st.container():
    if not st.session_state.chat_history:
        st.markdown(
            """
            <div style='text-align:center; color:#4a5280; margin-top:60px'>
                <div style='font-size:3rem'>📄</div>
                <p>Upload a file in the sidebar to get started.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for entry in st.session_state.chat_history:
            role = entry["role"]
            content = entry["content"]

            if role == "user":
                st.markdown(
                    f"<div class='chat-label'>You</div>"
                    f"<div class='chat-user'>{content}</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<div class='chat-label'>Bot</div>"
                    f"<div class='chat-bot'>{content}</div>",
                    unsafe_allow_html=True,
                )



# ── Input row ─────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col1, col2 = st.columns([6, 1])
with col1:
    # Using a dynamic key means the widget is recreated (and thus cleared)
    # each time input_key is incremented — prevents stale value re-submission.
    question = st.text_input(
        "Ask a question",
        placeholder="e.g. What is the leave policy? / Summarise the data…",
        label_visibility="collapsed",
        key=f"question_input_{st.session_state.input_key}",
    )
with col2:
    send = st.button("Send ➤", use_container_width=True)

# ── Handle submission — only fires once per explicit user action ──────────────
if send and question.strip():
    if not st.session_state.ingested_files:
        st.warning("⚠️ Please upload at least one document first.")
    else:
        user_q = question.strip()

        # Save user message
        st.session_state.chat_history.append({"role": "user", "content": user_q})

        # Get answer
        with st.spinner("Thinking…"):
            try:
                result = st.session_state.bot.ask(user_q)
                answer = result["answer"]
                chunks = result["retrieved_chunks"]
                used_fallback = result.get("used_fallback", False)
            except Exception as e:
                answer = f"❌ Error: {e}"
                chunks = []
                used_fallback = False

        # Save bot message
        st.session_state.chat_history.append({
            "role": "bot",
            "content": answer,
            "chunks": chunks,
            "used_fallback": used_fallback,
        })

        # Increment key to clear the input box, then rerun to refresh display
        st.session_state.input_key += 1
        st.rerun()
