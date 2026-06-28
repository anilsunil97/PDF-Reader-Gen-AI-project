"""
RAG-based Document Q&A Chatbot
--------------------------------
A Retrieval-Augmented Generation chatbot that answers questions
based on the content of uploaded documents (PDF, Excel, or image).

Pipeline:
1. Load and chunk documents
2. Vectorize chunks (TF-IDF) and store them in a local in-memory vector store
3. On a user query: vectorize the query, retrieve the most relevant chunks
   using cosine similarity
4. Pass retrieved chunks + query to an LLM via the OpenRouter API to
   generate a grounded answer

Supported file types:
- PDF  (.pdf)  — text extracted via pypdf
- Excel (.xlsx, .xls) — rows converted to text via openpyxl / pandas
- Image (.jpg, .jpeg, .png) — text extracted via pytesseract (OCR)
- Plain text (.txt)

Author: Anil Kumar
"""

import os
import io
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# Load .env for local development
load_dotenv()


def _get_api_key() -> str | None:
    """
    Resolve the OpenRouter API key.
    Priority: Streamlit secrets (cloud) → .env / environment variable (local).
    """
    try:
        import streamlit as st
        key = st.secrets.get("OPENROUTER_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.environ.get("OPENROUTER_API_KEY")

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"


# ---------- 1. DOCUMENT LOADING ----------

def load_pdf_bytes(file_bytes: bytes) -> str:
    """Extract text from PDF bytes."""
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_excel_bytes(file_bytes: bytes) -> str:
    """Convert Excel rows to plain text."""
    import pandas as pd
    df_dict = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    lines = []
    for sheet_name, df in df_dict.items():
        lines.append(f"[Sheet: {sheet_name}]")
        for _, row in df.iterrows():
            lines.append(" | ".join(str(v) for v in row.values if str(v) != "nan"))
    return "\n".join(lines)


def load_image_bytes(file_bytes: bytes) -> str:
    """Extract text from image using OCR (pytesseract)."""
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img)
    except ImportError:
        return "[OCR not available: install pytesseract and Pillow to extract text from images]"
    except Exception as e:
        return f"[Image OCR failed: {e}]"


def load_bytes(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the right loader based on file extension."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return load_pdf_bytes(file_bytes)
    elif ext in ("xlsx", "xls"):
        return load_excel_bytes(file_bytes)
    elif ext in ("jpg", "jpeg", "png"):
        return load_image_bytes(file_bytes)
    elif ext == "txt":
        return file_bytes.decode("utf-8", errors="replace")
    else:
        return file_bytes.decode("utf-8", errors="replace")


def load_document(file_path: str) -> str:
    """Load a document from disk (used by CLI mode)."""
    with open(file_path, "rb") as f:
        return load_bytes(f.read(), os.path.basename(file_path))


# ---------- 2. CHUNKING ----------

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> list[str]:
    """
    Split text into overlapping word-based chunks so context isn't lost
    at chunk boundaries.
    """
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# ---------- 3. VECTOR STORE (TF-IDF, fully local) ----------

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.doc_matrix = None
        # Full raw text per source — used as fallback when TF-IDF scores are all zero
        self.full_texts: dict[str, str] = {}

    def add_chunks(self, chunks: list[str], source: str, full_text: str = ""):
        start_index = len(self.documents)
        self.documents.extend(chunks)
        self.metadatas.extend(
            [{"source": source, "chunk_index": start_index + i} for i in range(len(chunks))]
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)
        if full_text:
            self.full_texts[source] = full_text

    def query(self, question: str, top_k: int = 5) -> tuple[list[str], bool]:
        """
        Returns (chunks, used_fallback).
        - Tries TF-IDF similarity first.
        - If all scores are zero (vague query / no keyword overlap), falls back to
          returning the top-K chunks by position so the LLM still has real context.
        """
        if self.doc_matrix is None or not self.documents:
            return [], False

        query_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        scored = [self.documents[i] for i in top_indices if similarities[i] > 0]
        if scored:
            return scored, False

        # Fallback: no TF-IDF match — return first top_k chunks (broadest context)
        fallback = self.documents[:top_k]
        return fallback, True

    def get_full_texts(self) -> list[str]:
        """Return all stored full-document texts for broad fallback prompting."""
        return list(self.full_texts.values())

    def clear(self):
        self.__init__()


# ---------- 4. LLM ANSWER GENERATION ----------

def build_prompt(question: str, contexts: list[str], used_fallback: bool = False) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    note = (
        "\n(Note: no exact keyword match was found, so the full document is provided as context.)"
        if used_fallback else ""
    )
    return f"""You are a helpful assistant. Answer the question using ONLY the context below.
If the answer is present, give it directly and quote the relevant part.
If the answer truly is not in the context, say so briefly.{note}

Context:
{context_block}

Question: {question}

Answer:"""


def get_answer(
    question: str,
    contexts: list[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    used_fallback: bool = False,
) -> str:
    if not contexts:
        return "No document has been indexed yet. Please upload a file first."

    prompt = build_prompt(question, contexts, used_fallback)

    response = requests.post(
        url=OPENROUTER_API_URL,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 600,
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"OpenRouter API error ({response.status_code}): {response.text}"
        )

    return response.json()["choices"][0]["message"]["content"]


# ---------- 5. END-TO-END PIPELINE ----------

class RAGChatbot:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL):
        self.store = VectorStore()
        self.api_key = api_key or _get_api_key()
        self.model = model
        if not self.api_key:
            raise ValueError(
                "No OpenRouter API key found. Set OPENROUTER_API_KEY in your "
                ".env file (local) or in the Streamlit Cloud Secrets dashboard."
            )

    def ingest_bytes(self, file_bytes: bytes, filename: str) -> int:
        """Ingest a file from raw bytes (used by Streamlit uploader)."""
        text = load_bytes(file_bytes, filename)
        chunks = chunk_text(text)
        self.store.add_chunks(chunks, source=filename, full_text=text)
        return len(chunks)

    def ingest(self, file_path: str) -> int:
        """Ingest a file from disk (used by CLI)."""
        with open(file_path, "rb") as f:
            return self.ingest_bytes(f.read(), os.path.basename(file_path))

    def ask(self, question: str, top_k: int = 5) -> dict:
        contexts, used_fallback = self.store.query(question, top_k=top_k)
        answer = get_answer(question, contexts, self.api_key, self.model, used_fallback)
        return {
            "question": question,
            "answer": answer,
            "retrieved_chunks": contexts,
            "used_fallback": used_fallback,
        }

    def reset(self):
        self.store.clear()


# ---------- CLI ----------

if __name__ == "__main__":
    bot = RAGChatbot()

    sample_docs = ["sample_data/company_handbook.txt"]
    for doc in sample_docs:
        if os.path.exists(doc):
            n = bot.ingest(doc)
            print(f"Loaded {doc} -> {n} chunks")

    print("\nRAG Chatbot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        result = bot.ask(q)
        print(f"\nBot: {result['answer']}\n")
