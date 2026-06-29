"""
Document Chat Bot — RAG pipeline powered by Google Gemini
----------------------------------------------------------
Supported file types:
- PDF  (.pdf)
- Excel (.xlsx, .xls)
- Image (.jpg, .jpeg, .png) — OCR via pytesseract if available
- Plain text (.txt)
"""

import os
import io
from typing import Optional, Tuple, List, Dict
import numpy as np
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

load_dotenv()

# ── Gemini API config ─────────────────────────────────────────────────────────
GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.0-flash:generateContent"
)
DEFAULT_MODEL = "gemini-2.0-flash"


def _get_api_key() -> Optional[str]:
    """
    Resolve the Gemini API key.
    Priority: Streamlit secrets (cloud) -> .env / environment variable (local).
    """
    try:
        import streamlit as st
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return str(key)
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY")


# ---------- 1. DOCUMENT LOADING ----------

def load_pdf_bytes(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_excel_bytes(file_bytes: bytes) -> str:
    import pandas as pd
    df_dict = pd.read_excel(io.BytesIO(file_bytes), sheet_name=None)
    lines = []
    for sheet_name, df in df_dict.items():
        lines.append(f"[Sheet: {sheet_name}]")
        for _, row in df.iterrows():
            lines.append(" | ".join(str(v) for v in row.values if str(v) != "nan"))
    return "\n".join(lines)


def load_image_bytes(file_bytes: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
        img = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(img)
    except Exception:
        return "[Image uploaded but OCR is not available on this server. Please upload a PDF or text file instead.]"


def load_bytes(file_bytes: bytes, filename: str) -> str:
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        return load_pdf_bytes(file_bytes)
    elif ext in ("xlsx", "xls"):
        return load_excel_bytes(file_bytes)
    elif ext in ("jpg", "jpeg", "png"):
        return load_image_bytes(file_bytes)
    else:
        return file_bytes.decode("utf-8", errors="replace")


def load_document(file_path: str) -> str:
    with open(file_path, "rb") as f:
        return load_bytes(f.read(), os.path.basename(file_path))


# ---------- 2. CHUNKING ----------

def chunk_text(text: str, chunk_size: int = 100, overlap: int = 20) -> List[str]:
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


# ---------- 3. VECTOR STORE ----------

class VectorStore:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.documents: List[str] = []
        self.metadatas: List[Dict] = []
        self.doc_matrix = None
        self.full_texts: Dict[str, str] = {}

    def add_chunks(self, chunks: List[str], source: str, full_text: str = ""):
        start_index = len(self.documents)
        self.documents.extend(chunks)
        self.metadatas.extend(
            [{"source": source, "chunk_index": start_index + i} for i in range(len(chunks))]
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.documents)
        if full_text:
            self.full_texts[source] = full_text

    def query(self, question: str, top_k: int = 5) -> Tuple[List[str], bool]:
        if self.doc_matrix is None or not self.documents:
            return [], False

        query_vec = self.vectorizer.transform([question])
        similarities = cosine_similarity(query_vec, self.doc_matrix).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        scored = [self.documents[i] for i in top_indices if similarities[i] > 0]
        if scored:
            return scored, False

        # Fallback: return first top_k chunks when no keyword match found
        return self.documents[:top_k], True

    def clear(self):
        self.__init__()


# ---------- 4. GEMINI LLM ----------

def build_prompt(question: str, contexts: List[str], used_fallback: bool = False) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    note = (
        "\n(Note: no exact keyword match was found, full document context is provided.)"
        if used_fallback else ""
    )
    return (
        "You are a helpful assistant. Answer the question using ONLY the context below.\n"
        "If the answer is present, give it directly and quote the relevant part.\n"
        f"If the answer truly is not in the context, say so briefly.{note}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )


def get_answer(
    question: str,
    contexts: List[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    used_fallback: bool = False,
) -> str:
    if not contexts:
        return "No document has been indexed yet. Please upload a file first."

    prompt = build_prompt(question, contexts, used_fallback)

    url = f"{GEMINI_API_URL}?key={api_key}"

    response = requests.post(
        url=url,
        headers={"Content-Type": "application/json"},
        json={
            "contents": [
                {"parts": [{"text": prompt}]}
            ],
            "generationConfig": {
                "maxOutputTokens": 600,
                "temperature": 0.2,
            },
        },
        timeout=60,
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"Gemini API error ({response.status_code}): {response.text}"
        )

    data = response.json()
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"Unexpected Gemini response format: {data}") from e


# ---------- 5. CHATBOT ----------

class RAGChatbot:
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        self.store = VectorStore()
        self.api_key = api_key or _get_api_key()
        self.model = model

    def is_ready(self) -> bool:
        return bool(self.api_key)

    def ingest_bytes(self, file_bytes: bytes, filename: str) -> int:
        text = load_bytes(file_bytes, filename)
        chunks = chunk_text(text)
        self.store.add_chunks(chunks, source=filename, full_text=text)
        return len(chunks)

    def ingest(self, file_path: str) -> int:
        with open(file_path, "rb") as f:
            return self.ingest_bytes(f.read(), os.path.basename(file_path))

    def ask(self, question: str, top_k: int = 5) -> Dict:
        if not self.api_key:
            return {
                "question": question,
                "answer": "⚠️ API key not configured. Please add GEMINI_API_KEY to Streamlit secrets.",
                "retrieved_chunks": [],
                "used_fallback": False,
            }
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
    if not bot.is_ready():
        print("ERROR: No GEMINI_API_KEY found in environment.")
        exit(1)

    sample_docs = ["sample_data/company_handbook.txt"]
    for doc in sample_docs:
        if os.path.exists(doc):
            n = bot.ingest(doc)
            print(f"Loaded {doc} -> {n} chunks")

    print("\nDocument Chat Bot ready. Type 'exit' to quit.\n")
    while True:
        q = input("You: ")
        if q.lower() == "exit":
            break
        result = bot.ask(q)
        print(f"\nBot: {result['answer']}\n")
