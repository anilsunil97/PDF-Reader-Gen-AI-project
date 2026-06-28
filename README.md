# RAG-based Document Q&A Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions
grounded in your own documents (PDF or text), instead of relying purely
on an LLM's training data.

## How it works

1. **Load & Chunk** — Documents are loaded and split into overlapping
   text chunks so context isn't lost at chunk boundaries.
2. **Vectorize & Store** — Each chunk is converted into a TF-IDF vector
   and stored in a local, in-memory vector store.
3. **Retrieve** — When a question is asked, it's vectorized the same way
   and compared to all stored chunks using cosine similarity. The most
   relevant chunks are retrieved.
4. **Generate** — The retrieved chunks are passed to Claude (Anthropic API)
   along with the question, so the model answers using only the
   retrieved context instead of guessing.

This is the same architecture used in production RAG systems (e.g.,
chat-with-your-docs tools, internal knowledge base assistants) — the
only difference here is the retrieval step uses TF-IDF instead of a
neural embedding model, which keeps the project fully self-contained
and runnable without downloading any external models. The LLM call
goes through OpenRouter, so you can swap between Claude, GPT, Gemini,
and other models just by changing one line.

## Setup

```bash
pip install -r requirements.txt
```

Then set your API key using the `.env` file (recommended):

1. Open the `.env` file in this folder
2. Replace `your-api-key-here` with your actual key from
   [openrouter.ai/keys](https://openrouter.ai/keys)

```
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxxxxxxxxxxxxxx
```

The script automatically loads this file at startup using `python-dotenv`,
so you don't need to set the environment variable manually each time.

**Do not commit your `.env` file or share it publicly** — it's already
listed in `.gitignore` to help prevent that. If you ever paste a real key
into chat, a doc, or a public repo, treat it as compromised and generate
a new one immediately from your provider's dashboard.

Alternatively, you can still set it directly in your shell instead of
using `.env`:
```bash
export OPENROUTER_API_KEY="your-api-key-here"   # macOS/Linux
set OPENROUTER_API_KEY=your-api-key-here        # Windows (cmd)
```

### Choosing a model

This project calls the LLM through [OpenRouter](https://openrouter.ai),
which gives access to many models (Claude, GPT, Gemini, etc.) through a
single API. The default model is set in `rag_chatbot.py`:

```python
DEFAULT_MODEL = "anthropic/claude-sonnet-4.5"
```

You can change this to any model slug listed on
[openrouter.ai/models](https://openrouter.ai/models), e.g.
`"openai/gpt-4o"` or `"google/gemini-2.0-flash-001"`.

## Run

```bash
python rag_chatbot.py
```

This will:
- Ingest the sample document at `sample_data/company_handbook.txt`
- Start an interactive chat loop where you can ask questions about it

Example:
```
You: How many days of WFH am I allowed?
Bot: You are allowed up to 2 days of work-from-home per week, subject
to manager approval...
```

## Using your own documents

```python
from rag_chatbot import RAGChatbot

bot = RAGChatbot()
bot.ingest("path/to/your_document.pdf")
bot.ingest("path/to/another_doc.txt")

result = bot.ask("Your question here")
print(result["answer"])
print(result["retrieved_chunks"])  # see exactly what context was used
```

## Possible extensions

- Swap TF-IDF for dense embeddings (sentence-transformers, OpenAI/Voyage
  embeddings) for better semantic retrieval on paraphrased questions
- Swap the in-memory store for a persistent vector database (ChromaDB,
  FAISS, Pinecone) for larger document sets
- Add a Streamlit UI for a simple web chat interface
- Add source citation (which document/chunk the answer came from)
