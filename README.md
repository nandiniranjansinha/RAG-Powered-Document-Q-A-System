# RAG-Powered Document Q&A System

A Retrieval-Augmented Generation pipeline for multi-format document Q&A using FAISS vector search and Groq's LLaMA 3.3 70B.

---

## How It Works

1. **Upload** a PDF, DOCX, TXT, or paste raw text
2. **Chunks** split via `CharacterTextSplitter` (size=1000, overlap=100)
3. **Embedded** using `sentence-transformers/all-mpnet-base-v2` → stored in FAISS `IndexFlatL2`
4. **Query** retrieves top-k=4 chunks → passed to Groq API with a strict context-grounded prompt
5. **Fallback** — if Groq API fails, returns raw retrieved chunks directly

---

## Tech Stack

| Component | Tool |
|---|---|
| UI | Streamlit |
| Vector Store | FAISS (IndexFlatL2) |
| Embeddings | Hugging Face (all-mpnet-base-v2) |
| LLM | Groq API — llama-3.3-70b-versatile |
| Document Parsing | PyPDF2, python-docx |

---

## Setup

```bash
git clone https://github.com/nandiniranjansinha/RAG-Powered-Document-Q-A-System.git
cd RAG-Powered-Document-Q-A-System
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key
HUGGINGFACEHUB_API_TOKEN=your_huggingface_api_key
```

```bash
streamlit run app.py
```

---

## Project Structure

```
├── app.py                    # Main Streamlit app
├── evaluate_chunking.ipynb   # Chunking ablation experiments
├── requirements.txt
├── .env                      # API keys (never commit this)
├── .env.example              # Template for API keys
└── .gitignore
```

---

## Known Limitations

- `CharacterTextSplitter` ignores semantic boundaries — splits mid-sentence on structured documents
- FAISS index is in-memory only (no persistence across sessions)
- `normalize_embeddings=False` — effect on ranking quality unexplored
- No automated evaluation yet (RAGAS planned)

---

## Research Questions / Next Iteration

- Does `RecursiveCharacterTextSplitter` improve retrieval coherence vs `CharacterTextSplitter`?
- What is the effect of `normalize_embeddings=True` on ranking for longer documents?
- Can RAGAS faithfulness + relevancy scores quantify chunking config tradeoffs?
- Does k=4 retrieval saturate context or leave signal on the table for multi-section docs?
