# 🤖 RAG-Powered Document Q&A System

[![Python](https://img.shields.io/badge/Python-3.10.7-blue.svg)]
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-FF4B4B.svg)]
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)]

An intelligent **Retrieval-Augmented Generation (RAG)** based document Q&A system that allows users to upload documents or provide URLs and get **accurate, context-aware answers**.
This system uses **Groq API** to generate responses with extremely low latency powered by **Groq’s LPU (Language Processing Unit) hardware acceleration** — resulting in faster and cost-efficient inference.

---

## 🚀 Key Features

* **Multi-format Document Support** (PDF, DOCX, TXT)
* **Semantic Search with Vector Embeddings**
* **Ultra-fast LLM Responses powered by Groq API**
* **Context-Referenced Answers (based strictly on retrieved chunks)**
* **Streamlit Frontend for Seamless Interaction**
* **Efficient Chunking + FAISS Vector Database**
* **Caching & Session State to avoid reprocessing**

---

## 🛠 Technology Stack

### Core

* **Python 3.10**
* **Streamlit** (Frontend UI)
* **LangChain** (Pipeline orchestration)

### AI & Vector Search

* **Groq API for LLM inference**
* **Sentence Transformers (mpnet-base-v2) for embeddings**
* **FAISS IndexFlatL2 for vector storage**

### Document Processing

* **PyPDF2** → PDFs
* **python-docx** → DOCX

---

## ⚡ Why Groq API?

| Feature                     | Advantage             |
| --------------------------- | --------------------- |
| Runs models on LPU hardware | Faster than GPUs      |
| Sub-second responses        | Ideal for Q&A systems |
| Cost-efficient inference    | Scales well           |
| Supports multiple models    | Mixtral, Gemma, etc.  |

📝 *Note:*
This project uses **Groq API for LLM inference** (not a custom Groq model; the backend model may vary).

---

## 🏗 Architecture

```
User → Streamlit UI → Document Loader → Chunking → Embeddings →
FAISS Vector Store → Groq API Model → Final Answer Response
```

---

## 📦 Installation

### 1️⃣ Clone Repository

```bash
git clone <your-repo-link>
cd RAG-QA
```

### 2️⃣ Setup Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate     # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add API Keys

Create `secret_api_keys.py`:

```python
groq_api_key = "your_groq_key_here"
huggingface_api_key = "your_hf_token_here"
```

---

## 🎯 Usage Guide

1️⃣ Upload document / paste URL
2️⃣ Click **Process Document**
3️⃣ Ask natural language queries
4️⃣ System retrieves relevant chunks
5️⃣ Groq API generates contextual answer

**Example Queries**

```
- Summarize the document in 5 bullet points.
- What are the challenges mentioned?
- Compare two approaches in this document.
```

---

## 🔧 RAG Configuration

Adjust inside `process_input()` :

```python
chunk_size = 1000
chunk_overlap = 100
k = 4   # retrieved chunks
```

LLM parameters inside `answer_question()`:

```python
temperature = 0.2
max_tokens = 400
```

---

## 🚀 Performance Highlights

* Cached embedding model loading
* FAISS optimized vector search
* Session state memory persistence
* Fast inference using Groq API



