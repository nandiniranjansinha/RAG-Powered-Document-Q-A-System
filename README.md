# 🧠 RAG-Powered Document Q&A System

A Retrieval-Augmented Generation (RAG) application that allows users to upload documents and ask questions, with answers generated using contextual retrieval and LLM inference.

---

## 🚀 Features

- Supports multiple input formats:
  - PDF
  - DOCX
  - TXT
  - Raw text input
- Semantic search using FAISS vector store  
- Context-grounded answer generation using Groq API  
- Fallback mechanism for API failures  
- Interactive UI built with Streamlit  

---

## 🏗️ Architecture Overview

### 1. Document Processing
- Extract text from uploaded documents (PDF, DOCX, TXT)
- Clean and prepare text for further processing

### 2. Chunking
- Text is split into smaller chunks:
  - chunk_size = 1000
  - chunk_overlap = 100

### 3. Embeddings
- Model: sentence-transformers/all-mpnet-base-v2
- Converts text into dense vector representations

### 4. Vector Store
- FAISS (IndexFlatL2)
- Stores embeddings for efficient similarity search

### 5. Retrieval
- Top-k similarity search (k=4)
- Retrieves most relevant chunks for a query

### 6. LLM Generation
- Groq API  
- Model: llama-3.3-70b-versatile  
- Uses custom prompt with retrieved context

### 7. Fallback Mechanism
- If API fails:
  - Returns top retrieved document chunks
  - Ensures system reliability

---

## ⚙️ Tech Stack

- Frontend: Streamlit  
- Vector Store: FAISS  
- Embeddings: Hugging Face (sentence-transformers)  
- LLM Inference: Groq API  
- Document Processing: PyPDF2, python-docx  
- NLP Utilities: NLTK  

---

## 📂 Project Structure

app.py  
requirements.txt  
secret_api_keys.py  
README.md  

---

## ▶️ How to Run

git clone https://github.com/nandiniranjansinha/RAG-Powered-Document-Q-A-System.git  
cd RAG-Powered-Document-Q-A-System  
pip install -r requirements.txt  
streamlit run app.py  

---

## 🔐 Setup API Keys

Create a file named `secret_api_keys.py`:

groq_api_key = "your_groq_api_key"  
huggingface_api_key = "your_huggingface_api_key"  

---

## 💡 Use Cases

- Academic document Q&A  
- Notes summarization  
- Extracting key information from reports  
- Quick document understanding  

---

## ⚠️ Limitations

- PDF text extraction may fail for scanned documents  
- Uses in-memory FAISS (no persistence)  
- Context length is limited before truncation  

---


## 📌 Summary

This project demonstrates an end-to-end RAG pipeline including document processing, vector search, LLM-based answer generation, and system robustness through fallback handling.
