# Ask PDF

A basic PDF Question Answering project built using Retrieval-Augmented Generation (RAG).

## Architecture

PDF → PyMuPDF → Text Chunks → Gemini Embeddings → FAISS → Similarity Search → Gemini → Answer

## Tech Stack

- Python
- Streamlit
- LangChain
- PyMuPDF
- Gemini Embeddings
- FAISS
- Gemini 2.5 Flash

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Open `.env` and add your Gemini API key:

```env
GOOGLE_API_KEY=your_actual_api_key
```

4. Run:

```bash
streamlit run app.py
```

Keep `.env` private and do not commit it to GitHub.

## Current Scope

This version intentionally focuses only on basic RAG:

1. Load PDF
2. Split PDF text into chunks
3. Generate embeddings
4. Store embeddings in FAISS
5. Retrieve the most relevant chunks for a question
6. Send retrieved context to Gemini
7. Generate the answer

Compression, reranking, hybrid search, advanced RAG and evaluation are intentionally excluded for now.
