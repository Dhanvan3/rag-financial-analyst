# RAG Financial Analyst

> Ask natural language questions about real SEC financial filings and get cited, grounded answers — powered by Retrieval-Augmented Generation.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![LangChain](https://img.shields.io/badge/LangChain-0.2-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5-orange)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-purple)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## What it does

This system lets you upload SEC 10-K/10-Q filings from any public company and ask questions like:

- *"What was Apple's total revenue in 2023 and how did it compare to 2022?"*
- *"What are the main risk factors Tesla identified in their latest annual report?"*
- *"How much cash did Microsoft have on hand, and what did they say about capital allocation?"*

Every answer includes **source citations** — the exact section and page of the filing it came from — so you know the answer is grounded in real data, not hallucinated.

## Architecture

```
SEC Filings (PDF)
      │
      ▼
 Text Extraction (PyMuPDF)
      │
      ▼
 Text Chunking (LangChain RecursiveCharacterTextSplitter)
      │
      ▼
 Embeddings (OpenAI text-embedding-3-small)
      │
      ▼
 Vector Store (ChromaDB — persisted to disk)
      │
      ▼  ◄── User Question (embedded same way)
 Semantic Search (cosine similarity, top-5 chunks)
      │
      ▼
 LLM (GPT-4o-mini) + Retrieved Context
      │
      ▼
 Answer + Source Citations (Streamlit UI)
```

## Tech stack

| Component | Technology | Why |
|---|---|---|
| PDF parsing | PyMuPDF | Fast, accurate text extraction |
| Orchestration | LangChain | Industry standard RAG framework |
| Embeddings | OpenAI text-embedding-3-small | Best price/performance ratio |
| Vector store | ChromaDB | Local, no infra needed, great for portfolios |
| LLM | GPT-4o-mini | GPT-4 quality at low cost |
| UI | Streamlit | Fast to build, looks professional |

## Dataset

Real SEC EDGAR filings downloaded via the `sec-edgar-downloader` library. Includes 10-K annual reports from:
- Apple (AAPL)
- Microsoft (MSFT)
- Tesla (TSLA)

All data is public domain — freely available at [sec.gov](https://www.sec.gov/cgi-bin/browse-edgar).

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/rag-financial-analyst.git
cd rag-financial-analyst

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up your environment variables
cp .env.example .env
# Edit .env and add your OpenAI API key

# 5. Download and index the SEC filings
python -m src.ingestion.download_filings
python -m src.ingestion.ingest

# 6. Launch the app
streamlit run app.py
```

## Project structure

```
rag-financial-analyst/
├── data/
│   ├── raw/          # Original PDF filings (git-ignored)
│   └── processed/    # Extracted text (git-ignored)
├── src/
│   ├── ingestion/    # Download + parse PDFs
│   ├── embeddings/   # Create and manage embeddings
│   ├── retrieval/    # Vector store search
│   └── chain/        # RAG chain (retrieval → LLM → answer)
├── vectorstore/      # ChromaDB on disk (git-ignored)
├── notebooks/        # Exploratory notebooks
├── tests/            # Unit tests
├── app.py            # Streamlit UI entry point
├── .env.example      # Environment variable template
└── requirements.txt  # Python dependencies
```

## Key concepts

**RAG (Retrieval-Augmented Generation)** — Instead of asking the LLM to answer from memory (which causes hallucinations), we first *retrieve* the relevant text from our documents, then pass it as context to the LLM. The LLM only answers based on what we retrieved.

**Embeddings** — Numbers that represent the *meaning* of text. Similar sentences have similar numbers (vectors). This is how we find relevant chunks without doing keyword matching.

**Chunking** — Breaking long documents into smaller pieces so each piece fits within the LLM's context window and retrieval is precise.

## License

MIT — free to use, fork, and build on.
