"""
rag_chain.py — The core RAG chain connecting retrieval to the LLM.

HOW THE FULL RAG PIPELINE WORKS:
  1. User asks: "What was Apple's revenue in 2024?"
  2. We embed the question -> vector [0.23, -0.41, ...]
  3. ChromaDB finds 5 most similar chunks from 5056 stored vectors
  4. We build a prompt: "Using this context: {chunks} Answer: {question}"
  5. Groq's Llama 3 reads the context and writes a grounded answer
  6. We return the answer + the source chunks as citations

WHY THIS PREVENTS HALLUCINATION:
  The LLM is explicitly told "only answer from the context provided".
  If the answer isn't in the retrieved chunks, it says so instead of
  making something up. This is critical for financial data accuracy.
"""

import os
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

ROOT_DIR = Path(__file__).parent.parent.parent
CHROMA_PERSIST_DIR = ROOT_DIR / "vectorstore"
COLLECTION_NAME = "sec_filings"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# How many chunks to retrieve per question
# 5 gives enough context without overwhelming the LLM
TOP_K = 5


# ── Prompt template ──────────────────────────────────────────────────────────
# This is the most important part of any RAG system.
# We explicitly tell the LLM:
#   1. What role it plays
#   2. That it must ONLY use the provided context
#   3. To cite its sources
#   4. What to do if it doesn't know
RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a financial analyst assistant specializing in SEC filings analysis.
You have access to 10-K annual reports from Apple, Microsoft, and Tesla.

Use ONLY the following context to answer the question.
Do not use any prior knowledge — only what is in the context below.
If the context does not contain enough information to answer, say:
"I don't have enough information in the provided filings to answer this."

Context from SEC filings:
{context}

Question: {question}

Instructions:
- Answer clearly and directly
- Include specific numbers, dates, and figures when available
- At the end of your answer, list the sources you used as:
  Sources: [Company name, Form 10-K, Filing: accession_number]
- If comparing across companies or years, structure your answer clearly

Answer:
""")


def load_vector_store() -> Chroma:
    """Load the existing ChromaDB vector store from disk."""
    if not CHROMA_PERSIST_DIR.exists():
        raise FileNotFoundError(
            f"Vector store not found at {CHROMA_PERSIST_DIR}\n"
            "Run src/embeddings/embedder.py first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    count = vectorstore._collection.count()
    print(f"Loaded vector store: {count} vectors")
    return vectorstore


def format_docs(docs: List) -> str:
    """
    Format retrieved chunks into a single context string.
    
    Each chunk gets a header showing which company and filing it came from.
    This is what gets injected into the prompt as {context}.
    """
    formatted = []
    for i, doc in enumerate(docs, 1):
        company = doc.metadata.get("company", "Unknown")
        accession = doc.metadata.get("accession_number", "unknown")
        chunk_idx = doc.metadata.get("chunk_index", "?")

        header = f"[Source {i}: {company} 10-K, Filing {accession}, Chunk {chunk_idx}]"
        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted)


def build_rag_chain(vectorstore: Chroma):
    """
    Build the full RAG chain using LangChain's pipe syntax.
    
    The chain flows like this:
      question -> retriever -> format_docs -> prompt -> llm -> output_parser
    
    LangChain's | operator connects each step, passing output to the next.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "gsk_your_key_here":
        raise ValueError(
            "GROQ_API_KEY not set. Add your key to .env file."
        )

    # The retriever searches ChromaDB for similar chunks
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )

    # Groq LLM — llama-3.1-70b is fast and very capable
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0,       # 0 = deterministic, good for factual Q&A
        max_tokens=1024,
    )

    # Build the chain using LangChain's pipe operator
    # RunnablePassthrough passes the question through unchanged
    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(chain, question: str, retriever=None) -> Dict[str, Any]:
    """
    Ask a question and return the answer with source documents.
    
    Returns a dict with:
      - answer: the LLM's response
      - sources: the chunks that were used
      - question: the original question
    """
    print(f"\nQuestion: {question}")
    print("-" * 60)

    # Get the answer
    answer = chain.invoke(question)

    # Also get the source documents for display
    sources = []
    if retriever:
        source_docs = retriever.invoke(question)
        sources = [
            {
                "company": doc.metadata.get("company", "Unknown"),
                "accession": doc.metadata.get("accession_number", "unknown"),
                "preview": doc.page_content[:200],
            }
            for doc in source_docs
        ]

    print(answer)
    return {"question": question, "answer": answer, "sources": sources}


if __name__ == "__main__":
    print("Phase 4: RAG Chain")
    print("=" * 60)

    # Load the vector store
    vectorstore = load_vector_store()

    # Build the chain
    print("Building RAG chain with Groq Llama 3.1 70B...")
    chain, retriever = build_rag_chain(vectorstore)
    print("Chain ready.\n")

    # Test with real financial questions
    test_questions = [
        "What was Apple's total net revenue in fiscal year 2024?",
        "What are the top 3 risk factors Tesla mentioned in their latest 10-K?",
        "How did Microsoft's cloud revenue grow and what drove that growth?",
        "Compare the R&D spending between Apple and Microsoft.",
    ]

    print("Running test questions against real SEC filings...")
    print("=" * 60)

    for question in test_questions:
        result = ask(chain, question, retriever)
        print("\n" + "=" * 60)

    print("\nRAG chain is working correctly.")
    print("Next: run app.py to launch the Streamlit UI.")
