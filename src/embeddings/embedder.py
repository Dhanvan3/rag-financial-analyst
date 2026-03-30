"""
embedder.py — Chunks documents and stores embeddings in ChromaDB.

WHAT IS CHUNKING?
  Each 10-K filing is ~200 pages. We can't feed that to an LLM at once.
  We split each document into overlapping chunks of ~1000 characters.
  
  Example of overlap:
    Chunk 1: "Apple's revenue was $394 billion in 2023. This represents..."
    Chunk 2: "This represents a 3% decline from 2022. The company attributed..."
  
  The overlap (200 chars) ensures no sentence is cut off at a boundary.

WHAT ARE EMBEDDINGS?
  An embedding converts text into a list of 384 numbers (a vector).
  Similar meanings = similar vectors = close together in space.
  
  Example:
    "Apple revenue 2023"     -> [0.23, -0.41, 0.87, ...]
    "AAPL annual sales"      -> [0.21, -0.39, 0.85, ...]  <- similar!
    "Tesla battery design"   -> [-0.54, 0.12, -0.33, ...] <- very different
  
  This is how we find relevant chunks without keyword matching.

WHAT IS CHROMADB?
  ChromaDB is a vector database — it stores chunks + their embeddings on disk.
  When a question comes in, we embed the question and ChromaDB finds the
  chunks whose embeddings are closest (cosine similarity).
"""

import json
from pathlib import Path
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
CHROMA_PERSIST_DIR = ROOT_DIR / "vectorstore"

# Chunking settings
# 1000 chars = ~250 tokens, fits well within LLM context
# 200 char overlap = ~2-3 sentences shared between chunks
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ChromaDB collection name
COLLECTION_NAME = "sec_filings"

# The embedding model — runs locally, no API key needed
# all-MiniLM-L6-v2: small (90MB), fast, great for retrieval tasks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_processed_documents() -> List[Document]:
    """Load the processed filings we saved in Phase 2."""
    processed_path = PROCESSED_DATA_DIR / "processed_filings.json"

    if not processed_path.exists():
        raise FileNotFoundError(
            f"No processed documents found at {processed_path}\n"
            "Run src/ingestion/ingest.py first."
        )

    with open(processed_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    documents = [
        Document(
            page_content=item["page_content"],
            metadata=item["metadata"]
        )
        for item in data
    ]

    print(f"Loaded {len(documents)} documents from {processed_path}")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split documents into overlapping chunks.
    
    RecursiveCharacterTextSplitter tries to split on paragraphs first,
    then sentences, then words — so chunks break at natural boundaries
    rather than mid-sentence.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Try to split at these boundaries in order
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    print(f"\nChunking {len(documents)} documents...")
    print(f"  Chunk size: {CHUNK_SIZE} chars")
    print(f"  Overlap: {CHUNK_OVERLAP} chars")

    chunks = []
    for doc in tqdm(documents, desc="Chunking"):
        doc_chunks = splitter.split_documents([doc])

        # Add chunk index to metadata so we know where each chunk came from
        for i, chunk in enumerate(doc_chunks):
            chunk.metadata["chunk_index"] = i
            chunk.metadata["total_chunks"] = len(doc_chunks)

        chunks.extend(doc_chunks)

    print(f"\nCreated {len(chunks)} chunks from {len(documents)} documents")
    print(f"  Average chunks per document: {len(chunks) / len(documents):.1f}")

    # Show breakdown by company
    by_company = {}
    for chunk in chunks:
        company = chunk.metadata.get("company", "Unknown")
        by_company[company] = by_company.get(company, 0) + 1

    print("\nChunks by company:")
    for company, count in sorted(by_company.items()):
        print(f"  {company}: {count} chunks")

    return chunks


def create_vector_store(chunks: List[Document]) -> Chroma:
    """
    Embed all chunks and store them in ChromaDB.
    
    This is the most computationally intensive step — we're converting
    every chunk into a 384-dimensional vector using the local model.
    For 9 filings this takes ~1-2 minutes on CPU.
    """
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    print("(This runs locally — no API key needed)")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # Test the embeddings work
    test_embedding = embeddings.embed_query("test")
    print(f"  Embedding dimension: {len(test_embedding)}")

    print(f"\nEmbedding {len(chunks)} chunks and storing in ChromaDB...")
    print(f"  Storage location: {CHROMA_PERSIST_DIR}")
    print("  This may take 1-3 minutes...")

    # Process in batches to show progress
    # ChromaDB handles persistence automatically
    BATCH_SIZE = 100
    
    # Create vector store with first batch
    first_batch = chunks[:BATCH_SIZE]
    vectorstore = Chroma.from_documents(
        documents=first_batch,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(CHROMA_PERSIST_DIR),
    )

    # Add remaining batches
    remaining = chunks[BATCH_SIZE:]
    for i in tqdm(
        range(0, len(remaining), BATCH_SIZE),
        desc="Embedding batches",
        total=len(remaining) // BATCH_SIZE + 1,
    ):
        batch = remaining[i : i + BATCH_SIZE]
        if batch:
            vectorstore.add_documents(batch)

    print(f"\nVector store created successfully")
    print(f"  Total vectors stored: {vectorstore._collection.count()}")

    return vectorstore


def test_retrieval(vectorstore: Chroma) -> None:
    """
    Run a quick test to verify retrieval works.
    
    This is important — we want to confirm that semantic search
    is actually finding relevant chunks before we build the full chain.
    """
    print("\nTesting retrieval with sample questions...")

    test_questions = [
        "What was Apple's total revenue?",
        "What are Tesla's main risk factors?",
        "How much cash does Microsoft have?",
    ]

    for question in test_questions:
        print(f"\nQ: {question}")
        results = vectorstore.similarity_search(question, k=2)
        for i, doc in enumerate(results):
            company = doc.metadata.get("company", "Unknown")
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"  Result {i+1} [{company}]: {preview}...")


if __name__ == "__main__":
    print("Phase 3: Chunking and Embedding Pipeline")
    print("=" * 50)

    # Step 1: Load processed documents
    documents = load_processed_documents()

    # Step 2: Chunk documents
    chunks = chunk_documents(documents)

    # Step 3: Embed and store in ChromaDB
    vectorstore = create_vector_store(chunks)

    # Step 4: Test it works
    test_retrieval(vectorstore)

    print("\nPhase 3 complete!")
    print("Vector store is ready for the RAG chain.")
    print("Next: run src/chain/rag_chain.py to build the Q&A system.")
