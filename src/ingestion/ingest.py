"""
ingest.py — Extracts clean text from downloaded SEC filings.

What this script does:
  1. Finds all downloaded SEC filing HTML files in data/raw/
  2. Extracts clean text from each one (strips HTML tags, cleans whitespace)
  3. Adds metadata: which company, which ticker, which filing date
  4. Saves processed documents to data/processed/ as JSON

Why do we need this step?
  Raw SEC filings are messy HTML with tables, headers, legal boilerplate,
  and page numbers. We need clean plain text before we can chunk and
  embed it. Garbage in = garbage out for RAG systems.

What is a LangChain Document?
  A Document is a simple object with two fields:
    - page_content: the actual text
    - metadata: a dict with extra info (source, company, date, etc.)
  Metadata is what lets us show citations in answers later.
"""

import json
import re
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DATA_DIR = ROOT_DIR / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Map tickers back to company names for metadata
TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "TSLA": "Tesla",
}


def extract_text_from_html(html_content: str) -> str:
    """
    Extract plain text from SEC HTML filing.
    
    SEC filings use HTML for formatting. We strip all tags and
    clean up the resulting text so it's readable plain text.
    """
    # Remove script and style blocks entirely
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html_content, flags=re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.DOTALL)

    # Replace common block tags with newlines to preserve structure
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<p[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</p>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<tr[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<td[^>]*>", " | ", text, flags=re.IGNORECASE)

    # Strip all remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&nbsp;", " ")
    text = text.replace("&#160;", " ")
    text = text.replace("&ldquo;", '"')
    text = text.replace("&rdquo;", '"')

    # Collapse multiple whitespace/newlines into single ones
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def is_meaningful_text(text: str, min_length: int = 200) -> bool:
    """
    Check if extracted text is worth keeping.
    
    Some SEC filing sections are just tables of numbers or
    exhibit indexes. We skip anything too short to be useful.
    """
    return len(text.strip()) >= min_length


def find_filing_files() -> List[Path]:
    """Find all SEC filing HTML files in the raw data directory."""
    filing_dir = RAW_DATA_DIR / "sec-edgar-filings"
    if not filing_dir.exists():
        print(f"No filings found at {filing_dir}")
        print("Run download_filings.py first.")
        return []

    html_files = []
    for html_file in filing_dir.rglob("*.htm"):
        html_files.append(html_file)
    for html_file in filing_dir.rglob("*.html"):
        html_files.append(html_file)

    return html_files


def extract_metadata_from_path(file_path: Path) -> dict:
    """
    Extract company and filing info from the file path.
    
    SEC EDGAR saves files in a predictable structure:
    sec-edgar-filings/{TICKER}/10-K/{accession-number}/filing-details.htm
    
    We parse this path to get the ticker, which we map to company name.
    """
    parts = file_path.parts

    # Find ticker in path
    ticker = "UNKNOWN"
    for i, part in enumerate(parts):
        if part in TICKER_TO_COMPANY:
            ticker = part
            break

    # Try to extract accession number (filing date is encoded in it)
    # Format: XXXXXXXXXX-YY-ZZZZZZ where YY is the year
    accession = "unknown"
    for part in parts:
        if re.match(r"\d{10}-\d{2}-\d{6}", part):
            accession = part
            break

    return {
        "ticker": ticker,
        "company": TICKER_TO_COMPANY.get(ticker, ticker),
        "form_type": "10-K",
        "accession_number": accession,
        "source": str(file_path),
    }


def process_filings() -> List[Document]:
    """
    Main function: find all filings, extract text, return Documents.
    
    Returns a list of LangChain Document objects ready for chunking.
    """
    html_files = find_filing_files()

    if not html_files:
        return []

    print(f"Found {len(html_files)} filing file(s) to process...")

    documents = []
    skipped = 0

    for file_path in tqdm(html_files, desc="Processing filings"):
        try:
            # Read the raw HTML
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            # Extract clean text
            text = extract_text_from_html(html_content)

            # Skip if not enough real content
            if not is_meaningful_text(text):
                skipped += 1
                continue

            # Build metadata from file path
            metadata = extract_metadata_from_path(file_path)

            # Create a LangChain Document
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)

        except Exception as e:
            print(f"\nError processing {file_path.name}: {e}")
            skipped += 1

    print(f"\nProcessed {len(documents)} documents ({skipped} skipped)")
    return documents


def save_documents(documents: List[Document]) -> None:
    """Save processed documents to JSON for inspection and reuse."""
    output_path = PROCESSED_DATA_DIR / "processed_filings.json"

    serializable = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata,
            "char_count": len(doc.page_content),
        }
        for doc in documents
    ]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(documents)} documents to {output_path}")

    # Print a summary
    print("\nDocument summary:")
    by_company = {}
    for doc in documents:
        company = doc.metadata.get("company", "Unknown")
        by_company[company] = by_company.get(company, 0) + 1

    for company, count in sorted(by_company.items()):
        print(f"  {company}: {count} document(s)")


if __name__ == "__main__":
    print("Starting SEC filing ingestion pipeline...")
    print("=" * 50)

    documents = process_filings()

    if documents:
        save_documents(documents)
        print("\nIngestion complete. Ready for chunking and embedding.")
    else:
        print("\nNo documents processed. Check that filings were downloaded first.")
