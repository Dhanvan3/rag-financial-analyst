"""
download_filings.py — Downloads real SEC 10-K filings from EDGAR.

What is a 10-K?
  A 10-K is the annual report every public US company must file with the SEC.
  It contains: revenue, expenses, risk factors, business outlook, and more.
  These are the documents financial analysts read to understand a company.

What this script does:
  1. Connects to SEC EDGAR (the public database at sec.gov)
  2. Downloads 10-K filings for Apple, Microsoft, and Tesla
  3. Saves them as HTML files in data/raw/
  
Why HTML and not PDF?
  SEC EDGAR stores filings as HTML. We parse them directly —
  it's actually cleaner than PDF because there's no layout noise.
"""

import os
from pathlib import Path
from sec_edgar_downloader import Downloader

# Project root — two levels up from this file
ROOT_DIR = Path(__file__).parent.parent.parent
RAW_DATA_DIR = ROOT_DIR / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Companies we're downloading filings for
# Format: "Company Name" -> "Stock ticker"
COMPANIES = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
}

# How many years of 10-K filings to download per company
# 3 years gives us rich data without being too slow
NUM_FILINGS = 3


def download_filings():
    """Download 10-K filings for all companies."""

    print("Initializing SEC EDGAR downloader...")
    # SEC requires a company name and email for their rate limiter
    # This is just identification — no account needed
    # We change into the raw data dir so files are saved there
    import os
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.chdir(RAW_DATA_DIR)
    dl = Downloader(
        company_name="RAG Financial Analyst",
        email_address="research@ragfinancial.com",
    )

    for company_name, ticker in COMPANIES.items():
        print(f"\nDownloading {company_name} ({ticker}) 10-K filings...")
        try:
            dl.get(
                form="10-K",
                ticker_or_cik=ticker,
                limit=NUM_FILINGS,
                download_details=True,
            )
            print(f"  Downloaded {NUM_FILINGS} filings for {ticker}")
        except Exception as e:
            print(f"  Error downloading {ticker}: {e}")

    print("\nAll downloads complete.")
    print(f"Files saved to: {RAW_DATA_DIR}")
    _show_downloaded_files()


def _show_downloaded_files():
    """Print a summary of what was downloaded."""
    print("\nDownloaded files:")
    for ticker in COMPANIES.values():
        ticker_dir = RAW_DATA_DIR / "sec-edgar-filings" / ticker / "10-K"
        if ticker_dir.exists():
            filings = list(ticker_dir.iterdir())
            print(f"  {ticker}: {len(filings)} filing(s)")
            for filing in sorted(filings)[:3]:
                print(f"    - {filing.name}")
        else:
            print(f"  {ticker}: no files found")


if __name__ == "__main__":
    download_filings()