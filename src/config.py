"""
config.py — Central configuration for the entire project.

All settings are loaded from the .env file using python-dotenv.
Import this module anywhere you need a setting:
    from src.config import settings
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load the .env file from the project root
load_dotenv()

# Project root directory (two levels up from this file)
ROOT_DIR = Path(__file__).parent.parent


class Settings:
    """All project settings in one place."""

    # === API Keys ===
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # === Model names ===
    # text-embedding-3-small is cheap and very good for retrieval
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # gpt-4o-mini gives GPT-4 quality at 10x lower cost
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")

    # === Paths ===
    RAW_DATA_DIR: Path = ROOT_DIR / "data" / "raw"
    PROCESSED_DATA_DIR: Path = ROOT_DIR / "data" / "processed"
    CHROMA_PERSIST_DIR: Path = ROOT_DIR / os.getenv("CHROMA_PERSIST_DIR", "vectorstore")

    # === Chunking strategy ===
    # CHUNK_SIZE: how many characters per chunk
    # Think of it like cutting a book into pages — too small = loses context,
    # too large = retrieves too much noise. 1000 chars (~250 tokens) is a
    # good balance for financial documents.
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))

    # CHUNK_OVERLAP: how many characters the chunks share at their edges
    # This prevents answers being split across chunk boundaries.
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

    # === Retrieval settings ===
    # How many chunks to retrieve per question
    TOP_K_RESULTS: int = 5

    def validate(self) -> None:
        """Check that required settings are present before running."""
        if not self.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and add your key."
            )


# Single shared instance — import this everywhere
settings = Settings()
