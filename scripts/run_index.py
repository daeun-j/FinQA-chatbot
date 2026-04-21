"""CLI entry point to build the FAISS index."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.indexing.build_index import build_index

if __name__ == "__main__":
    build_index()
