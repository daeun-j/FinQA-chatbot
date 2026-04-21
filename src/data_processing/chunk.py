"""Chunk dataclass for fine-grained retrieval over FinQA documents."""

from dataclasses import dataclass, field
from typing import Optional

from src.data_processing.document import FinQADocument


@dataclass
class Chunk:
    """A retrievable fragment of a FinQADocument.

    Attributes:
        chunk_id: Unique chunk identifier (e.g., "V/2008/page_17.pdf-1::pre_2").
        parent_doc_id: doc_id of the FinQADocument this chunk came from.
        text: The chunk content used for embedding and BM25.
        chunk_type: One of "pre", "table_row", "post".
        parent_doc: Reference back to the full parent document. Pickle dedups
            by object identity, so multiple chunks of the same doc don't blow up.
        row_index: For table_row chunks, the 1-based row index within the table
            (excluding the header). None for non-table chunks.
        row_label: For table_row chunks, the row's label (first cell). Used
            for the evidence-hint surface in the agent prompt.
    """

    chunk_id: str
    parent_doc_id: str
    text: str
    chunk_type: str
    parent_doc: FinQADocument
    row_index: Optional[int] = None
    row_label: Optional[str] = None
