"""Split FinQADocuments into retrieval-friendly chunks."""

from src.data_processing.chunk import Chunk
from src.data_processing.document import FinQADocument


def _group_short_paragraphs(paragraphs: list[str], min_chars: int = 120) -> list[str]:
    """Merge consecutive short paragraphs so each chunk has enough signal.

    FinQA pre/post text often contains 1-sentence fragments that, on their
    own, embed poorly (high noise, low specificity). Greedily concatenate
    until each group reaches min_chars.
    """
    grouped: list[str] = []
    buf = ""
    for p in paragraphs:
        p = p.strip()
        if not p:
            continue
        buf = f"{buf} {p}".strip() if buf else p
        if len(buf) >= min_chars:
            grouped.append(buf)
            buf = ""
    if buf:
        if grouped:
            grouped[-1] = f"{grouped[-1]} {buf}"
        else:
            grouped.append(buf)
    return grouped


def chunk_document(doc: FinQADocument, min_chars: int = 120) -> list[Chunk]:
    """Split one FinQADocument into Chunks.

    Strategy: paragraph-level for pre/post text (with short-paragraph merging),
    whole-table-as-one-chunk for the table. FinQA tables are small (~5-20 rows)
    so further row-level splitting hurts more than it helps.
    """
    chunks: list[Chunk] = []

    pre_groups = _group_short_paragraphs(doc.pre_text, min_chars=min_chars)
    for i, text in enumerate(pre_groups):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::pre_{i}",
                parent_doc_id=doc.doc_id,
                text=text,
                chunk_type="pre",
                parent_doc=doc,
            )
        )

    # Table indexing — hybrid: one whole-table chunk for retrieval signal
    # (long, lots of vocabulary for BM25 + dense), plus per-row chunks for
    # evidence hinting (the agent uses these to highlight specific rows in
    # the prompt, helping the LLM focus when tables have many rows).
    if doc.table_linearized:
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::table",
                parent_doc_id=doc.doc_id,
                text=doc.table_linearized,
                chunk_type="table",
                parent_doc=doc,
            )
        )

    if doc.table and len(doc.table) >= 2:
        header_cells = [str(c).strip() for c in doc.table[0]]
        for r_idx, row in enumerate(doc.table[1:], start=1):
            cells = [str(c).strip() for c in row]
            row_label = cells[0] if cells else f"row_{r_idx}"
            pairs = [f"{h}={v}" for h, v in zip(header_cells[1:], cells[1:]) if v]
            # Prepend the doc question hint isn't possible (we don't have one
            # at index time), so we add the row label twice for retrieval weight.
            text = f"{row_label} {row_label} | " + " ; ".join(pairs) if pairs else row_label
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::row_{r_idx}",
                    parent_doc_id=doc.doc_id,
                    text=text,
                    chunk_type="table_row",
                    parent_doc=doc,
                    row_index=r_idx,
                    row_label=row_label,
                )
            )

    post_groups = _group_short_paragraphs(doc.post_text, min_chars=min_chars)
    for i, text in enumerate(post_groups):
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}::post_{i}",
                parent_doc_id=doc.doc_id,
                text=text,
                chunk_type="post",
                parent_doc=doc,
            )
        )

    return chunks


def chunk_documents(docs: list[FinQADocument], min_chars: int = 120) -> list[Chunk]:
    """Chunk a corpus of FinQADocuments."""
    out: list[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, min_chars=min_chars))
    return out
