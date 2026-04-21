"""Load FinQA JSON files into FinQADocument objects."""

import json
import os
from typing import Optional

from src.data_processing.document import FinQADocument
from src.data_processing.table_parser import table_to_markdown, table_to_linearized


def load_finqa_file(filepath: str) -> list[FinQADocument]:
    """Load a single FinQA JSON file into a list of FinQADocuments.

    Args:
        filepath: Path to train.json, dev.json, or test.json.

    Returns:
        List of FinQADocument objects.
    """
    with open(filepath, "r") as f:
        raw_data = json.load(f)

    documents = []
    for item in raw_data:
        doc_id = item.get("id", "unknown")
        pre_text = item.get("pre_text", [])
        post_text = item.get("post_text", [])
        table = item.get("table", [])

        # Parse table into both formats
        table_md = table_to_markdown(table)
        table_lin = table_to_linearized(table)

        # Build full text for embedding
        full_text_parts = pre_text + [table_lin] + post_text
        full_text = "\n".join(full_text_parts)

        # Extract QA fields
        qa = item.get("qa", {})
        question = qa.get("question", "")
        gold_program = qa.get("program", "")
        gold_answer = qa.get("exe_ans", None)
        gold_evidence = qa.get("gold_inds", {})

        # Convert gold_answer to float if possible
        if gold_answer is not None:
            try:
                gold_answer = float(gold_answer)
            except (ValueError, TypeError):
                gold_answer = None

        doc = FinQADocument(
            doc_id=doc_id,
            pre_text=pre_text,
            post_text=post_text,
            table=table,
            table_md=table_md,
            table_linearized=table_lin,
            question=question,
            gold_program=gold_program,
            gold_answer=gold_answer,
            gold_evidence=gold_evidence,
            full_text=full_text,
        )
        documents.append(doc)

    return documents


def load_all_splits(
    data_dir: str,
    splits: Optional[list[str]] = None,
) -> dict[str, list[FinQADocument]]:
    """Load multiple dataset splits.

    Args:
        data_dir: Directory containing train.json, dev.json, test.json.
        splits: Which splits to load. Defaults to all three.

    Returns:
        Dict mapping split name to list of documents.
    """
    if splits is None:
        splits = ["train", "dev", "test"]

    result = {}
    for split in splits:
        filepath = os.path.join(data_dir, f"{split}.json")
        if os.path.exists(filepath):
            result[split] = load_finqa_file(filepath)
            print(f"[loaded] {split}: {len(result[split])} documents")
        else:
            print(f"[warning] {filepath} not found, skipping")

    return result
