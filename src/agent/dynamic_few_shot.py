"""Dynamic few-shot prompting using FinQA train gold programs.

Instead of static hand-written few-shot examples, retrieve the top-K
train questions most similar to the user's question and inject their
(evidence, question, gold_program) as the few-shot examples.

This is in-context learning with retrieved gold demonstrations — the same
spirit as the FinQA paper's supervised generator (which trains on gold
programs), but applied at inference time so no GPU training is needed.
"""

import json
from dataclasses import dataclass
from typing import Optional

import numpy as np

from src.data_processing.document import FinQADocument
from src.indexing.embedder import Embedder


@dataclass
class FewShotExample:
    question: str
    evidence: str  # gold_inds joined, or table_linearized as fallback
    program: str
    answer: str


class DynamicFewShotPool:
    """In-memory pool of (question, gold_program) examples from FinQA train."""

    def __init__(self, docs: list[FinQADocument], embedder: Embedder):
        self.embedder = embedder
        self.examples: list[FewShotExample] = []
        question_texts: list[str] = []

        for doc in docs:
            if not doc.question or not doc.gold_program:
                continue
            ev_parts = [
                v.strip() for v in doc.gold_evidence.values()
                if isinstance(v, str) and v.strip()
            ]
            evidence = " ".join(ev_parts) if ev_parts else doc.table_linearized
            if not evidence:
                continue
            self.examples.append(FewShotExample(
                question=doc.question,
                evidence=evidence,
                program=doc.gold_program,
                answer=str(doc.gold_answer) if doc.gold_answer is not None else "",
            ))
            question_texts.append(doc.question)

        # Embed pool questions WITHOUT the BGE query prefix so they live on the
        # "passage" side of the asymmetric retrieval setup. Test queries get
        # the prefix at lookup time.
        self.embeddings = embedder.embed(question_texts) if question_texts else np.zeros((0, embedder.dimension), dtype=np.float32)
        print(f"[dynamic few-shot] pool size: {len(self.examples)}")

    def get_examples(self, query: str, k: int = 3) -> list[FewShotExample]:
        if not self.examples:
            return []
        q_emb = self.embedder.embed_query(query)
        sims = self.embeddings @ q_emb
        top_idx = np.argsort(-sims)[:k]
        return [self.examples[int(i)] for i in top_idx]

    def format_messages_langgraph(self, examples: list[FewShotExample]) -> list[dict]:
        """Format examples as LangGraph-style (JSON action) message pairs."""
        messages: list[dict] = []
        for ex in examples:
            user = f"Document:\n{_truncate(ex.evidence, 600)}\n\nQuestion: {ex.question}"
            assistant = json.dumps({
                "action": "calculate",
                "reasoning": "Apply the calculation indicated by the question.",
                "expression": ex.program,
            }, indent=2)
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        return messages

    def format_messages_baseline(self, examples: list[FewShotExample]) -> list[tuple[str, str]]:
        """Format examples as baseline-style (Program:/Answer:) pairs.

        Returns list of (user_text, assistant_text) tuples. Caller wraps in
        the appropriate LangChain message types.
        """
        out: list[tuple[str, str]] = []
        for ex in examples:
            user = f"Document:\n{_truncate(ex.evidence, 600)}\n\nQuestion: {ex.question}"
            assistant = (
                f"Identifying the relevant numbers and applying the formula.\n"
                f"Program: {ex.program}\n"
                f"Answer: {ex.answer}"
            )
            out.append((user, assistant))
        return out


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_pool_from_config(config: dict) -> Optional[DynamicFewShotPool]:
    """Convenience: load train docs + embedder from config and build a pool."""
    import os
    from src.data_processing.loader import load_finqa_file

    train_path = os.path.join(config["data"]["raw_dir"], config["data"]["train_file"])
    if not os.path.exists(train_path):
        print(f"[dynamic few-shot] train file not found: {train_path}")
        return None
    train_docs = load_finqa_file(train_path)
    embedder = Embedder(model_name=config["embedding"]["model_name"])
    return DynamicFewShotPool(train_docs, embedder)
