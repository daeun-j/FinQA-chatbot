"""FinQA Document dataclass."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FinQADocument:
    """Represents a single FinQA example (one financial report page).

    Attributes:
        doc_id: Unique identifier (e.g., "V/2008/page_17.pdf-1")
        pre_text: List of text paragraphs before the table
        post_text: List of text paragraphs after the table
        table: Raw table as list of lists (header first)
        table_md: Table rendered as markdown string
        table_linearized: Table as linearized natural language text
        question: The question string (from qa dict)
        gold_program: Gold DSL program (e.g., "subtract(920, 95), divide(#0, 95)")
        gold_answer: Gold execution answer (float)
        gold_evidence: Dict mapping evidence indices to text
        full_text: Combined pre_text + table_linearized + post_text for embedding
    """

    doc_id: str
    pre_text: list[str]
    post_text: list[str]
    table: list[list[str]]
    table_md: str = ""
    table_linearized: str = ""
    question: str = ""
    gold_program: str = ""
    gold_answer: Optional[float] = None
    gold_evidence: dict = field(default_factory=dict)
    full_text: str = ""

    def get_context_for_llm(self) -> str:
        """Format document context for the LLM prompt."""
        parts = []
        if self.pre_text:
            parts.append("### Background Text")
            parts.append("\n".join(self.pre_text))
        if self.table_md:
            parts.append("\n### Financial Table")
            parts.append(self.table_md)
        if self.post_text:
            parts.append("\n### Additional Text")
            parts.append("\n".join(self.post_text))
        return "\n\n".join(parts)

    def get_text_for_embedding(self) -> str:
        """Get combined text representation for embedding."""
        if self.full_text:
            return self.full_text
        parts = []
        parts.extend(self.pre_text)
        if self.table_linearized:
            parts.append(self.table_linearized)
        parts.extend(self.post_text)
        return "\n".join(parts)
