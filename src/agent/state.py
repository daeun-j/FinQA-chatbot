"""LangGraph agent state definition."""

from typing import Optional, TypedDict, Annotated
from operator import add


class AgentState(TypedDict):
    """State that flows through the LangGraph agent.

    Attributes:
        question: The user's question.
        retrieved_docs: List of retrieved document dicts (context, metadata).
        messages: Accumulated LLM messages (system + few-shot + conversation).
        reasoning: The LLM's latest parsed output (action, reasoning, expression/value).
        calc_result: Result from the calculator tool (if invoked).
        final_answer: The final answer string.
        iteration: Current iteration count (guards against infinite loops).
        error: Error message if something went wrong.
    """
    question: str
    retrieved_docs: list[dict]
    messages: list[dict]
    reasoning: Optional[dict]
    calc_result: Optional[str]
    final_answer: Optional[str]
    iteration: int
    error: Optional[str]
    # Oracle mode only: pre-supplied gold document, bypasses retrieval.
    oracle_doc: Optional[dict]
    # First DSL expression the LLM emitted; used for FinQA program accuracy.
    predicted_program: Optional[str]
    # How many times VERIFY has flagged the answer for revision (capped at 1).
    verify_count: int
