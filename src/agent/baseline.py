"""Vanilla single-call RAG baseline (no LangGraph, no tool use).

Used as an ablation to isolate what LangGraph + the calculator tool actually
contribute. Same retriever, same LLM, same context — only the orchestration
differs: one prompt, one LLM call, model does arithmetic in-head and emits
the final number. No JSON, no DSL, no iterative refinement.
"""

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


BASELINE_SYSTEM_PROMPT = """You are a financial analyst. You will be given a financial document (text + table) and a question that requires numerical reasoning.

Think step by step, then output (a) a one-line DSL program of the calculation and (b) the final numeric answer.

OUTPUT FORMAT (the last two lines must be exactly):
Program: <dsl-program>
Answer: <number>

DSL operations: add(a, b), subtract(a, b), multiply(a, b), divide(a, b), exp(a, b), greater(a, b).
Reference earlier steps as #0, #1, ... and constants as const_100, const_1000, etc.
Example program: subtract(920, 825), divide(#0, 825)

NUMBER RULES:
- "Answer:" value must be plain digits — no units, no "%", no "$", no commas.
- For percentage questions, give the answer as a decimal fraction
  (e.g., 0.1434 for 14.34%) and end the program at divide(...) — do NOT
  multiply by const_100.
- If the answer cannot be computed, output "Program: none" and "Answer: 0".
"""


BASELINE_FEW_SHOT = [
    {
        "context": """### Background Text
the company's total revenue was $920 million in 2008 compared to $825 million in 2007.

### Financial Table
| | 2008 | 2007 |
| --- | --- | --- |
| total revenue | $920 | $825 |""",
        "question": "What was the percentage change in total revenue from 2007 to 2008?",
        "response": """Step 1: Find the values. Revenue 2008 = 920, Revenue 2007 = 825.
Step 2: Compute change = 920 - 825 = 95.
Step 3: Divide by base = 95 / 825 = 0.1152.
Program: subtract(920, 825), divide(#0, 825)
Answer: 0.1152""",
    },
    {
        "context": """### Financial Table
| | 2019 | 2018 |
| --- | --- | --- |
| net income | 450 | 380 |
| shares outstanding | 100 | 100 |""",
        "question": "What was the earnings per share in 2019?",
        "response": """Step 1: net income 2019 = 450, shares 2019 = 100.
Step 2: EPS = 450 / 100 = 4.5.
Program: divide(450, 100)
Answer: 4.5""",
    },
    {
        "context": """### Background Text
as of december 31, 2020, the company had 5,234 employees.

### Financial Table
| segment | employees |
| --- | --- |
| north america | 3200 |
| europe | 1500 |
| asia | 534 |""",
        "question": "What percentage of total employees work in north america?",
        "response": """Step 1: NA = 3200, total = 5234.
Step 2: 3200 / 5234 = 0.6113.
Program: divide(3200, 5234)
Answer: 0.6113""",
    },
]


def _format_user_message(context: str, question: str) -> str:
    return f"Document:\n{context}\n\nQuestion: {question}"


def _parse_baseline_answer(text: str) -> str:
    """Extract the value after the final 'Answer:' marker.

    Falls back to the last number-looking token if the marker is missing,
    so a stray reformatting doesn't tank the score.
    """
    import re

    matches = re.findall(r"[Aa]nswer\s*[:=]\s*([\-\d.,eE]+)", text)
    if matches:
        return matches[-1].replace(",", "").strip().rstrip(".")

    # Fallback: last numeric token
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:[eE][+\-]?\d+)?", text)
    if nums:
        return nums[-1]

    return text.strip()


def _parse_baseline_program(text: str) -> str:
    """Extract the DSL program after the final 'Program:' marker.

    Empty string when missing or when the model declared none — both count
    as a program-accuracy miss for fair comparison with the agent.
    """
    import re

    matches = re.findall(r"[Pp]rogram\s*[:=]\s*(.+)", text)
    if not matches:
        return ""
    prog = matches[-1].strip()
    if prog.lower() in {"none", "n/a", ""}:
        return ""
    return prog


def run_baseline_question(
    llm,
    retriever,
    question: str,
    oracle_doc: Optional[dict] = None,
    num_few_shot: int = 3,
    dynamic_pool=None,
) -> dict:
    """Run a single question through the no-LangGraph baseline.

    Args:
        llm: ChatOpenAI client (vLLM-backed).
        retriever: Retriever, or None if oracle_doc is supplied.
        question: User question.
        oracle_doc: If supplied, skip retrieval and use this doc dict (must
                    have a "content" field).
        num_few_shot: How many few-shot examples to include.

    Returns:
        Dict shaped to match LangGraph runner output:
            final_answer, predicted_program (always ""), retrieved_docs.
    """
    if oracle_doc is not None:
        context = oracle_doc["content"]
        retrieved_docs = [{"content": context, "metadata": oracle_doc}]
    else:
        docs = retriever.invoke(question)
        retrieved_docs = [{"content": d.page_content, "metadata": d.metadata} for d in docs]
        context = retrieved_docs[0]["content"] if retrieved_docs else "(No relevant documents found)"

    messages = [SystemMessage(content=BASELINE_SYSTEM_PROMPT)]
    if dynamic_pool is not None:
        examples = dynamic_pool.get_examples(question, k=num_few_shot)
        for user_text, asst_text in dynamic_pool.format_messages_baseline(examples):
            messages.append(HumanMessage(content=user_text))
            messages.append(AIMessage(content=asst_text))
    else:
        for ex in BASELINE_FEW_SHOT[:num_few_shot]:
            messages.append(HumanMessage(content=_format_user_message(ex["context"], ex["question"])))
            messages.append(AIMessage(content=ex["response"]))
    messages.append(HumanMessage(content=_format_user_message(context, question)))

    response = llm.invoke(messages)
    raw = response.content.strip()
    final_answer = _parse_baseline_answer(raw)
    predicted_program = _parse_baseline_program(raw)

    return {
        "final_answer": final_answer,
        "predicted_program": predicted_program,
        "retrieved_docs": retrieved_docs,
        "raw_response": raw,
    }
