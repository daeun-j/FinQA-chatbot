"""Node functions for the LangGraph agent."""

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agent.state import AgentState
from src.agent.prompts import (
    SYSTEM_PROMPT,
    build_few_shot_messages,
    format_user_message,
    FOLLOWUP_PROMPT,
    VERIFY_SYSTEM_PROMPT,
    VERIFY_USER_PROMPT,
)
from src.tools.calculator import execute_program


def inject_oracle_doc(state: AgentState, dynamic_pool=None, num_few_shot: int = 3) -> dict:
    """Oracle-mode replacement for `retrieve`: use the gold document directly."""
    question = state["question"]
    oracle = state["oracle_doc"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if dynamic_pool is not None:
        examples = dynamic_pool.get_examples(question, k=num_few_shot)
        messages.extend(dynamic_pool.format_messages_langgraph(examples))
    else:
        messages.extend(build_few_shot_messages(num_examples=num_few_shot))
    messages.append({
        "role": "user",
        "content": format_user_message(oracle["content"], question),
    })

    return {
        "retrieved_docs": [{"content": oracle["content"], "metadata": oracle}],
        "messages": messages,
        "iteration": 0,
    }


def _format_evidence_hint(evidence_chunks: list[dict], max_hints: int = 5) -> str:
    """Render top evidence chunks as a hint block for the LLM.

    The retriever does chunk-level scoring (per row for tables, per paragraph
    for text). Surfacing the top chunks tells the model where to look inside
    the document — a major boost when the table has many rows.
    """
    if not evidence_chunks:
        return ""
    lines = ["### MOST RELEVANT EVIDENCE (focus your reasoning here)"]
    for i, c in enumerate(evidence_chunks[:max_hints], start=1):
        kind = c.get("chunk_type", "?")
        if kind == "table_row":
            label = c.get("row_label") or "row"
            lines.append(f"  {i}. [table row '{label}'] {c.get('text', '')}")
        else:
            text = c.get("text", "")
            if len(text) > 240:
                text = text[:240] + "..."
            lines.append(f"  {i}. [{kind}] {text}")
    return "\n".join(lines)


def retrieve(state: AgentState, retriever, dynamic_pool=None, num_few_shot: int = 3) -> dict:
    """Retrieve relevant documents for the question."""
    question = state["question"]
    docs = retriever.invoke(question)

    retrieved_docs = []
    for doc in docs:
        retrieved_docs.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
        })

    # Build initial messages: system prompt + few-shot (static or dynamic)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if dynamic_pool is not None:
        examples = dynamic_pool.get_examples(question, k=num_few_shot)
        messages.extend(dynamic_pool.format_messages_langgraph(examples))
    else:
        messages.extend(build_few_shot_messages(num_examples=num_few_shot))

    # Use the top-1 retrieved document as context, then append evidence hints
    # derived from the chunk-level retrieval (specific rows / paragraphs that
    # scored highest for this query).
    if retrieved_docs:
        top = retrieved_docs[0]
        context = top["content"]
        evidence = top["metadata"].get("evidence_chunks", [])
        hint = _format_evidence_hint(evidence)
        if hint:
            context = f"{context}\n\n{hint}"
    else:
        context = "(No relevant documents found)"

    user_msg = format_user_message(context, question)
    messages.append({"role": "user", "content": user_msg})

    return {
        "retrieved_docs": retrieved_docs,
        "messages": messages,
        "iteration": 0,
    }


def reason(state: AgentState, llm) -> dict:
    """Call the LLM to reason about the question and decide on action.

    The LLM either outputs a calculation expression or a direct answer.

    Args:
        state: Current agent state.
        llm: ChatOpenAI instance (pointing at vLLM).

    Returns:
        State update with reasoning dict and incremented iteration.
    """
    messages = state["messages"]

    # Convert to LangChain message objects
    lc_messages = []
    for msg in messages:
        if msg["role"] == "system":
            lc_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            lc_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            lc_messages.append(AIMessage(content=msg["content"]))

    # Call LLM
    response = llm.invoke(lc_messages)
    response_text = response.content.strip()

    # Parse JSON response
    reasoning = _parse_llm_response(response_text)

    # Add the assistant response to messages
    updated_messages = messages + [{"role": "assistant", "content": response_text}]

    update = {
        "reasoning": reasoning,
        "messages": updated_messages,
        "iteration": state["iteration"] + 1,
    }
    # Capture the first emitted DSL expression for program-accuracy eval.
    # The agent may iterate (calculate->reason->...), but FinQA gold programs
    # are single multi-step expressions, so we keep only the first one.
    if (
        state.get("predicted_program") is None
        and reasoning.get("action") == "calculate"
        and reasoning.get("expression")
    ):
        update["predicted_program"] = reasoning["expression"]
    return update


def calculate(state: AgentState) -> dict:
    """Execute the calculator on the expression from the REASON step.

    Args:
        state: Current agent state (must have reasoning with expression).

    Returns:
        State update with calc_result and updated messages for follow-up.
    """
    reasoning = state.get("reasoning", {})
    expression = reasoning.get("expression", "")

    if not expression:
        return {
            "calc_result": "Error: No expression provided",
            "error": "No calculation expression in reasoning output",
        }

    # Get table from retrieved docs for table_* operations
    table = None
    if state.get("retrieved_docs"):
        table = state["retrieved_docs"][0].get("metadata", {}).get("table")

    result, trace = execute_program(expression, table=table)

    if result is None:
        calc_result = f"Error executing '{expression}'. Trace: {trace}"
    else:
        trace_str = "\n".join(
            f"  {s['variable']} = {s['operation']}({', '.join(s['args'])}) = {s['result']}"
            for s in trace
        )
        calc_result = f"Result: {result}\n\nExecution trace:\n{trace_str}"

    # Add follow-up message to ask LLM to interpret the result
    followup = FOLLOWUP_PROMPT.format(result=calc_result)
    updated_messages = state["messages"] + [{"role": "user", "content": followup}]

    return {
        "calc_result": calc_result,
        "messages": updated_messages,
    }


def answer(state: AgentState) -> dict:
    """Format the final answer.

    Args:
        state: Current agent state.

    Returns:
        State update with final_answer.
    """
    reasoning = state.get("reasoning", {})

    if reasoning and reasoning.get("action") == "answer":
        value = reasoning.get("value", "")
        explanation = reasoning.get("reasoning", "")
        final_answer = f"{value}"

        # Try to clean up the answer
        try:
            num = float(str(value).replace(",", "").replace("%", "").strip())
            final_answer = str(round(num, 4))
        except (ValueError, TypeError):
            pass

        return {"final_answer": final_answer}

    # Fallback: extract from calc_result
    calc_result = state.get("calc_result", "")
    if calc_result and "Result:" in calc_result:
        match = re.search(r"Result:\s*([\-\d.]+)", calc_result)
        if match:
            return {"final_answer": match.group(1)}

    return {"final_answer": "Unable to determine answer", "error": "Could not extract answer"}


def verify(state: AgentState, llm) -> dict:
    """Critic node: check the proposed answer against common failure modes.

    If the verifier accepts, the graph terminates. If it requests revision,
    the critique is appended to the message history and control returns to
    REASON for one more pass. Bounded at 1 revision cycle to avoid loops.
    """
    final_answer = state.get("final_answer", "") or ""
    question = state["question"]
    program = state.get("predicted_program") or ""

    docs = state.get("retrieved_docs", [])
    context = docs[0]["content"] if docs else "(no context)"
    # Truncate to keep verify call cheap
    if len(context) > 2000:
        context = context[:2000] + "\n[...truncated...]"

    user_msg = VERIFY_USER_PROMPT.format(
        question=question,
        context=context,
        answer=final_answer,
        program=program or "(direct answer, no program)",
    )

    messages = [
        SystemMessage(content=VERIFY_SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]
    response = llm.invoke(messages)
    parsed = _parse_llm_response(response.content.strip())

    return {
        "reasoning": parsed,  # carries action: accept|revise + critique
        "verify_count": state.get("verify_count", 0) + 1,
    }


def revise(state: AgentState) -> dict:
    """Inject the verifier's critique into the message history for REASON."""
    parsed = state.get("reasoning", {}) or {}
    critique = parsed.get("critique", "Reconsider the answer.")
    revise_msg = (
        f"Your previous answer ({state.get('final_answer', '')}) was flagged: "
        f"{critique}\n\n"
        "Re-examine the document and provide a corrected answer. "
        "Respond in the same JSON format as before."
    )
    updated_messages = state["messages"] + [{"role": "user", "content": revise_msg}]
    return {
        # Reset transient fields so REASON starts fresh on this revision
        "messages": updated_messages,
        "calc_result": None,
        "iteration": 0,
    }


def _parse_llm_response(text: str) -> dict:
    """Parse the LLM's JSON response, with fallback extraction.

    Args:
        text: Raw LLM output text.

    Returns:
        Dict with action, reasoning, and expression/value fields.
    """
    # Try direct JSON parse
    try:
        # Find JSON block (might be wrapped in markdown code block)
        json_match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if "action" in parsed:
                return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: try to extract action and expression/value
    action_match = re.search(r'"action"\s*:\s*"(\w+)"', text)
    expr_match = re.search(r'"expression"\s*:\s*"([^"]+)"', text)
    value_match = re.search(r'"value"\s*:\s*"([^"]*)"', text)
    reason_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)

    if action_match:
        result = {"action": action_match.group(1)}
        if reason_match:
            result["reasoning"] = reason_match.group(1)
        if expr_match:
            result["expression"] = expr_match.group(1)
        if value_match:
            result["value"] = value_match.group(1)
        return result

    # Last resort: treat entire response as a direct answer
    return {"action": "answer", "value": text.strip(), "reasoning": "Could not parse structured output"}
