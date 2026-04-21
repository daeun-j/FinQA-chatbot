"""LangGraph StateGraph definition for the FinQA agent."""

from functools import partial

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.nodes import retrieve, reason, calculate, answer, inject_oracle_doc, verify, revise


def should_calculate(state: AgentState) -> str:
    """Route from REASON: calculate, answer, or safety-stop."""
    if state.get("iteration", 0) >= 3:
        return "answer"
    reasoning = state.get("reasoning", {})
    return "calculate" if reasoning.get("action") == "calculate" else "answer"


def should_revise(state: AgentState) -> str:
    """Route from VERIFY: accept (END) or revise (back to REASON).

    Bounded at one revision cycle so a stubborn verifier can't trap us.
    """
    if state.get("verify_count", 0) >= 2:
        return "accept"
    reasoning = state.get("reasoning", {}) or {}
    return "revise" if reasoning.get("action") == "revise" else "accept"


def build_graph(retriever, llm, use_verify: bool = True, dynamic_pool=None, num_few_shot: int = 3) -> StateGraph:
    """Build and compile the LangGraph agent.

    Architecture (with use_verify=True):
        RETRIEVE -> REASON -> [CALCULATE -> REASON]* -> ANSWER -> VERIFY
                       ^                                            |
                       |________________ revise (max 1) _____________|

    With use_verify=False, the graph terminates at ANSWER. VERIFY can hurt
    accuracy when the critic over-corrects valid answers, so it is exposed
    as a toggle.
    """
    retrieve_node = partial(retrieve, retriever=retriever, dynamic_pool=dynamic_pool, num_few_shot=num_few_shot)
    reason_node = partial(reason, llm=llm)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("reason", reason_node)
    graph.add_node("calculate", calculate)
    graph.add_node("answer", answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "reason")
    graph.add_conditional_edges(
        "reason", should_calculate,
        {"calculate": "calculate", "answer": "answer"},
    )
    graph.add_edge("calculate", "reason")

    if use_verify:
        verify_node = partial(verify, llm=llm)
        graph.add_node("verify", verify_node)
        graph.add_node("revise", revise)
        graph.add_edge("answer", "verify")
        graph.add_conditional_edges(
            "verify", should_revise,
            {"accept": END, "revise": "revise"},
        )
        graph.add_edge("revise", "reason")
    else:
        graph.add_edge("answer", END)

    return graph.compile()


def build_oracle_graph(llm, use_verify: bool = True, dynamic_pool=None, num_few_shot: int = 3) -> StateGraph:
    """Oracle-mode graph: skips retrieval, starts from an injected gold doc."""
    reason_node = partial(reason, llm=llm)
    inject_node = partial(inject_oracle_doc, dynamic_pool=dynamic_pool, num_few_shot=num_few_shot)

    graph = StateGraph(AgentState)
    graph.add_node("inject_oracle", inject_node)
    graph.add_node("reason", reason_node)
    graph.add_node("calculate", calculate)
    graph.add_node("answer", answer)

    graph.set_entry_point("inject_oracle")
    graph.add_edge("inject_oracle", "reason")
    graph.add_conditional_edges(
        "reason", should_calculate,
        {"calculate": "calculate", "answer": "answer"},
    )
    graph.add_edge("calculate", "reason")

    if use_verify:
        verify_node = partial(verify, llm=llm)
        graph.add_node("verify", verify_node)
        graph.add_node("revise", revise)
        graph.add_edge("answer", "verify")
        graph.add_conditional_edges(
            "verify", should_revise,
            {"accept": END, "revise": "revise"},
        )
        graph.add_edge("revise", "reason")
    else:
        graph.add_edge("answer", END)

    return graph.compile()


def run_question(graph, question: str, oracle_doc: dict = None) -> dict:
    """Run a single question through the agent graph.

    Args:
        graph: Compiled LangGraph StateGraph (either retrieval or oracle variant).
        question: The user's financial question.
        oracle_doc: If provided, used instead of retrieval. The graph must have
            been built with build_oracle_graph.

    Returns:
        Final agent state with the answer.
    """
    initial_state = {
        "question": question,
        "retrieved_docs": [],
        "messages": [],
        "reasoning": None,
        "calc_result": None,
        "final_answer": None,
        "iteration": 0,
        "error": None,
        "oracle_doc": oracle_doc,
        "predicted_program": None,
        "verify_count": 0,
    }

    result = graph.invoke(initial_state)
    return result
