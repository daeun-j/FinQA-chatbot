"""LLM client factory for vLLM-served models via OpenAI-compatible API."""

from langchain_openai import ChatOpenAI


def create_llm(
    base_url: str = "http://localhost:8000/v1",
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    temperature: float = 0.1,
    max_tokens: int = 1024,
) -> ChatOpenAI:
    """Create a ChatOpenAI client pointing at a vLLM server.

    vLLM serves models via an OpenAI-compatible API, so we can use
    LangChain's ChatOpenAI directly. No API key is needed for local
    vLLM servers.

    Args:
        base_url: URL of the vLLM server.
        model_name: Model identifier (must match what vLLM is serving).
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        Configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        base_url=base_url,
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key="not-needed",  # vLLM doesn't require an API key by default
    )
