from __future__ import annotations

import os
from collections.abc import Mapping

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_groq import ChatGroq

from src.prompts.prompt_templates import format_prompt

load_dotenv()

DEFAULT_MODEL = "openai/gpt-oss-120b"


def get_llm(
    model: str | None = None,
    temperature: float = 0.2,
    **kwargs: object,
) -> ChatGroq:
    """Create a Groq chat model using GROQ_API_KEY from the environment."""
    selected_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    api_key = os.getenv("GROQ_API_KEY")

    if not selected_model.strip():
        raise ValueError("Groq model name must not be empty")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. Add it to .env or export it in the shell."
        )
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    return ChatGroq(
        model=selected_model,
        api_key=api_key,
        temperature=temperature,
        **kwargs,
    )


def _message_text(response: BaseMessage) -> str:
    content = response.content
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "")
        for block in content
        if isinstance(block, Mapping) and isinstance(block.get("text"), str)
    )


def generate_answer(
    question: str,
    context: str,
    document_type: str = "generic",
    *,
    llm: ChatGroq | None = None,
    model: str | None = None,
    temperature: float = 0.2,
) -> str:
    """Generate a grounded answer from retrieved context using Groq."""
    prompt = format_prompt(
        question=question,
        context=context,
        document_type=document_type,
    )
    client = llm or get_llm(
        model=model,
        temperature=temperature,
    )
    selected_model = model or os.getenv("GROQ_MODEL", DEFAULT_MODEL)
    try:
        answer = _message_text(client.invoke(prompt)).strip()
    except Exception as error:
        raise RuntimeError(
            f"Groq request failed for model '{selected_model}': {error}"
        ) from error
    if not answer:
        raise RuntimeError("Groq returned an empty answer")
    return answer


__all__ = ["DEFAULT_MODEL", "generate_answer", "get_llm"]
