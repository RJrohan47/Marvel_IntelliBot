"""Gemini chat with an optional Groq-hosted OpenAI model as fallback."""

from __future__ import annotations

import os
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI


def build_chat_models(*, temperature: float = 0, primary_model_name: str | None = None):
    """Build Gemini as primary and Groq as optional fallback when configured."""
    primary = ChatGoogleGenerativeAI(
        model=primary_model_name or os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash"),
        temperature=temperature,
    )
    fallback = None
    if os.getenv("GROQ_API_KEY"):
        from langchain_groq import ChatGroq

        fallback = ChatGroq(
            model=os.getenv("GROQ_CHAT_MODEL", "openai/gpt-oss-20b"),
            temperature=temperature,
        )
    return primary, fallback


def is_provider_fallback_error(error: Exception) -> bool:
    """Recognize provider availability failures that are reasonable to fail over."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    response = getattr(error, "response", None)
    status = status or getattr(response, "status_code", None)
    try:
        status = int(status)
    except (TypeError, ValueError):
        status = None
    if status in {404, 408, 429} or (status is not None and status >= 500):
        return True
    if type(error).__name__.lower() in {
        "resourceexhausted", "toomanyrequests", "connecterror", "timeoutexception", "timeout",
    }:
        return True
    message = str(error).lower()
    if status is None and "404" in message and "not found" in message:
        return True
    return any(term in message for term in (
        "rate limit", "rate_limit", "quota exceeded", "resource exhausted", "too many requests",
        "timed out", "timeout", "connection error", "connection reset", "temporarily unavailable",
    ))


def is_rate_limit_error(error: Exception) -> bool:
    """Backward-compatible name for callers that previously checked provider errors."""
    return is_provider_fallback_error(error)


def invoke_with_fallback(primary: Any, fallback: Any, request: Any):
    """Invoke the primary model and fail over for provider availability errors."""
    try:
        return primary.invoke(request)
    except Exception as error:
        if fallback is None or not is_provider_fallback_error(error):
            raise
        return fallback.invoke(request)
