from typing import Any, Dict, List

import httpx

from config import settings


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


def call_openrouter_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2500,
) -> str:
    """Call OpenRouter chat completions and return the assistant text.

    This module is infrastructure only. Existing FastAPI endpoints do not call
    this function yet.
    """
    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is required to call OpenRouter chat completions")

    if not settings.OPENROUTER_MODEL:
        raise RuntimeError("OPENROUTER_MODEL is required to call OpenRouter chat completions")

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of OpenRouter chat messages")

    payload: Dict[str, Any] = {
        "model": settings.OPENROUTER_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            response = client.post(
                OPENROUTER_CHAT_COMPLETIONS_URL,
                headers=headers,
                json=payload,
            )
    except httpx.TimeoutException as exc:
        raise RuntimeError("OpenRouter request timed out") from exc
    except httpx.RequestError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    if response.status_code >= 400:
        error_detail = response.text
        try:
            error_json = response.json()
            error_detail = str(error_json.get("error") or error_json)
        except ValueError:
            pass
        raise RuntimeError(
            f"OpenRouter request failed with status {response.status_code}: {error_detail}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("OpenRouter returned a non-JSON response") from exc

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("OpenRouter response did not contain assistant message content") from exc

    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("OpenRouter returned empty assistant message content")

    return content.strip()
