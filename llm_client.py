from typing import Any, Dict, List, Optional

import httpx
import time

from config import settings


OPENROUTER_CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_GENERATE_CONTENT_URL = "https://generativelanguage.googleapis.com/v1beta/{model_path}:generateContent"
GEMINI_RETRY_STATUS_CODES = {429, 500, 502, 503, 504}
GEMINI_MAX_RETRIES = 2
GEMINI_RETRY_BASE_DELAY_SECONDS = 1.0


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


def _message_text(message: Dict[str, str]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return str(content)


def _messages_to_gemini_payload(
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    response_schema: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    system_parts = []
    contents = []

    for message in messages:
        role = (message.get("role") or "user").strip().lower()
        text = _message_text(message).strip()
        if not text:
            continue

        if role == "system":
            system_parts.append(text)
            continue

        gemini_role = "model" if role == "assistant" else "user"
        contents.append({
            "role": gemini_role,
            "parts": [{"text": text}],
        })

    if not contents:
        contents.append({
            "role": "user",
            "parts": [{"text": "Return JSON only."}],
        })

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
        },
    }

    if response_schema:
        payload["generationConfig"]["responseSchema"] = response_schema

    if system_parts:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_parts)}],
        }

    return payload


def _gemini_error_detail(response: httpx.Response) -> str:
    error_detail = response.text
    try:
        error_json = response.json()
        error_detail = str(error_json.get("error") or error_json)
    except ValueError:
        pass

    if settings.GEMINI_API_KEY:
        error_detail = error_detail.replace(settings.GEMINI_API_KEY, "[redacted]")

    return error_detail


def call_gemini_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2500,
    response_schema: Optional[Dict[str, Any]] = None,
) -> str:
    """Call Gemini generateContent and return assistant text."""
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is required to call Gemini generation")

    if not settings.GEMINI_GENERATION_MODEL:
        raise RuntimeError("GEMINI_GENERATION_MODEL is required to call Gemini generation")

    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of chat messages")

    model = settings.GEMINI_GENERATION_MODEL.strip()
    model_path = model if model.startswith("models/") else f"models/{model}"
    url = GEMINI_GENERATE_CONTENT_URL.format(model_path=model_path)
    payload = _messages_to_gemini_payload(
        messages,
        temperature,
        max_tokens,
        response_schema=response_schema,
    )
    if model.lower().startswith("gemini-2.5"):
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": 0}

    attempts = GEMINI_MAX_RETRIES + 1
    last_response: Optional[httpx.Response] = None

    with httpx.Client(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
        for attempt_index in range(attempts):
            if attempt_index > 0:
                delay = GEMINI_RETRY_BASE_DELAY_SECONDS * (2 ** (attempt_index - 1))
                time.sleep(delay)

            try:
                response = client.post(
                    url,
                    params={"key": settings.GEMINI_API_KEY},
                    headers={"Content-Type": "application/json"},
                    json=payload,
                )
            except httpx.TimeoutException as exc:
                raise RuntimeError("Gemini request timed out") from exc
            except httpx.RequestError as exc:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc

            last_response = response
            if (
                response.status_code in GEMINI_RETRY_STATUS_CODES
                and attempt_index < GEMINI_MAX_RETRIES
            ):
                continue
            break

    response = last_response
    if response is None:
        raise RuntimeError("Gemini request failed before receiving a response")

    if response.status_code >= 400:
        error_detail = _gemini_error_detail(response)
        retry_note = ""
        if response.status_code in GEMINI_RETRY_STATUS_CODES:
            retry_note = f" after {attempts} attempts"
        raise RuntimeError(
            f"Gemini request failed with status {response.status_code}{retry_note}: {error_detail}"
        )

    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Gemini returned a non-JSON response") from exc

    try:
        parts = data["candidates"][0]["content"]["parts"]
    except (KeyError, IndexError, TypeError) as exc:
        finish_reason = ""
        try:
            finish_reason = data["candidates"][0].get("finishReason", "")
        except (KeyError, IndexError, TypeError, AttributeError):
            pass
        detail = f" finishReason={finish_reason}" if finish_reason else ""
        raise RuntimeError(f"Gemini response did not contain assistant text{detail}") from exc

    content = "\n".join(
        part.get("text", "")
        for part in parts
        if isinstance(part, dict) and part.get("text")
    ).strip()

    if not content:
        raise RuntimeError("Gemini returned empty assistant text")

    return content


def call_llm_chat(
    messages: List[Dict[str, str]],
    temperature: float = 0.2,
    max_tokens: int = 2500,
    response_schema: Optional[Dict[str, Any]] = None,
) -> str:
    provider = (settings.LLM_PROVIDER or "gemini").strip().lower()
    if provider == "openrouter":
        # OpenRouter providers are kept prompt-based for now; schema is ignored
        # safely to preserve existing provider behavior.
        return call_openrouter_chat(messages, temperature=temperature, max_tokens=max_tokens)
    if provider == "gemini":
        return call_gemini_chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_schema=response_schema,
        )
    raise RuntimeError(f"Unsupported LLM_PROVIDER '{provider}'")
