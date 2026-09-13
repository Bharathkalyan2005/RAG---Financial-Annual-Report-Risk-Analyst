"""
generation/client.py — Robust OpenRouter API Client with Fallback & Retries
===========================================================================

Features:
  - Calls https://openrouter.ai/api/v1/chat/completions
  - Reads OPENROUTER_API_KEY, OPENROUTER_MODEL, OPENROUTER_FALLBACK_MODEL
  - 3-attempt exponential backoff on 429 (rate-limit) and 5xx (server errors)
  - Automatic fallback model invocation if primary model exhausts retries
  - Custom OpenRouterError exception on complete failure
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

log = logging.getLogger("generation.client")

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(Exception):
    """Raised when both primary and fallback OpenRouter models fail."""
    pass


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        primary_model: str | None = None,
        fallback_model: str | None = None,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        timeout: int = 60,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError("OPENROUTER_API_KEY is not set in environment or .env file.")

        self.primary_model = primary_model or os.getenv("OPENROUTER_MODEL", "inclusionai/ling-3.0-flash-fin:free")
        self.fallback_model = fallback_model or os.getenv("OPENROUTER_FALLBACK_MODEL", "liquid/lfm-2.5-2.6b:free")
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.timeout = timeout

    def _call_model(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> str:
        """Call OpenRouter with exponential backoff on 429 and 5xx."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/fin-risk-analyst",
            "X-Title": "Financial Annual Report Risk Analyst",
        }
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }


        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                log.info("Sending request to %s (attempt %d/%d)", model, attempt, self.max_retries)
                response = requests.post(
                    OPENROUTER_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    data = response.json()
                    choices = data.get("choices", [])
                    if choices:
                        msg = choices[0].get("message", {})
                        content = msg.get("content") or msg.get("reasoning") or choices[0].get("text") or ""
                        if content:
                            return content.strip()
                    log.warning("Model %s returned empty message content. Retrying...", model)

                if response.status_code in (429, 500, 502, 503, 504):
                    # If daily free model quota is exhausted, fail fast immediately
                    if response.status_code == 429 and ("free-models-per-day" in response.text or "credits" in response.text):
                        log.warning("Daily free tier quota exhausted for model %s: %s", model, response.text[:150])
                        raise requests.HTTPError(
                            f"Daily quota limit (50 requests/day) reached on OpenRouter free tier: {response.text[:150]}",
                            response=response,
                        )

                    last_err = requests.HTTPError(f"HTTP {response.status_code}: {response.text[:150]}", response=response)
                    sleep_time = self.backoff_factor ** attempt
                    log.warning(
                        "OpenRouter returned %d for model %s. Retrying in %.1fs... (error: %s)",
                        response.status_code,
                        model,
                        sleep_time,
                        response.text[:200],
                    )
                    time.sleep(sleep_time)
                    continue

                # Unrecoverable error (e.g. 400, 401)
                last_err = requests.HTTPError(
                    f"OpenRouter HTTP {response.status_code}: {response.text}",
                    response=response,
                )
                raise last_err


            except (requests.RequestException, requests.Timeout) as e:
                last_err = e
                sleep_time = self.backoff_factor ** attempt
                log.warning("Network error with model %s (%s). Retrying in %.1fs...", model, e, sleep_time)
                time.sleep(sleep_time)

        raise OpenRouterError(f"Model {model} failed after {self.max_retries} attempts. Last error: {last_err}")

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 1000,
    ) -> str:
        """
        Chat completion with automatic fallback model support.
        """
        try:
            return self._call_model(
                model=self.primary_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except OpenRouterError as primary_err:
            log.warning("Primary model %s failed. Switching to fallback model: %s", self.primary_model, self.fallback_model)
            try:
                return self._call_model(
                    model=self.fallback_model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            except Exception as fallback_err:
                raise OpenRouterError(
                    f"Both primary ({self.primary_model}) and fallback ({self.fallback_model}) models failed. "
                    f"Primary error: {primary_err} | Fallback error: {fallback_err}"
                )
