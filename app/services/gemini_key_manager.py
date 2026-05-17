"""
gemini_key_manager.py
---------------------
Thread-safe round-robin API key rotator for the Gemini SDK.

Setup in .env:
    GEMINI_API_KEY=key1
    GEMINI_API_KEY_2=key2
    GEMINI_API_KEY_3=key3
    ...

Any number of keys are supported. Keys are rotated automatically on
429 (quota exceeded) or 403 (forbidden/suspended) errors. If all keys
are exhausted the caller receives a GeminiKeysExhausted exception.
"""

import logging
import os
import threading
from typing import List, Optional

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted, PermissionDenied, InvalidArgument

logger = logging.getLogger(__name__)


class GeminiKeysExhausted(Exception):
    """Raised when every available API key has hit its quota or is invalid."""


class GeminiKeyManager:
    """Round-robin key pool with per-call rotation on quota errors."""

    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("At least one Gemini API key must be provided.")
        self._keys = keys
        self._index = 0
        self._lock = threading.Lock()
        logger.info("GeminiKeyManager initialised with %d key(s).", len(keys))

    @property
    def current_key(self) -> str:
        with self._lock:
            return self._keys[self._index]

    def rotate(self) -> Optional[str]:
        """Advance to the next key. Returns the new key, or None if all exhausted."""
        with self._lock:
            self._index += 1
            if self._index >= len(self._keys):
                self._index = 0  # wrap around for next cycle
                return None  # signals full rotation completed
            return self._keys[self._index]

    def generate_with_rotation(
        self,
        model_name: str,
        system_instruction: str,
        contents: str,
        generation_config,
    ) -> str:
        """
        Call Gemini generate_content, rotating keys automatically on 429/403.
        Returns the response text string.
        Raises GeminiKeysExhausted if every key fails.
        """
        keys_tried = 0
        total_keys = len(self._keys)

        while keys_tried < total_keys:
            api_key = self.current_key
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                )
                response = model.generate_content(
                    contents=contents,
                    generation_config=generation_config,
                )
                if response and response.text:
                    return response.text
                raise ValueError("Empty response from Gemini.")

            except (ResourceExhausted, PermissionDenied, InvalidArgument) as exc:
                keys_tried += 1
                key_preview = api_key[:8] + "..."
                logger.warning(
                    "Key %s failed (%s). Tried %d/%d keys.",
                    key_preview, type(exc).__name__, keys_tried, total_keys,
                )
                next_key = self.rotate()
                if next_key is None and keys_tried >= total_keys:
                    break
            except Exception:
                # Non-quota errors (bad model name, network, etc.) — don't rotate, just raise
                raise

        raise GeminiKeysExhausted(
            f"All {total_keys} Gemini API key(s) have exceeded their quota or are invalid."
        )

    def generate_structured_with_rotation(
        self,
        model_name: str,
        system_instruction: str,
        contents: str,
        generation_config,
    ):
        """
        Same as generate_with_rotation but returns the raw Gemini response object,
        needed when the caller uses response_schema (structured JSON output).
        Rotates keys on 429/403.
        Raises GeminiKeysExhausted if every key fails.
        """
        keys_tried = 0
        total_keys = len(self._keys)

        while keys_tried < total_keys:
            api_key = self.current_key
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(
                    model_name=model_name,
                    system_instruction=system_instruction,
                )
                response = model.generate_content(
                    contents=contents,
                    generation_config=generation_config,
                )
                if response and response.text:
                    return response
                raise ValueError("Empty response from Gemini.")

            except (ResourceExhausted, PermissionDenied, InvalidArgument) as exc:
                keys_tried += 1
                key_preview = api_key[:8] + "..."
                logger.warning(
                    "Key %s [structured] failed (%s). Tried %d/%d keys.",
                    key_preview, type(exc).__name__, keys_tried, total_keys,
                )
                next_key = self.rotate()
                if next_key is None and keys_tried >= total_keys:
                    break
            except Exception:
                raise

        raise GeminiKeysExhausted(
            f"All {total_keys} Gemini API key(s) have exceeded their quota or are invalid."
        )


# ---------------------------------------------------------------------------
# Module-level singleton — imported by parser_service and synthesis_service
# ---------------------------------------------------------------------------
_manager: Optional[GeminiKeyManager] = None
_manager_lock = threading.Lock()


def _collect_keys() -> List[str]:
    """
    Collect API keys from a single comma-separated environment variable.

    .env:
        GEMINI_API_KEYS=key1,key2,key3

    Falls back to GEMINI_API_KEY (single key) for backwards compatibility.
    """
    # Primary: comma-separated list
    raw = os.environ.get("GEMINI_API_KEYS", "").strip()
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            logger.info("Loaded %d Gemini key(s) from GEMINI_API_KEYS.", len(keys))
            return keys

    # Fallback: single key
    single = os.environ.get("GEMINI_API_KEY", "").strip()
    if single:
        logger.info("Loaded 1 Gemini key from GEMINI_API_KEY.")
        return [single]

    return []


def get_key_manager() -> GeminiKeyManager:
    """Return the shared singleton GeminiKeyManager, initialising it on first call."""
    global _manager
    with _manager_lock:
        if _manager is None:
            keys = _collect_keys()
            if not keys:
                raise RuntimeError(
                    "No Gemini API keys found. Set GEMINI_API_KEY (and optionally "
                    "GEMINI_API_KEY_2, GEMINI_API_KEY_3, ...) in your .env file."
                )
            _manager = GeminiKeyManager(keys)
    return _manager