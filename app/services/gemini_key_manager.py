"""
gemini_key_manager.py
---------------------
Safe stateless Gemini API key rotator.
Production-safe for FastAPI/cloud deployments.
"""

import logging
import os
from typing import List

import google.generativeai as genai
from google.api_core.exceptions import (
    ResourceExhausted,
    PermissionDenied,
    InvalidArgument,
)

logger = logging.getLogger(__name__)


class GeminiKeysExhausted(Exception):
    """Raised when every available API key fails."""


class GeminiKeyManager:
    def __init__(self, keys: List[str]):
        if not keys:
            raise ValueError("At least one Gemini API key must be provided.")

        self._keys = keys

        logger.info(
            "GeminiKeyManager initialized with %d key(s).",
            len(keys),
        )

    def _try_generate(
        self,
        api_key: str,
        model_name: str,
        system_instruction: str,
        contents: str,
        generation_config,
    ):
        """
        Executes a single Gemini request using one API key.
        """

        # IMPORTANT:
        # Configure globally BEFORE model creation
        genai.configure(api_key=api_key)

        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction,
        )

        response = model.generate_content(
            contents=contents,
            generation_config=generation_config,
        )

        return response

    def generate_with_rotation(
        self,
        model_name: str,
        system_instruction: str,
        contents: str,
        generation_config,
    ) -> str:
        """
        Normal text generation with automatic key rotation.
        """

        total_keys = len(self._keys)

        for idx, api_key in enumerate(self._keys, start=1):
            try:
                logger.info(
                    "Trying Gemini key %d/%d [%s...]",
                    idx,
                    total_keys,
                    api_key[:8],
                )

                response = self._try_generate(
                    api_key=api_key,
                    model_name=model_name,
                    system_instruction=system_instruction,
                    contents=contents,
                    generation_config=generation_config,
                )

                if response and response.text:
                    logger.info(
                        "Gemini request succeeded with key [%s...]",
                        api_key[:8],
                    )
                    return response.text

                raise ValueError("Empty Gemini response.")

            except (
                ResourceExhausted,
                PermissionDenied,
                InvalidArgument,
            ) as exc:

                logger.warning(
                    "Key [%s...] failed with %s",
                    api_key[:8],
                    type(exc).__name__,
                )

                continue

            except Exception as exc:
                logger.exception(
                    "Unexpected Gemini error with key [%s...]: %s",
                    api_key[:8],
                    exc,
                )

                continue

        raise GeminiKeysExhausted(
            f"All {total_keys} Gemini API key(s) failed."
        )

    def generate_structured_with_rotation(
        self,
        model_name: str,
        system_instruction: str,
        contents: str,
        generation_config,
    ):
        """
        Structured JSON generation with automatic key rotation.
        """

        total_keys = len(self._keys)

        for idx, api_key in enumerate(self._keys, start=1):
            try:
                logger.info(
                    "Trying structured Gemini key %d/%d [%s...]",
                    idx,
                    total_keys,
                    api_key[:8],
                )

                response = self._try_generate(
                    api_key=api_key,
                    model_name=model_name,
                    system_instruction=system_instruction,
                    contents=contents,
                    generation_config=generation_config,
                )

                if response and response.text:
                    logger.info(
                        "Structured Gemini request succeeded with key [%s...]",
                        api_key[:8],
                    )
                    return response

                raise ValueError("Empty Gemini response.")

            except (
                ResourceExhausted,
                PermissionDenied,
                InvalidArgument,
            ) as exc:

                logger.warning(
                    "Structured key [%s...] failed with %s",
                    api_key[:8],
                    type(exc).__name__,
                )

                continue

            except Exception as exc:
                logger.exception(
                    "Unexpected structured Gemini error with key [%s...]: %s",
                    api_key[:8],
                    exc,
                )

                continue

        raise GeminiKeysExhausted(
            f"All {total_keys} Gemini API key(s) failed."
        )


# -------------------------------------------------------------------
# Singleton manager
# -------------------------------------------------------------------

_manager = None


def _collect_keys() -> List[str]:
    """
    Loads Gemini API keys from environment.
    Supports:
    - GEMINI_API_KEYS=key1,key2,key3
    - GEMINI_API_KEY=singlekey
    """

    raw = os.environ.get("GEMINI_API_KEYS", "").strip()

    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]

        if keys:
            logger.info(
                "Loaded %d Gemini key(s) from GEMINI_API_KEYS.",
                len(keys),
            )
            return keys

    single = os.environ.get("GEMINI_API_KEY", "").strip()

    if single:
        logger.info("Loaded single Gemini key.")
        return [single]

    return []


def get_key_manager() -> GeminiKeyManager:
    global _manager

    if _manager is None:
        keys = _collect_keys()

        if not keys:
            raise RuntimeError(
                "No Gemini API keys found in environment."
            )

        _manager = GeminiKeyManager(keys)

    return _manager