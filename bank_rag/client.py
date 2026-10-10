"""Create the API client only when an API-backed operation is requested."""

import os
from functools import lru_cache

from openai import OpenAI

from bank_rag import config  # Load the project-local .env before reading settings.


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    if not os.getenv("OPENAI_API_KEY", "").strip():
        raise RuntimeError("Set OPENAI_API_KEY in your environment or local .env file.")
    return OpenAI()
