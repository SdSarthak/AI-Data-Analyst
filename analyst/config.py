"""
Configuration for the analyst package.

Every credential is read from the environment. Nothing is hardcoded, so the
same code runs locally with a ``.env`` file and in a deployment where the
values come from real secrets.
"""
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv

from analyst.errors import ConfigError

load_dotenv()

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
DEFAULT_PROVIDER = "nebius"

# Hugging Face reads several of these itself; accept all of them so a user who
# already exported one of the common names does not have to add another.
_TOKEN_VARS = ("HF_TOKEN", "HUGGINGFACEHUB_API_TOKEN", "HUGGINGFACE_API_KEY")


@dataclass(frozen=True)
class LLMConfig:
    """Settings needed to reach the text-to-SQL model."""

    api_key: Optional[str] = None
    model: str = DEFAULT_MODEL
    provider: str = DEFAULT_PROVIDER
    max_tokens: int = 500
    temperature: float = 0.0
    timeout: float = 60.0

    def require_api_key(self) -> str:
        """Return the API key or explain exactly which variable to set."""
        if not self.api_key:
            raise ConfigError(
                "No Hugging Face token found. Set HF_TOKEN in your environment "
                "or in a .env file (see .env.example)."
            )
        return self.api_key


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def find_api_key() -> Optional[str]:
    """Return the first Hugging Face token present in the environment."""
    for var in _TOKEN_VARS:
        value = os.getenv(var)
        if value and value.strip():
            return value.strip()
    return None


def load_config() -> LLMConfig:
    """Build an :class:`LLMConfig` from the current environment."""
    return LLMConfig(
        api_key=find_api_key(),
        model=os.getenv("HF_MODEL", DEFAULT_MODEL),
        provider=os.getenv("HF_PROVIDER", DEFAULT_PROVIDER),
        max_tokens=_int_env("HF_MAX_TOKENS", 500),
        temperature=_float_env("HF_TEMPERATURE", 0.0),
        timeout=_float_env("HF_TIMEOUT", 60.0),
    )


# Row cap applied to every generated query so a careless SELECT cannot pull a
# multi-million row table into the browser.
MAX_RESULT_ROWS = _int_env("MAX_RESULT_ROWS", 5000)

# Number of example rows shown to the model alongside the schema.
PROMPT_SAMPLE_ROWS = _int_env("PROMPT_SAMPLE_ROWS", 3)
