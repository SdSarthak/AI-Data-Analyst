"""
Configuration settings for the Text-to-SQL LLM application.

Everything is read from the environment (or a local ``.env``). No credential
is ever hardcoded here.
"""
import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
OPENAI_TEMPERATURE = _float_env("OPENAI_TEMPERATURE", 0.0)
OPENAI_MAX_TOKENS = _int_env("OPENAI_MAX_TOKENS", 2000)

# Databricks Configuration
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST")
DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG")
DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA")

# Application Configuration
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Where the user store is persisted. Kept outside the repository by default.
USER_STORE_PATH = os.getenv("USER_STORE_PATH", ".data/users.json")

# AWS Configuration (deployment only)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

# Session Configuration
SESSION_TIMEOUT_MINUTES = _int_env("SESSION_TIMEOUT_MINUTES", 30)
MAX_QUERY_LENGTH = _int_env("MAX_QUERY_LENGTH", 1000)

# SQL Configuration
MAX_ROWS_DISPLAY = _int_env("MAX_ROWS_DISPLAY", 100)
ENABLE_QUERY_OPTIMIZATION = os.getenv("ENABLE_QUERY_OPTIMIZATION", "True").lower() == "true"

# Settings that must be present before the app can talk to anything real.
REQUIRED_SETTINGS = (
    ("OPENAI_API_KEY", OPENAI_API_KEY),
    ("DATABRICKS_HOST", DATABRICKS_HOST),
    ("DATABRICKS_HTTP_PATH", DATABRICKS_HTTP_PATH),
    ("DATABRICKS_TOKEN", DATABRICKS_TOKEN),
    ("DATABRICKS_CATALOG", DATABRICKS_CATALOG),
    ("DATABRICKS_SCHEMA", DATABRICKS_SCHEMA),
)


def missing_settings() -> List[str]:
    """Return the names of required settings that are unset or blank."""
    return [name for name, value in REQUIRED_SETTINGS if not value or not str(value).strip()]
