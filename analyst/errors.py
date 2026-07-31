"""Exceptions raised by the analyst package."""


class AnalystError(Exception):
    """Base exception for every failure raised by this package."""


class ConfigError(AnalystError):
    """Raised when required configuration is missing or invalid."""


class LLMError(AnalystError):
    """Raised when the inference provider cannot produce a usable answer."""


class SQLValidationError(AnalystError):
    """Raised when generated SQL is unsafe or syntactically invalid."""
