"""Shared synthetic fixtures. No network, no real data, no external database."""
import pandas as pd
import pytest


@pytest.fixture
def sales_df() -> pd.DataFrame:
    """A small deterministic sales table used across the SQL tests."""
    return pd.DataFrame(
        {
            "region": ["east", "west", "east", "north"],
            "product": ["widget", "widget", "gizmo", "gizmo"],
            "units": [10, 20, 5, 7],
            "revenue": [100.5, 200.0, 75.25, 90.0],
            "in_stock": [True, False, True, True],
        }
    )


@pytest.fixture
def awkward_df() -> pd.DataFrame:
    """A table with column names that need quoting."""
    return pd.DataFrame(
        {
            "order date": ["2024-01-01", "2024-02-01"],
            "total $": [10.0, 20.0],
        }
    )


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeChoice:
    def __init__(self, content):
        self.message = FakeMessage(content)


class FakeCompletion:
    def __init__(self, content):
        self.choices = [FakeChoice(content)]


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self._content = content
        self._error = error
        self.last_kwargs = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        if self._error is not None:
            raise self._error
        return FakeCompletion(self._content)


class FakeChat:
    def __init__(self, completions):
        self.completions = completions


class FakeClient:
    """Stands in for ``huggingface_hub.InferenceClient`` in tests."""

    def __init__(self, content=None, error=None):
        self.completions = FakeCompletions(content=content, error=error)
        self.chat = FakeChat(self.completions)


@pytest.fixture
def fake_client():
    """Factory returning a client that replies with the given text."""
    return FakeClient
