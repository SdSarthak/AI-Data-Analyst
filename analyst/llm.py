"""
Natural language to SQL generation.

``generate_sql`` builds the prompt and talks to the provider;
``clean_generated_sql`` turns whatever the model returns into something a
database will accept. They are separate so the cleaning rules, which is where
the real complexity lives, can be tested without a network call.
"""
import re
from typing import Any, Optional

import pandas as pd

from analyst.config import PROMPT_SAMPLE_ROWS, LLMConfig, load_config
from analyst.errors import LLMError
from analyst.schema import DEFAULT_TABLE, build_schema_ddl, sample_rows_markdown

PROMPT_TEMPLATE = """You are an expert SQLite analyst. Translate the question \
into a single SQL query.

The data lives in one table named "{table}" with this schema:

{schema}

{samples}Rules:
- Return a single SELECT statement and nothing else.
- Use only the columns listed above, quoted with double quotes.
- Use SQLite syntax. Do not use vendor specific functions.
- Do not add explanations, comments or markdown fences.

Question: {question}

SQL:"""

# Placeholder table names models reach for when they ignore the schema.
_TABLE_PLACEHOLDERS = (
    "table_name",
    "your_table",
    "your_table_name",
    "your_data",
    "your_data_name",
    "my_table",
    "tablename",
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n?|```", re.MULTILINE)
_QUERY_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def build_prompt(
    question: str,
    df: Optional[pd.DataFrame] = None,
    schema: Optional[str] = None,
    table_name: str = DEFAULT_TABLE,
    sample_rows: int = PROMPT_SAMPLE_ROWS,
) -> str:
    """
    Assemble the prompt sent to the model.

    Args:
        question: The user's question in plain English.
        df: DataFrame used to derive schema and example rows. Optional when
            ``schema`` is given directly.
        schema: Pre-rendered schema text. Overrides the one derived from ``df``.
        table_name: Table name referenced in the prompt.
        sample_rows: How many example rows to include.

    Returns:
        The full prompt string.
    """
    if schema is None:
        if df is None:
            raise ValueError("build_prompt needs either a DataFrame or a schema string")
        schema = build_schema_ddl(df, table_name=table_name)

    samples = ""
    if df is not None and sample_rows > 0:
        rendered = sample_rows_markdown(df, limit=sample_rows)
        if rendered:
            samples = f"Example rows:\n\n{rendered}\n\n"

    return PROMPT_TEMPLATE.format(
        table=table_name,
        schema=schema,
        samples=samples,
        question=question.strip(),
    )


def clean_generated_sql(raw: Optional[str], table_name: str = DEFAULT_TABLE) -> Optional[str]:
    """
    Extract a runnable SQL statement from a raw model response.

    Handles reasoning models that emit ``<think>`` blocks, markdown fences,
    leading prose and placeholder table names.

    Args:
        raw: The model's response.
        table_name: Name that placeholder table references are rewritten to.

    Returns:
        The cleaned query, or ``None`` if no statement could be found.
    """
    if not raw or not isinstance(raw, str):
        return None

    text = _THINK_BLOCK.sub("", raw)
    # Complete blocks are gone by now. A leftover closing tag means the model
    # started reasoning without an opening tag, so drop everything before it.
    # A leftover opening tag means the response was cut off mid-thought.
    if "</think>" in text.lower():
        text = _UNCLOSED_THINK.sub("", text)
    if "<think>" in text.lower():
        return None

    text = _CODE_FENCE.sub("", text)

    match = _QUERY_START.search(text)
    if not match:
        return None
    text = text[match.start():]

    # Reasoning models sometimes append commentary after the statement. When a
    # semicolon terminates the query, everything after it is prose.
    semicolon = text.find(";")
    if semicolon != -1:
        text = text[:semicolon]

    for placeholder in _TABLE_PLACEHOLDERS:
        text = re.sub(rf"\b{placeholder}\b", table_name, text, flags=re.IGNORECASE)

    # "COUNT Row" and "COUNT ROWS" show up when the model paraphrases COUNT(*).
    text = re.sub(r"\bCOUNT\s+ROWS?\b", "COUNT(*)", text, flags=re.IGNORECASE)

    # Collapse blank lines left behind by the removals above.
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text or None


def build_client(config: Optional[LLMConfig] = None) -> Any:
    """
    Create a Hugging Face ``InferenceClient`` for the configured provider.

    Imported lazily so the rest of the package stays importable without the
    ``huggingface_hub`` dependency installed.
    """
    config = config or load_config()
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise LLMError(
            "huggingface_hub is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        return InferenceClient(
            provider=config.provider,
            api_key=config.require_api_key(),
            timeout=config.timeout,
        )
    except Exception as exc:
        raise LLMError(f"Failed to initialise the inference client: {exc}") from exc


def generate_sql(
    question: str,
    df: Optional[pd.DataFrame] = None,
    schema: Optional[str] = None,
    client: Optional[Any] = None,
    config: Optional[LLMConfig] = None,
    table_name: str = DEFAULT_TABLE,
) -> str:
    """
    Translate a natural language question into SQL.

    Args:
        question: Question in plain English.
        df: Dataset the question is about, used for schema and sample rows.
        schema: Optional pre-rendered schema, overriding ``df``.
        client: Pre-built inference client. Injected by tests.
        config: Model settings. Defaults to the environment configuration.
        table_name: Table name used in the prompt.

    Returns:
        A cleaned SQL statement.

    Raises:
        LLMError: If the request fails or no SQL can be extracted.
    """
    if not question or not question.strip():
        raise LLMError("Ask a question before generating SQL.")

    config = config or load_config()
    client = client or build_client(config)
    prompt = build_prompt(question, df=df, schema=schema, table_name=table_name)

    try:
        completion = client.chat.completions.create(
            model=config.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=config.max_tokens,
            temperature=config.temperature,
        )
    except Exception as exc:
        raise LLMError(f"Inference request failed: {exc}") from exc

    raw = _extract_message(completion)
    sql = clean_generated_sql(raw, table_name=table_name)
    if not sql:
        raise LLMError(
            "The model did not return a SQL query. Try rephrasing the question."
        )
    return sql


def _extract_message(completion: Any) -> Optional[str]:
    """Pull the assistant text out of a chat completion response."""
    try:
        choices = completion.choices
    except AttributeError:
        raise LLMError("Unexpected response from the inference provider.")

    if not choices:
        raise LLMError("The inference provider returned no choices.")

    content = choices[0].message.content
    return content.strip() if isinstance(content, str) else None
