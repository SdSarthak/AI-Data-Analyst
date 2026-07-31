"""
LLM integration module for Text-to-SQL generation.

Prompt construction and response cleaning are plain functions with no
LangChain import, so they can be unit tested without the provider packages
installed. Only :class:`TextToSQLLLM` touches the network.
"""
import re
from typing import Any, Optional

from config.settings import (
    OPENAI_API_KEY,
    OPENAI_MAX_TOKENS,
    OPENAI_MODEL,
    OPENAI_TEMPERATURE,
)
from utils.errors import LLMError
from utils.logger import setup_logger

logger = setup_logger(__name__)

SQL_PROMPT = """You are an expert SQL developer working against a Databricks \
SQL warehouse. Convert the following natural language query into a valid SQL query.

Database Schema:
{schema}

Table Definitions:
{tables}

User Query: {query}

Generate only the SQL query, without any explanation or markdown formatting. \
Ensure the query is:
1. Valid and executable on Databricks SQL
2. Read only: a single SELECT statement, or a WITH clause followed by SELECT
3. Using proper JOIN syntax where needed
4. Including appropriate WHERE clauses
5. Using CTEs for complex queries when appropriate

SQL Query:"""

EXPLAIN_PROMPT = """Provide a clear and concise explanation of what this SQL \
query does:

{sql}

Explanation:"""

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```[a-zA-Z]*\n?|```", re.MULTILINE)
# A bare \bWITH\b also matches the English word, so a refusal such as
# "I cannot help with that" was being treated as a CTE. Require the shape of a
# real common table expression instead.
_QUERY_START = re.compile(
    r"\bSELECT\b|\bWITH\b\s+(?:RECURSIVE\s+)?[\w\"`\[\]]+\s*(?:\([^)]*\))?\s+AS\s*\(",
    re.IGNORECASE,
)


def build_sql_prompt(
    natural_language_query: str,
    schema_context: str,
    table_definitions: str,
) -> str:
    """Render the text-to-SQL prompt."""
    return SQL_PROMPT.format(
        schema=schema_context,
        tables=table_definitions,
        query=natural_language_query.strip(),
    )


def clean_sql_response(raw: Optional[str]) -> Optional[str]:
    """
    Extract a runnable statement from a raw model response.

    Models routinely wrap SQL in markdown fences or precede it with a sentence
    of explanation despite being told not to. Passing that text straight to the
    validator would fail every time, so it is stripped here.

    Args:
        raw: Raw text returned by the model.

    Returns:
        The cleaned SQL, or ``None`` when no statement is present.
    """
    if not raw or not isinstance(raw, str):
        return None

    text = _THINK_BLOCK.sub("", raw)
    if "<think>" in text.lower():
        return None
    text = _CODE_FENCE.sub("", text)

    match = _QUERY_START.search(text)
    if not match:
        return None
    text = text[match.start():]

    semicolon = text.find(";")
    if semicolon != -1:
        text = text[:semicolon]

    return re.sub(r"\n{3,}", "\n\n", text).strip() or None


def response_text(response: Any) -> Optional[str]:
    """Read the text out of a LangChain message, a plain string, or a dict."""
    if response is None:
        return None
    if isinstance(response, str):
        return response
    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(response, dict):
        for key in ("content", "text", "output"):
            if isinstance(response.get(key), str):
                return response[key]
    return None


class TextToSQLLLM:
    """Text-to-SQL LLM integration backed by an OpenAI chat model."""

    def __init__(self, llm: Optional[Any] = None):
        """
        Initialise the engine.

        Args:
            llm: A pre-built chat model exposing ``invoke``. Injected by tests;
                when omitted a ``ChatOpenAI`` client is built from settings.
        """
        self.llm = llm if llm is not None else self._build_llm()

    @staticmethod
    def _build_llm() -> Any:
        """Construct the chat model, importing LangChain lazily."""
        if not OPENAI_API_KEY:
            raise LLMError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and fill it in."
            )
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise LLMError(
                "langchain-openai is not installed. Run: pip install -r requirements.txt"
            ) from exc

        try:
            llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
                temperature=OPENAI_TEMPERATURE,
                max_tokens=OPENAI_MAX_TOKENS,
            )
            logger.info("LLM initialised with model: %s", OPENAI_MODEL)
            return llm
        except Exception as exc:
            logger.error("Failed to initialise LLM: %s", exc)
            raise LLMError(f"Failed to initialise LLM: {exc}") from exc

    def _invoke(self, prompt: str) -> str:
        """Send a prompt and return the raw response text."""
        try:
            response = self.llm.invoke(prompt)
        except Exception as exc:
            logger.error("LLM request failed: %s", exc)
            raise LLMError(f"LLM request failed: {exc}") from exc

        text = response_text(response)
        if text is None:
            raise LLMError("Unexpected response format from the language model.")
        return text

    def generate_sql(
        self,
        natural_language_query: str,
        schema_context: str,
        table_definitions: str,
    ) -> str:
        """
        Generate an SQL query from natural language input.

        Args:
            natural_language_query: The user's question.
            schema_context: Database schema information.
            table_definitions: Detailed table structure definitions.

        Returns:
            The generated SQL query.

        Raises:
            LLMError: If generation fails or no SQL can be extracted.
        """
        if not natural_language_query or not natural_language_query.strip():
            raise LLMError("Enter a question before generating SQL.")

        prompt = build_sql_prompt(natural_language_query, schema_context, table_definitions)
        sql = clean_sql_response(self._invoke(prompt))
        if not sql:
            raise LLMError(
                "The model did not return an SQL query. Try rephrasing the question."
            )
        logger.info("SQL generated successfully")
        return sql

    def explain_query(self, sql_query: str) -> str:
        """
        Generate a plain English explanation of an SQL query.

        Args:
            sql_query: Query to explain.

        Returns:
            The explanation text.

        Raises:
            LLMError: If the request fails.
        """
        if not sql_query or not sql_query.strip():
            raise LLMError("There is no query to explain.")
        return self._invoke(EXPLAIN_PROMPT.format(sql=sql_query)).strip()
