"""
Main query engine combining the LLM, the database and validation.
"""
from typing import Any, Dict, Optional

from config.settings import ENABLE_QUERY_OPTIMIZATION, MAX_QUERY_LENGTH, missing_settings
from src.database_connector import DatabaseConnector
from src.llm_engine import TextToSQLLLM
from src.sql_validator import SQLValidator
from utils.errors import QueryExecutionError, TextToSQLError
from utils.logger import setup_logger

logger = setup_logger(__name__)


class QueryEngine:
    """Orchestrates the natural language to SQL pipeline."""

    def __init__(
        self,
        llm: Optional[TextToSQLLLM] = None,
        db: Optional[DatabaseConnector] = None,
        validator: Optional[SQLValidator] = None,
    ):
        """
        Args:
            llm: Pre-built LLM engine. Injected by tests.
            db: Pre-built database connector. Injected by tests.
            validator: Validator instance. Defaults to a new ``SQLValidator``.

        Raises:
            QueryExecutionError: If configuration is missing or a component
                cannot be built.
        """
        if llm is None or db is None:
            absent = missing_settings()
            if absent:
                raise QueryExecutionError(
                    "Missing configuration: "
                    + ", ".join(absent)
                    + ". Copy .env.example to .env and fill in the values."
                )

        try:
            self.llm = llm if llm is not None else TextToSQLLLM()
            self.db = db if db is not None else DatabaseConnector()
            self.validator = validator if validator is not None else SQLValidator()
            # Schema and table definitions rarely change within a session, so
            # they are fetched once instead of on every question.
            self._schema_cache: Optional[str] = None
            self._definitions_cache: Optional[str] = None
            logger.info("Query engine initialised successfully")
        except TextToSQLError:
            raise
        except Exception as exc:
            logger.error("Failed to initialise query engine: %s", exc)
            raise QueryExecutionError(f"Failed to initialise query engine: {exc}") from exc

    def refresh_schema(self) -> None:
        """Drop the cached schema so the next question re-reads it."""
        self._schema_cache = None
        self._definitions_cache = None

    def _schema_context(self) -> str:
        if self._schema_cache is None:
            self._schema_cache = self.db.get_schema_info()
        return self._schema_cache

    def _table_definitions(self) -> str:
        """
        Fetch real column-level definitions for every table.

        The engine used to send the literal string "Table definitions will be
        retrieved from database" to the model, which meant it had to guess
        every column name.
        """
        if self._definitions_cache is None:
            self._definitions_cache = self.db.get_all_table_definitions()
        return self._definitions_cache

    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Run a natural language question end to end.

        Args:
            natural_language_query: The user's question.

        Returns:
            A dictionary describing the outcome. ``success`` is always present;
            failures carry an ``error`` and, when one was produced, the ``sql``
            that failed.
        """
        if not natural_language_query or not natural_language_query.strip():
            return {
                "success": False,
                "error": "Enter a question first.",
                "natural_query": natural_language_query,
            }

        if len(natural_language_query) > MAX_QUERY_LENGTH:
            return {
                "success": False,
                "error": f"Question is too long (limit {MAX_QUERY_LENGTH} characters).",
                "natural_query": natural_language_query,
            }

        sql_query = None
        try:
            logger.info("Retrieving schema context")
            schema_context = self._schema_context()

            logger.info("Retrieving table definitions")
            table_definitions = self._table_definitions()

            logger.info("Generating SQL query")
            sql_query = self.llm.generate_sql(
                natural_language_query,
                schema_context,
                table_definitions,
            )

            logger.info("Validating SQL query")
            is_valid, validation_message = self.validator.validate_sql(sql_query)
            if not is_valid:
                return {
                    "success": False,
                    "error": validation_message,
                    "sql": sql_query,
                    "natural_query": natural_language_query,
                }

            final_sql = (
                self.validator.format_query(sql_query)
                if ENABLE_QUERY_OPTIMIZATION
                else sql_query
            )

            logger.info("Executing SQL query")
            result = self.db.execute_query(final_sql)

            return {
                "success": True,
                "sql": final_sql,
                "natural_query": natural_language_query,
                "results": result,
                "query_info": self.validator.get_query_info(final_sql),
            }

        except TextToSQLError as exc:
            logger.error("Error processing query: %s", exc)
            payload = {
                "success": False,
                "error": str(exc),
                "natural_query": natural_language_query,
            }
            if sql_query:
                payload["sql"] = sql_query
            return payload
        except Exception as exc:
            logger.exception("Unexpected error processing query")
            payload = {
                "success": False,
                "error": f"Unexpected error: {exc}",
                "natural_query": natural_language_query,
            }
            if sql_query:
                payload["sql"] = sql_query
            return payload

    def explain_generated_sql(self, sql_query: str) -> str:
        """
        Explain an SQL query in plain English.

        Args:
            sql_query: Query to explain.

        Returns:
            The explanation, or a message describing why it could not be made.
        """
        try:
            return self.llm.explain_query(sql_query)
        except TextToSQLError as exc:
            logger.error("Error explaining query: %s", exc)
            return f"Could not explain query: {exc}"

    def close(self) -> None:
        """Release the database connection."""
        self.db.close()
        logger.info("Query engine closed")
