"""
analyst - natural language to SQL analysis over a tabular dataset.

The package holds every piece of logic the Streamlit app in ``main.py`` uses,
so the pipeline can be imported, unit tested and reused without a browser.
"""
from analyst.config import LLMConfig, load_config
from analyst.errors import AnalystError, LLMError, SQLValidationError
from analyst.llm import clean_generated_sql, generate_sql
from analyst.schema import build_schema_ddl, describe_columns, sample_rows_markdown
from analyst.sql import execute_sql, load_dataframe, validate_sql

__version__ = "1.0.0"

__all__ = [
    "AnalystError",
    "LLMConfig",
    "LLMError",
    "SQLValidationError",
    "build_schema_ddl",
    "clean_generated_sql",
    "describe_columns",
    "execute_sql",
    "generate_sql",
    "load_config",
    "load_dataframe",
    "sample_rows_markdown",
    "validate_sql",
]
