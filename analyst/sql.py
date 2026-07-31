"""
Safe validation and execution of generated SQL against an in-memory SQLite
copy of the uploaded dataset.

Two rules drive this module:

1. A query produced by a language model is untrusted input. It is checked
   before it runs and the connection itself is locked to read-only through a
   SQLite authorizer, so a ``DROP`` slipping past the text checks still fails.
2. Validation must know the real schema. Parsing ``SELECT ... FROM data``
   against an empty database always fails with "no such table", so the
   DataFrame is loaded first and the query is checked with ``EXPLAIN``.
"""
import re
import sqlite3
from typing import List, Optional, Tuple

import pandas as pd

from analyst.config import MAX_RESULT_ROWS
from analyst.schema import DEFAULT_TABLE

# Statements that must never reach the database, even though the connection
# authorizer would also refuse them.
FORBIDDEN_KEYWORDS = (
    "ALTER",
    "ATTACH",
    "CREATE",
    "DELETE",
    "DETACH",
    "DROP",
    "INSERT",
    "PRAGMA",
    "REINDEX",
    "REPLACE",
    "TRUNCATE",
    "UPDATE",
    "VACUUM",
)

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_literals_and_comments(sql: str) -> str:
    """
    Blank out string literals and comments.

    Keyword checks run against the result so a value such as
    ``WHERE city = 'Update Falls'`` is not mistaken for an UPDATE statement.
    """
    without_comments = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql))
    return _STRING_LITERAL.sub("''", without_comments)


def split_statements(sql: str) -> List[str]:
    """Split on semicolons that sit outside string literals and comments."""
    masked = strip_literals_and_comments(sql)
    statements = []
    start = 0
    for index, char in enumerate(masked):
        if char == ";":
            chunk = sql[start:index].strip()
            if chunk:
                statements.append(chunk)
            start = index + 1
    tail = sql[start:].strip()
    if tail:
        statements.append(tail)
    return statements


def check_read_only(sql: str) -> Tuple[bool, Optional[str]]:
    """
    Check that ``sql`` is a single read-only statement.

    Returns:
        ``(True, None)`` when the query is acceptable, otherwise
        ``(False, reason)``.
    """
    if not sql or not isinstance(sql, str) or not sql.strip():
        return False, "Query is empty."

    statements = split_statements(sql)
    if len(statements) > 1:
        return False, "Only a single SQL statement may be executed."
    if not statements:
        return False, "Query is empty."

    masked = strip_literals_and_comments(statements[0]).strip()
    if not masked:
        return False, "Query is empty."

    upper = masked.upper()
    if not (upper.startswith("SELECT") or upper.startswith("WITH")):
        return False, "Only SELECT queries (optionally starting with WITH) are allowed."

    for keyword in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{keyword}\b", upper):
            return False, f"Statement type '{keyword}' is not permitted."

    if masked.count("(") != masked.count(")"):
        return False, "Unbalanced parentheses in query."

    return True, None


def _allowed_actions() -> set:
    """
    Build the set of SQLite authorizer actions a read query needs.

    Constant availability varies across Python versions, so each name is
    looked up defensively.
    """
    names = (
        "SQLITE_SELECT",
        "SQLITE_READ",
        "SQLITE_FUNCTION",
        "SQLITE_RECURSIVE",
    )
    actions = set()
    for name in names:
        value = getattr(sqlite3, name, None)
        if value is not None:
            actions.add(value)
    return actions


_READ_ONLY_ACTIONS = _allowed_actions()


def _read_only_authorizer(action, arg1, arg2, db_name, trigger):  # noqa: ARG001
    """SQLite authorizer callback allowing reads and denying everything else."""
    if action in _READ_ONLY_ACTIONS:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def load_dataframe(
    df: pd.DataFrame,
    table_name: str = DEFAULT_TABLE,
    read_only: bool = True,
) -> sqlite3.Connection:
    """
    Load a DataFrame into a fresh in-memory SQLite database.

    Args:
        df: Data to expose to SQL.
        table_name: Name the table is registered under.
        read_only: Install the authorizer that blocks writes. Disable only
            while building the database.

    Returns:
        An open connection. The caller owns it and must close it.
    """
    conn = sqlite3.connect(":memory:")
    try:
        df.to_sql(table_name, conn, index=False, if_exists="replace")
    except Exception:
        conn.close()
        raise
    if read_only:
        conn.set_authorizer(_read_only_authorizer)
    return conn


def validate_sql(
    sql: str,
    conn: Optional[sqlite3.Connection] = None,
) -> Tuple[bool, Optional[str]]:
    """
    Validate a generated query.

    Always returns a two element tuple, so callers can unpack the result
    unconditionally.

    Args:
        sql: Query to check.
        conn: Optional connection holding the real table. When supplied the
            query is additionally prepared with ``EXPLAIN``, which catches
            unknown tables and columns without running the query.

    Returns:
        ``(is_valid, error_message)`` where ``error_message`` is ``None`` on
        success.
    """
    is_read_only, reason = check_read_only(sql)
    if not is_read_only:
        return False, reason

    if conn is None:
        return True, None

    try:
        cursor = conn.cursor()
        cursor.execute(f"EXPLAIN {sql}")
        cursor.close()
    except sqlite3.Error as exc:
        return False, f"SQL error: {exc}"
    return True, None


def has_limit(sql: str) -> bool:
    """Report whether the query already constrains its row count."""
    return bool(re.search(r"\bLIMIT\b", strip_literals_and_comments(sql), re.IGNORECASE))


def apply_row_limit(sql: str, max_rows: int = MAX_RESULT_ROWS) -> str:
    """
    Append a ``LIMIT`` when the query does not have one.

    Keeps an unbounded ``SELECT *`` over a large upload from exhausting memory.
    """
    if max_rows <= 0 or has_limit(sql):
        return sql.rstrip().rstrip(";")
    return f"{sql.rstrip().rstrip(';')}\nLIMIT {max_rows}"


def execute_sql(
    df: pd.DataFrame,
    sql: str,
    table_name: str = DEFAULT_TABLE,
    max_rows: int = MAX_RESULT_ROWS,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Validate and run a query against the DataFrame.

    Args:
        df: Dataset to query.
        sql: SQL to execute.
        table_name: Name the DataFrame is registered under.
        max_rows: Row cap applied when the query has no ``LIMIT``.

    Returns:
        ``(result, None)`` on success or ``(None, error_message)`` on failure.
        Errors are returned rather than raised so the UI can show them next to
        the query that produced them.
    """
    if df is None:
        return None, "No dataset loaded."

    try:
        conn = load_dataframe(df, table_name=table_name)
    except Exception as exc:  # pragma: no cover - depends on pandas internals
        return None, f"Could not load the dataset into SQLite: {exc}"

    try:
        limited = apply_row_limit(sql, max_rows)
        is_valid, error = validate_sql(limited, conn)
        if not is_valid:
            return None, error
        return pd.read_sql_query(limited, conn), None
    except sqlite3.Error as exc:
        return None, f"SQL error: {exc}"
    except Exception as exc:
        return None, f"Error executing SQL query: {exc}"
    finally:
        conn.close()
