"""
Database connection and query execution against Databricks SQL.

The ``databricks`` driver is imported lazily so the rest of the package stays
importable (and testable) on a machine without it installed.
"""
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from config.settings import (
    DATABRICKS_CATALOG,
    DATABRICKS_HOST,
    DATABRICKS_HTTP_PATH,
    DATABRICKS_SCHEMA,
    DATABRICKS_TOKEN,
    MAX_ROWS_DISPLAY,
)
from utils.errors import DatabaseError
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Catalog, schema and table names are interpolated into SQL, so they are
# restricted to plain identifiers.
_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def validate_identifier(name: str, kind: str = "identifier") -> str:
    """
    Check that a name is a bare SQL identifier before interpolating it.

    Args:
        name: Candidate identifier.
        kind: Label used in the error message.

    Returns:
        The identifier unchanged.

    Raises:
        DatabaseError: If the name is not a plain identifier.
    """
    if not name or not _IDENTIFIER.match(str(name)):
        raise DatabaseError(f"Invalid {kind}: {name!r}")
    return str(name)


def normalise_host(host: Optional[str]) -> str:
    """
    Reduce a Databricks workspace URL to the bare host name.

    The driver wants ``adb-123.4.azuredatabricks.net``, but the value copied
    out of a browser is usually ``https://adb-123.4.azuredatabricks.net/``.

    Raises:
        DatabaseError: If the host is missing.
    """
    if not host or not str(host).strip():
        raise DatabaseError("DATABRICKS_HOST is not set.")
    cleaned = str(host).strip()
    cleaned = re.sub(r"^https?://", "", cleaned)
    return cleaned.rstrip("/")


class DatabaseConnector:
    """Read-only connector for a Databricks SQL warehouse."""

    def __init__(self, connection: Optional[Any] = None):
        """
        Args:
            connection: An existing DB-API connection. Injected by tests; when
                omitted a real Databricks connection is opened.
        """
        self.connection = connection
        if self.connection is None:
            self.connect()

    def connect(self) -> None:
        """
        Open a connection to the configured Databricks warehouse.

        Raises:
            DatabaseError: If configuration is missing or the connection fails.
        """
        try:
            from databricks import sql
        except ImportError as exc:
            raise DatabaseError(
                "databricks-sql-connector is not installed. "
                "Run: pip install -r requirements.txt"
            ) from exc

        if not DATABRICKS_HTTP_PATH:
            raise DatabaseError("DATABRICKS_HTTP_PATH is not set.")
        if not DATABRICKS_TOKEN:
            raise DatabaseError("DATABRICKS_TOKEN is not set.")

        try:
            # The driver's parameters are server_hostname / http_path /
            # access_token. Passing host= and token= raises a TypeError.
            self.connection = sql.connect(
                server_hostname=normalise_host(DATABRICKS_HOST),
                http_path=DATABRICKS_HTTP_PATH,
                access_token=DATABRICKS_TOKEN,
            )
            logger.info("Connected to Databricks successfully")
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error("Failed to connect to Databricks: %s", exc)
            raise DatabaseError(f"Failed to connect to Databricks: {exc}") from exc

    def _cursor(self):
        """Open a cursor positioned on the configured catalog and schema."""
        if self.connection is None:
            raise DatabaseError("Not connected to Databricks.")
        catalog = validate_identifier(DATABRICKS_CATALOG, "catalog")
        schema = validate_identifier(DATABRICKS_SCHEMA, "schema")
        cursor = self.connection.cursor()
        cursor.execute(f"USE CATALOG {catalog}")
        cursor.execute(f"USE SCHEMA {schema}")
        return cursor

    def list_tables(self) -> List[str]:
        """
        Return the table names in the configured schema.

        Raises:
            DatabaseError: If the listing fails.
        """
        try:
            cursor = self._cursor()
            try:
                cursor.execute("SHOW TABLES")
                rows = cursor.fetchall()
            finally:
                cursor.close()
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error("Error listing tables: %s", exc)
            raise DatabaseError(f"Failed to list tables: {exc}") from exc

        return [self._table_name_from_row(row) for row in rows if self._table_name_from_row(row)]

    @staticmethod
    def _table_name_from_row(row: Any) -> Optional[str]:
        """
        Pull the table name out of a ``SHOW TABLES`` row.

        Databricks returns ``(database, tableName, isTemporary)``, so the name
        is the second column rather than the first.
        """
        if row is None:
            return None
        name = getattr(row, "tableName", None)
        if isinstance(name, str):
            return name
        if isinstance(row, (list, tuple)):
            if len(row) >= 2 and isinstance(row[1], str):
                return row[1]
            if row and isinstance(row[0], str):
                return row[0]
        if isinstance(row, str):
            return row
        return None

    def get_schema_info(self) -> str:
        """
        Build the schema context string handed to the language model.

        Returns:
            A description of the catalog, schema and available tables.
        """
        tables = self.list_tables()
        header = f"Catalog: {DATABRICKS_CATALOG}\nSchema: {DATABRICKS_SCHEMA}\n\nTables:\n"
        if not tables:
            return header + "  (no tables found)"
        return header + "\n".join(f"  - {name}" for name in tables)

    def get_table_definitions(self, table_name: str) -> str:
        """
        Describe one table's columns.

        Args:
            table_name: Table to describe.

        Returns:
            A ``Table: ...`` block listing each column and its type.

        Raises:
            DatabaseError: If the description fails.
        """
        table = validate_identifier(table_name, "table name")
        try:
            cursor = self._cursor()
            try:
                cursor.execute(f"DESCRIBE TABLE {table}")
                columns = cursor.fetchall()
            finally:
                cursor.close()
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error("Error describing %s: %s", table, exc)
            raise DatabaseError(f"Failed to get table definition: {exc}") from exc

        lines = [f"Table: {table}", "Columns:"]
        for col in columns:
            if not col or not col[0] or str(col[0]).startswith("#"):
                # DESCRIBE appends partition metadata rows after a blank row.
                continue
            col_type = col[1] if len(col) > 1 else "unknown"
            lines.append(f"  - {col[0]}: {col_type}")
        return "\n".join(lines)

    def get_all_table_definitions(self, tables: Optional[List[str]] = None) -> str:
        """
        Describe every table in the schema.

        Failures on a single table are logged and skipped so one inaccessible
        table does not block the whole request.

        Args:
            tables: Optional explicit table list. Defaults to every table.

        Returns:
            Concatenated definitions, or a note when nothing could be read.
        """
        names = tables if tables is not None else self.list_tables()
        blocks = []
        for name in names:
            try:
                blocks.append(self.get_table_definitions(name))
            except DatabaseError as exc:
                logger.warning("Skipping table %s: %s", name, exc)
        return "\n\n".join(blocks) if blocks else "No table definitions available."

    def execute_query(self, sql_query: str, max_rows: int = MAX_ROWS_DISPLAY) -> Dict[str, Any]:
        """
        Execute a query and return its results.

        Args:
            sql_query: SQL to run.
            max_rows: Maximum number of rows to fetch.

        Returns:
            A dictionary with the result DataFrame and row/column metadata.

        Raises:
            DatabaseError: If execution fails.
        """
        try:
            cursor = self._cursor()
            try:
                cursor.execute(sql_query)
                columns = [desc[0] for desc in (cursor.description or [])]
                rows = cursor.fetchmany(max_rows) if max_rows > 0 else cursor.fetchall()
            finally:
                cursor.close()
        except DatabaseError:
            raise
        except Exception as exc:
            logger.error("Error executing query: %s", exc)
            raise DatabaseError(f"Query execution failed: {exc}") from exc

        df = pd.DataFrame(list(rows), columns=columns)
        return {
            "success": True,
            "data": df,
            "row_count": len(df),
            "column_count": len(columns),
            "columns": columns,
            "truncated": max_rows > 0 and len(df) >= max_rows,
        }

    def validate_table_exists(self, table_name: str) -> bool:
        """Report whether a table exists in the configured schema."""
        try:
            return validate_identifier(table_name, "table name") in self.list_tables()
        except DatabaseError as exc:
            logger.warning("Error validating table existence: %s", exc)
            return False

    def close(self) -> None:
        """Close the connection if one is open."""
        if self.connection is not None:
            try:
                self.connection.close()
            except Exception as exc:  # pragma: no cover - driver specific
                logger.warning("Error closing connection: %s", exc)
            finally:
                self.connection = None
                logger.info("Database connection closed")
