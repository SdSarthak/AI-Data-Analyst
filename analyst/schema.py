"""
Turn a pandas DataFrame into schema context a language model can use.

Giving the model real ``CREATE TABLE`` DDL (with quoted identifiers) plus a
few example rows produces far fewer invalid column references than a bare
comma separated list of names.
"""
import pandas as pd

DEFAULT_TABLE = "data"


def sqlite_type(series: pd.Series) -> str:
    """Map a pandas dtype onto the SQLite type ``df.to_sql`` will create."""
    dtype = series.dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_integer_dtype(dtype):
        return "INTEGER"
    if pd.api.types.is_float_dtype(dtype):
        return "REAL"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    return "TEXT"


def quote_identifier(name: str) -> str:
    """Quote an identifier for SQLite, escaping embedded double quotes."""
    return '"' + str(name).replace('"', '""') + '"'


def build_schema_ddl(df: pd.DataFrame, table_name: str = DEFAULT_TABLE) -> str:
    """
    Render the DataFrame as a ``CREATE TABLE`` statement.

    Args:
        df: DataFrame that will be loaded into SQLite.
        table_name: Name the table is registered under.

    Returns:
        A ``CREATE TABLE`` statement describing every column.
    """
    if len(df.columns) == 0:
        return f"CREATE TABLE {quote_identifier(table_name)} ();"

    columns = ",\n".join(
        f"    {quote_identifier(col)} {sqlite_type(df[col])}" for col in df.columns
    )
    return f"CREATE TABLE {quote_identifier(table_name)} (\n{columns}\n);"


def describe_columns(df: pd.DataFrame) -> str:
    """Return a compact ``name (TYPE), ...`` summary of the columns."""
    return ", ".join(f"{col} ({sqlite_type(df[col])})" for col in df.columns)


def sample_rows_markdown(df: pd.DataFrame, limit: int = 3) -> str:
    """
    Render the first ``limit`` rows as a markdown table for the prompt.

    Values are truncated so a column holding long free text cannot dominate
    the prompt budget. Returns an empty string when there is nothing to show.
    """
    if limit <= 0 or df.empty or len(df.columns) == 0:
        return ""

    head = df.head(limit)
    header = "| " + " | ".join(str(c) for c in head.columns) + " |"
    divider = "| " + " | ".join("---" for _ in head.columns) + " |"
    rows = [
        "| " + " | ".join(_cell(value) for value in row) + " |"
        for row in head.itertuples(index=False, name=None)
    ]
    return "\n".join([header, divider, *rows])


def _cell(value: object, max_len: int = 40) -> str:
    text = "NULL" if pd.isna(value) else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > max_len:
        text = text[: max_len - 3] + "..."
    return text
