"""Tests for schema rendering."""
import pandas as pd

from analyst.schema import (
    build_schema_ddl,
    describe_columns,
    quote_identifier,
    sample_rows_markdown,
    sqlite_type,
    unknown_columns,
)


class TestSqliteType:
    def test_integer(self):
        assert sqlite_type(pd.Series([1, 2, 3])) == "INTEGER"

    def test_float(self):
        assert sqlite_type(pd.Series([1.5, 2.5])) == "REAL"

    def test_bool(self):
        assert sqlite_type(pd.Series([True, False])) == "INTEGER"

    def test_text(self):
        assert sqlite_type(pd.Series(["a", "b"])) == "TEXT"

    def test_datetime(self):
        series = pd.to_datetime(pd.Series(["2024-01-01", "2024-02-01"]))
        assert sqlite_type(series) == "TIMESTAMP"


class TestQuoteIdentifier:
    def test_wraps_in_double_quotes(self):
        assert quote_identifier("order date") == '"order date"'

    def test_escapes_embedded_quotes(self):
        assert quote_identifier('we"ird') == '"we""ird"'


class TestBuildSchemaDdl:
    def test_lists_every_column(self, sales_df):
        ddl = build_schema_ddl(sales_df)
        for column in sales_df.columns:
            assert f'"{column}"' in ddl

    def test_uses_the_table_name(self, sales_df):
        assert '"sales"' in build_schema_ddl(sales_df, table_name="sales")

    def test_quotes_awkward_names(self, awkward_df):
        ddl = build_schema_ddl(awkward_df)
        assert '"order date"' in ddl
        assert '"total $"' in ddl

    def test_columns_match_the_table_pandas_creates(self, sales_df):
        """The DDL shown to the model must describe the real table."""
        from analyst.sql import load_dataframe

        conn = load_dataframe(sales_df)
        try:
            described = [d[0] for d in conn.execute("SELECT * FROM data LIMIT 1").description]
        finally:
            conn.close()
        assert described == list(sales_df.columns)

    def test_handles_a_table_with_no_columns(self):
        assert "CREATE TABLE" in build_schema_ddl(pd.DataFrame())


class TestDescribeColumns:
    def test_compact_summary(self, sales_df):
        summary = describe_columns(sales_df)
        assert "region (TEXT)" in summary
        assert "units (INTEGER)" in summary


class TestSampleRowsMarkdown:
    def test_renders_a_markdown_table(self, sales_df):
        rendered = sample_rows_markdown(sales_df, limit=2)
        assert rendered.count("\n") == 3  # header, divider, two rows
        assert "region" in rendered

    def test_empty_frame_renders_nothing(self):
        assert sample_rows_markdown(pd.DataFrame()) == ""

    def test_zero_limit_renders_nothing(self, sales_df):
        assert sample_rows_markdown(sales_df, limit=0) == ""

    def test_long_values_are_truncated(self):
        df = pd.DataFrame({"note": ["x" * 200]})
        assert len(sample_rows_markdown(df, limit=1).splitlines()[2]) < 60

    def test_pipes_are_escaped(self):
        df = pd.DataFrame({"note": ["a|b"]})
        assert "\\|" in sample_rows_markdown(df, limit=1)

    def test_nulls_render_as_null(self):
        df = pd.DataFrame({"note": [None]})
        assert "NULL" in sample_rows_markdown(df, limit=1)


class TestUnknownColumns:
    def test_reports_missing_names(self, sales_df):
        assert unknown_columns(sales_df, ["region", "nope"]) == ["nope"]

    def test_is_case_insensitive(self, sales_df):
        assert unknown_columns(sales_df, ["REGION"]) == []
