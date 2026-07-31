"""Tests for validation and execution of generated SQL."""
import sqlite3

import pytest

from analyst.sql import (
    apply_row_limit,
    check_read_only,
    execute_sql,
    has_limit,
    load_dataframe,
    split_statements,
    strip_literals_and_comments,
    validate_sql,
)


class TestStripLiteralsAndComments:
    def test_blanks_string_contents(self):
        masked = strip_literals_and_comments("SELECT * FROM data WHERE x = 'drop me'")
        assert "drop" not in masked.lower()

    def test_keeps_escaped_quotes_intact(self):
        masked = strip_literals_and_comments("SELECT 'it''s fine' FROM data")
        assert "FROM data" in masked

    def test_removes_line_comment(self):
        assert "DROP" not in strip_literals_and_comments("SELECT 1 -- DROP TABLE data")

    def test_removes_block_comment(self):
        masked = strip_literals_and_comments("SELECT /* DROP TABLE data */ 1")
        assert "DROP" not in masked


class TestSplitStatements:
    def test_single_statement(self):
        assert split_statements("SELECT 1") == ["SELECT 1"]

    def test_trailing_semicolon_is_not_a_second_statement(self):
        assert split_statements("SELECT 1;") == ["SELECT 1"]

    def test_two_statements(self):
        assert len(split_statements("SELECT 1; SELECT 2")) == 2

    def test_semicolon_inside_literal_does_not_split(self):
        assert len(split_statements("SELECT * FROM data WHERE x = 'a;b'")) == 1


class TestCheckReadOnly:
    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM data",
            "select region from data",
            "WITH t AS (SELECT 1 AS a) SELECT * FROM t",
            "SELECT * FROM data WHERE region = 'update this'",
        ],
    )
    def test_accepts_read_queries(self, query):
        assert check_read_only(query) == (True, None)

    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE data",
            "DELETE FROM data",
            "INSERT INTO data VALUES (1)",
            "UPDATE data SET units = 0",
            "PRAGMA table_info(data)",
            "ATTACH DATABASE 'x.db' AS x",
            "CREATE TABLE t (a INT)",
        ],
    )
    def test_rejects_write_queries(self, query):
        is_valid, error = check_read_only(query)
        assert is_valid is False
        assert error

    def test_rejects_stacked_statements(self):
        is_valid, error = check_read_only("SELECT 1; DROP TABLE data")
        assert is_valid is False
        assert "single SQL statement" in error

    def test_rejects_unbalanced_parentheses(self):
        is_valid, error = check_read_only("SELECT COUNT( FROM data")
        assert is_valid is False
        assert "parenthes" in error.lower()

    @pytest.mark.parametrize("query", ["", "   ", None, 42])
    def test_rejects_empty_input(self, query):
        assert check_read_only(query)[0] is False

    def test_column_named_like_a_keyword_is_allowed(self):
        assert check_read_only("SELECT updated_at, created_at FROM data")[0] is True


class TestValidateSql:
    def test_always_returns_a_two_tuple(self, sales_df):
        """Regression: the old helper returned a bare True on success, so
        `is_valid, error = validate_sql(...)` blew up on every valid query."""
        conn = load_dataframe(sales_df)
        try:
            for query in ("SELECT * FROM data", "DROP TABLE data", "SELECT bad FROM data"):
                result = validate_sql(query, conn)
                assert isinstance(result, tuple) and len(result) == 2
                is_valid, error = result
                assert isinstance(is_valid, bool)
        finally:
            conn.close()

    def test_valid_query_against_real_schema(self, sales_df):
        """Regression: validating against an empty database made every query
        referencing the uploaded table fail with 'no such table'."""
        conn = load_dataframe(sales_df)
        try:
            assert validate_sql("SELECT region FROM data", conn) == (True, None)
        finally:
            conn.close()

    def test_unknown_column_is_caught_before_execution(self, sales_df):
        conn = load_dataframe(sales_df)
        try:
            is_valid, error = validate_sql("SELECT nope FROM data", conn)
            assert is_valid is False
            assert "nope" in error
        finally:
            conn.close()

    def test_works_without_a_connection(self):
        assert validate_sql("SELECT * FROM data") == (True, None)


class TestLoadDataframe:
    def test_authorizer_blocks_writes_even_without_text_checks(self, sales_df):
        conn = load_dataframe(sales_df)
        try:
            with pytest.raises(sqlite3.Error):
                conn.execute("DROP TABLE data")
        finally:
            conn.close()

    def test_reads_still_work(self, sales_df):
        conn = load_dataframe(sales_df)
        try:
            assert conn.execute("SELECT COUNT(*) FROM data").fetchone()[0] == 4
        finally:
            conn.close()


class TestRowLimit:
    def test_detects_existing_limit(self):
        assert has_limit("SELECT * FROM data LIMIT 10") is True

    def test_ignores_limit_inside_a_literal(self):
        assert has_limit("SELECT * FROM data WHERE note = 'no limit'") is False

    def test_appends_limit_when_missing(self):
        assert apply_row_limit("SELECT * FROM data", 50).endswith("LIMIT 50")

    def test_leaves_existing_limit_alone(self):
        assert apply_row_limit("SELECT * FROM data LIMIT 5", 50) == "SELECT * FROM data LIMIT 5"

    def test_strips_trailing_semicolon(self):
        assert ";" not in apply_row_limit("SELECT * FROM data;", 50)

    def test_zero_disables_the_cap(self):
        assert apply_row_limit("SELECT * FROM data", 0) == "SELECT * FROM data"


class TestExecuteSql:
    def test_aggregation(self, sales_df):
        result, error = execute_sql(
            sales_df, "SELECT region, SUM(units) AS total FROM data GROUP BY region"
        )
        assert error is None
        assert dict(zip(result["region"], result["total"])) == {
            "east": 15,
            "west": 20,
            "north": 7,
        }

    def test_cte(self, sales_df):
        result, error = execute_sql(
            sales_df,
            "WITH totals AS (SELECT region, SUM(revenue) r FROM data GROUP BY region) "
            "SELECT region FROM totals ORDER BY r DESC",
        )
        assert error is None
        assert result["region"].iloc[0] == "west"

    def test_quoted_column_names(self, awkward_df):
        result, error = execute_sql(awkward_df, 'SELECT "order date", "total $" FROM data')
        assert error is None
        assert len(result) == 2

    def test_empty_result_is_not_an_error(self, sales_df):
        result, error = execute_sql(sales_df, "SELECT * FROM data WHERE units > 1000")
        assert error is None
        assert result.empty

    def test_row_cap_is_applied(self, sales_df):
        result, error = execute_sql(sales_df, "SELECT * FROM data", max_rows=2)
        assert error is None
        assert len(result) == 2

    def test_write_query_is_refused(self, sales_df):
        result, error = execute_sql(sales_df, "DROP TABLE data")
        assert result is None
        assert error

    def test_bad_column_returns_error_not_exception(self, sales_df):
        result, error = execute_sql(sales_df, "SELECT missing FROM data")
        assert result is None
        assert "missing" in error

    def test_source_dataframe_is_untouched(self, sales_df):
        before = sales_df.copy()
        execute_sql(sales_df, "SELECT * FROM data")
        assert sales_df.equals(before)

    def test_missing_dataframe(self):
        result, error = execute_sql(None, "SELECT 1")
        assert result is None
        assert error
