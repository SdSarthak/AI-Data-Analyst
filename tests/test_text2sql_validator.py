"""Tests for the text2sql SQL validator."""
import pytest

from src.sql_validator import SQLValidator, strip_literals_and_comments


class TestStripLiteralsAndComments:
    def test_blanks_literal_contents(self):
        assert "DROP" not in strip_literals_and_comments(
            "SELECT * FROM t WHERE c = 'DROP TABLE'"
        ).upper()

    def test_handles_doubled_quotes(self):
        assert "FROM t" in strip_literals_and_comments("SELECT 'it''s' FROM t")

    def test_removes_comments(self):
        assert "DROP" not in strip_literals_and_comments("SELECT 1 -- DROP TABLE t").upper()
        assert "DROP" not in strip_literals_and_comments("SELECT /* DROP */ 1").upper()


class TestValidateSql:
    @pytest.mark.parametrize(
        "query",
        [
            "SELECT * FROM orders",
            "select id from orders",
            "WITH t AS (SELECT 1 AS a) SELECT * FROM t",
            "SELECT a FROM t UNION SELECT b FROM u",
        ],
    )
    def test_accepts_read_queries(self, query):
        assert SQLValidator.validate_sql(query)[0] is True

    def test_union_select_is_allowed(self):
        """UNION is ordinary SQL a model produces for 'combine A and B'; the
        read-only checks already prevent damage."""
        assert SQLValidator.validate_sql("SELECT a FROM t UNION SELECT b FROM u")[0] is True

    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE orders",
            "DELETE FROM orders",
            "INSERT INTO orders VALUES (1)",
            "UPDATE orders SET total = 0",
            "MERGE INTO orders USING x ON 1=1",
            "GRANT SELECT ON orders TO bob",
        ],
    )
    def test_rejects_write_queries(self, query):
        is_valid, message = SQLValidator.validate_sql(query)
        assert is_valid is False
        assert message

    def test_rejects_stacked_statements(self):
        is_valid, message = SQLValidator.validate_sql("SELECT 1; DROP TABLE orders")
        assert is_valid is False
        assert "single SQL statement" in message

    def test_trailing_semicolon_is_fine(self):
        assert SQLValidator.validate_sql("SELECT 1 FROM t;")[0] is True

    def test_keyword_inside_a_literal_is_not_flagged(self):
        """Regression: checking the raw text rejected any row whose data
        happened to contain a word like 'update'."""
        assert SQLValidator.validate_sql("SELECT * FROM t WHERE note = 'update me'")[0] is True

    def test_comment_smuggled_write_is_ignored_not_rejected(self):
        assert SQLValidator.validate_sql("SELECT 1 FROM t -- DROP TABLE t")[0] is True

    def test_rejects_unbalanced_parentheses(self):
        assert SQLValidator.validate_sql("SELECT COUNT( FROM t")[0] is False

    @pytest.mark.parametrize("query", ["", "   ", None, 123])
    def test_rejects_empty_input(self, query):
        assert SQLValidator.validate_sql(query)[0] is False


class TestFormatQuery:
    def test_breaks_clauses_onto_new_lines(self):
        formatted = SQLValidator.format_query("SELECT a FROM t WHERE a > 1 ORDER BY a")
        assert formatted.splitlines() == ["SELECT a", "FROM t", "WHERE a > 1", "ORDER BY a"]

    def test_collapses_repeated_whitespace(self):
        assert "  " not in SQLValidator.format_query("SELECT    a     FROM t")

    def test_leaves_string_literals_untouched(self):
        """Regression: formatting rewrote keywords inside quoted values."""
        formatted = SQLValidator.format_query(
            "SELECT a FROM t WHERE note = 'travelled from here to where'"
        )
        assert "'travelled from here to where'" in formatted

    def test_formatting_preserves_validity(self):
        query = "SELECT a, b FROM t WHERE a > 1 GROUP BY a HAVING COUNT(*) > 2 ORDER BY a LIMIT 10"
        assert SQLValidator.validate_sql(SQLValidator.format_query(query))[0] is True

    def test_optimize_query_is_still_available(self):
        assert SQLValidator.optimize_query("SELECT a FROM t") == "SELECT a\nFROM t"


class TestGetQueryInfo:
    def test_detects_features(self):
        info = SQLValidator.get_query_info(
            "SELECT a, COUNT(*) FROM t JOIN u ON t.id = u.id "
            "WHERE a > 1 GROUP BY a ORDER BY a LIMIT 5"
        )
        assert info["has_join"] and info["has_group_by"] and info["has_order_by"]
        assert info["has_limit"] and info["has_where"] and info["has_aggregation"]

    def test_cte_only_flagged_at_the_start(self):
        """Regression: any query containing the word WITH was flagged as a CTE."""
        assert SQLValidator.get_query_info("SELECT * FROM t")["has_cte"] is False
        assert SQLValidator.get_query_info("WITH x AS (SELECT 1) SELECT * FROM x")["has_cte"] is True

    def test_reports_referenced_tables(self):
        info = SQLValidator.get_query_info("SELECT * FROM orders JOIN users ON 1=1")
        assert info["tables"] == ["orders", "users"]


class TestReferencedTables:
    def test_deduplicates(self):
        tables = SQLValidator.referenced_tables("SELECT * FROM t JOIN t ON 1=1")
        assert tables == ["t"]

    def test_handles_qualified_names(self):
        assert SQLValidator.referenced_tables("SELECT * FROM cat.sch.tbl") == ["cat.sch.tbl"]

    def test_ignores_names_inside_literals(self):
        assert SQLValidator.referenced_tables("SELECT 'from secrets' FROM t") == ["t"]
