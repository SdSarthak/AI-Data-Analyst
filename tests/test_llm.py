"""Tests for prompt building and cleaning of model responses. No network."""
import pytest

from analyst.config import LLMConfig
from analyst.errors import LLMError
from analyst.llm import build_prompt, clean_generated_sql, generate_sql

CONFIG = LLMConfig(api_key="test-token", model="test/model", provider="test")


class TestCleanGeneratedSql:
    def test_plain_query_passes_through(self):
        assert clean_generated_sql("SELECT * FROM data") == "SELECT * FROM data"

    def test_strips_markdown_fence(self):
        raw = "```sql\nSELECT * FROM data\n```"
        assert clean_generated_sql(raw) == "SELECT * FROM data"

    def test_strips_unlabelled_fence(self):
        assert clean_generated_sql("```\nSELECT 1 FROM data\n```") == "SELECT 1 FROM data"

    def test_strips_reasoning_block(self):
        raw = "<think>The user wants a count.</think>\nSELECT COUNT(*) FROM data"
        assert clean_generated_sql(raw) == "SELECT COUNT(*) FROM data"

    def test_drops_leading_prose(self):
        raw = "Here is the query you asked for:\nSELECT * FROM data"
        assert clean_generated_sql(raw) == "SELECT * FROM data"

    def test_drops_trailing_prose_after_semicolon(self):
        raw = "SELECT * FROM data;\nThis returns every row."
        assert clean_generated_sql(raw) == "SELECT * FROM data"

    def test_rewrites_placeholder_table_names(self):
        assert "FROM data" in clean_generated_sql("SELECT * FROM your_table")

    def test_rewrites_count_row_paraphrase(self):
        assert clean_generated_sql("SELECT COUNT ROWS FROM data") == "SELECT COUNT(*) FROM data"

    def test_keeps_cte_queries(self):
        raw = "WITH t AS (SELECT 1 AS a) SELECT * FROM t"
        assert clean_generated_sql(raw) == raw

    def test_custom_table_name(self):
        cleaned = clean_generated_sql("SELECT * FROM table_name", table_name="sales")
        assert "FROM sales" in cleaned

    @pytest.mark.parametrize(
        "raw",
        [
            None,
            "",
            "   ",
            "I cannot answer that question.",
            "<think>still reasoning and then the response was cut off",
        ],
    )
    def test_returns_none_when_there_is_no_query(self, raw):
        assert clean_generated_sql(raw) is None


class TestBuildPrompt:
    def test_includes_schema_and_question(self, sales_df):
        prompt = build_prompt("How many rows?", df=sales_df)
        assert "How many rows?" in prompt
        assert "CREATE TABLE" in prompt
        assert '"region"' in prompt

    def test_includes_sample_rows(self, sales_df):
        assert "Example rows" in build_prompt("How many rows?", df=sales_df)

    def test_accepts_a_schema_string_without_a_dataframe(self):
        prompt = build_prompt("How many?", schema="CREATE TABLE data (a INT);")
        assert "CREATE TABLE data (a INT);" in prompt
        assert "Example rows" not in prompt

    def test_requires_schema_or_dataframe(self):
        with pytest.raises(ValueError):
            build_prompt("How many?")


class TestGenerateSql:
    def test_returns_cleaned_sql(self, sales_df, fake_client):
        client = fake_client("```sql\nSELECT COUNT(*) FROM data\n```")
        sql = generate_sql("How many rows?", df=sales_df, client=client, config=CONFIG)
        assert sql == "SELECT COUNT(*) FROM data"

    def test_passes_configured_model_through(self, sales_df, fake_client):
        client = fake_client("SELECT 1 FROM data")
        generate_sql("anything", df=sales_df, client=client, config=CONFIG)
        assert client.completions.last_kwargs["model"] == "test/model"

    def test_empty_question_is_rejected(self, sales_df, fake_client):
        with pytest.raises(LLMError):
            generate_sql("   ", df=sales_df, client=fake_client("SELECT 1"), config=CONFIG)

    def test_provider_failure_becomes_llm_error(self, sales_df, fake_client):
        client = fake_client(error=RuntimeError("connection reset"))
        with pytest.raises(LLMError, match="connection reset"):
            generate_sql("How many?", df=sales_df, client=client, config=CONFIG)

    def test_unusable_response_becomes_llm_error(self, sales_df, fake_client):
        client = fake_client("Sorry, I do not know.")
        with pytest.raises(LLMError):
            generate_sql("How many?", df=sales_df, client=client, config=CONFIG)


class TestGeneratedSqlIsExecutable:
    """The generation and execution halves must fit together."""

    def test_round_trip(self, sales_df, fake_client):
        from analyst.sql import execute_sql

        client = fake_client(
            "<think>group by region</think>\n```sql\n"
            "SELECT region, SUM(units) AS total FROM your_table GROUP BY region;\n```"
        )
        sql = generate_sql("Units by region?", df=sales_df, client=client, config=CONFIG)
        result, error = execute_sql(sales_df, sql)
        assert error is None
        assert set(result["region"]) == {"east", "west", "north"}
