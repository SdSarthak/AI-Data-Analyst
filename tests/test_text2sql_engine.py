"""Tests for the text2sql LLM engine, connector helpers and query pipeline."""
import pandas as pd
import pytest

from src.database_connector import DatabaseConnector, normalise_host, validate_identifier
from src.llm_engine import TextToSQLLLM, build_sql_prompt, clean_sql_response, response_text
from src.query_engine import QueryEngine
from src.sql_validator import SQLValidator
from utils.errors import DatabaseError, LLMError


class FakeMessage:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Stands in for a LangChain chat model."""

    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error is not None:
            raise self.error
        return FakeMessage(self.reply)


class FakeCursor:
    def __init__(self, script):
        self.script = script
        self.executed = []
        self.description = None
        self._rows = []
        self.closed = False

    def execute(self, sql):
        self.executed.append(sql)
        payload = self.script.get(sql)
        if payload is None:
            for key, value in self.script.items():
                if sql.startswith(key):
                    payload = value
                    break
        if isinstance(payload, Exception):
            raise payload
        if payload is None:
            self._rows, self.description = [], None
        else:
            columns, rows = payload
            self.description = [(c,) for c in columns]
            self._rows = rows

    def fetchall(self):
        return self._rows

    def fetchmany(self, size):
        return self._rows[:size]

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, script):
        self.script = script
        self.cursors = []
        self.closed = False

    def cursor(self):
        cursor = FakeCursor(self.script)
        self.cursors.append(cursor)
        return cursor

    def close(self):
        self.closed = True


@pytest.fixture
def configured(monkeypatch):
    """Provide catalog and schema names the connector interpolates into SQL."""
    import src.database_connector as dc

    monkeypatch.setattr(dc, "DATABRICKS_CATALOG", "main")
    monkeypatch.setattr(dc, "DATABRICKS_SCHEMA", "analytics")


class TestCleanSqlResponse:
    def test_plain_query(self):
        assert clean_sql_response("SELECT 1 FROM t") == "SELECT 1 FROM t"

    def test_strips_markdown_fence(self):
        """Regression: fenced output reached the validator and was rejected
        because it did not start with SELECT."""
        assert clean_sql_response("```sql\nSELECT 1 FROM t\n```") == "SELECT 1 FROM t"

    def test_strips_leading_prose(self):
        assert clean_sql_response("Sure, here it is:\nSELECT 1 FROM t") == "SELECT 1 FROM t"

    def test_strips_trailing_prose(self):
        assert clean_sql_response("SELECT 1 FROM t;\nThat counts rows.") == "SELECT 1 FROM t"

    def test_keeps_cte(self):
        query = "WITH x AS (SELECT 1 AS a) SELECT * FROM x"
        assert clean_sql_response(query) == query

    def test_output_passes_validation(self):
        cleaned = clean_sql_response("```sql\nSELECT a FROM t WHERE a > 1;\n```")
        assert SQLValidator.validate_sql(cleaned)[0] is True

    @pytest.mark.parametrize("raw", [None, "", "   ", "I cannot help with that."])
    def test_returns_none_when_no_query(self, raw):
        assert clean_sql_response(raw) is None


class TestResponseText:
    def test_reads_message_content(self):
        assert response_text(FakeMessage("hello")) == "hello"

    def test_reads_plain_string(self):
        assert response_text("hello") == "hello"

    def test_reads_dict(self):
        assert response_text({"content": "hello"}) == "hello"

    def test_unknown_shape_returns_none(self):
        assert response_text(object()) is None


class TestBuildSqlPrompt:
    def test_includes_all_inputs(self):
        prompt = build_sql_prompt("how many orders?", "schema here", "definitions here")
        assert "how many orders?" in prompt
        assert "schema here" in prompt
        assert "definitions here" in prompt


class TestTextToSQLLLM:
    def test_generate_sql_cleans_the_response(self):
        engine = TextToSQLLLM(llm=FakeLLM("```sql\nSELECT 1 FROM t\n```"))
        assert engine.generate_sql("how many?", "schema", "defs") == "SELECT 1 FROM t"

    def test_blank_question_is_rejected(self):
        engine = TextToSQLLLM(llm=FakeLLM("SELECT 1"))
        with pytest.raises(LLMError):
            engine.generate_sql("  ", "schema", "defs")

    def test_provider_failure_becomes_llm_error(self):
        engine = TextToSQLLLM(llm=FakeLLM(error=RuntimeError("rate limited")))
        with pytest.raises(LLMError, match="rate limited"):
            engine.generate_sql("how many?", "schema", "defs")

    def test_unusable_response_becomes_llm_error(self):
        engine = TextToSQLLLM(llm=FakeLLM("I do not know."))
        with pytest.raises(LLMError):
            engine.generate_sql("how many?", "schema", "defs")

    def test_explain_query(self):
        engine = TextToSQLLLM(llm=FakeLLM("It counts rows."))
        assert engine.explain_query("SELECT COUNT(*) FROM t") == "It counts rows."

    def test_explain_rejects_empty_query(self):
        with pytest.raises(LLMError):
            TextToSQLLLM(llm=FakeLLM("x")).explain_query("")


class TestConnectorHelpers:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("adb-1.2.azuredatabricks.net", "adb-1.2.azuredatabricks.net"),
            ("https://adb-1.2.azuredatabricks.net", "adb-1.2.azuredatabricks.net"),
            ("https://adb-1.2.azuredatabricks.net/", "adb-1.2.azuredatabricks.net"),
            ("  http://host/  ", "host"),
        ],
    )
    def test_normalise_host(self, raw, expected):
        assert normalise_host(raw) == expected

    @pytest.mark.parametrize("raw", [None, "", "   "])
    def test_missing_host_is_reported(self, raw):
        with pytest.raises(DatabaseError):
            normalise_host(raw)

    def test_validate_identifier_accepts_plain_names(self):
        assert validate_identifier("orders_2024") == "orders_2024"

    @pytest.mark.parametrize("raw", ["orders; DROP TABLE x", "1abc", "", None, "a-b"])
    def test_validate_identifier_rejects_injection(self, raw):
        with pytest.raises(DatabaseError):
            validate_identifier(raw)


class TestDatabaseConnector:
    def test_show_tables_reads_the_name_column(self, configured):
        """Regression: Databricks returns (database, tableName, isTemporary),
        so taking column 0 produced the database name for every table."""
        conn = FakeConnection(
            {"SHOW TABLES": (["database", "tableName", "isTemporary"], [
                ("analytics", "orders", False),
                ("analytics", "users", False),
            ])}
        )
        assert DatabaseConnector(connection=conn).list_tables() == ["orders", "users"]

    def test_schema_info_lists_tables(self, configured):
        conn = FakeConnection(
            {"SHOW TABLES": (["database", "tableName"], [("analytics", "orders")])}
        )
        info = DatabaseConnector(connection=conn).get_schema_info()
        assert "orders" in info

    def test_schema_info_when_empty(self, configured):
        conn = FakeConnection({"SHOW TABLES": (["database", "tableName"], [])})
        assert "no tables found" in DatabaseConnector(connection=conn).get_schema_info()

    def test_table_definitions(self, configured):
        conn = FakeConnection(
            {"DESCRIBE TABLE orders": (["col_name", "data_type"], [
                ("id", "bigint"),
                ("total", "double"),
                ("", ""),
                ("# Partition Information", ""),
            ])}
        )
        definition = DatabaseConnector(connection=conn).get_table_definitions("orders")
        assert "id: bigint" in definition
        assert "Partition Information" not in definition

    def test_table_definitions_rejects_injection(self, configured):
        conn = FakeConnection({})
        with pytest.raises(DatabaseError):
            DatabaseConnector(connection=conn).get_table_definitions("orders; DROP TABLE x")

    def test_all_definitions_skips_unreadable_tables(self, configured):
        conn = FakeConnection(
            {
                "SHOW TABLES": (["database", "tableName"], [
                    ("analytics", "orders"),
                    ("analytics", "broken"),
                ]),
                "DESCRIBE TABLE orders": (["col_name", "data_type"], [("id", "bigint")]),
                "DESCRIBE TABLE broken": RuntimeError("permission denied"),
            }
        )
        definitions = DatabaseConnector(connection=conn).get_all_table_definitions()
        assert "orders" in definitions
        assert "broken" not in definitions

    def test_execute_query_returns_a_dataframe(self, configured):
        conn = FakeConnection(
            {"SELECT id FROM orders": (["id"], [(1,), (2,)])}
        )
        result = DatabaseConnector(connection=conn).execute_query("SELECT id FROM orders")
        assert isinstance(result["data"], pd.DataFrame)
        assert result["row_count"] == 2
        assert result["columns"] == ["id"]

    def test_execute_query_applies_the_row_cap(self, configured):
        conn = FakeConnection({"SELECT id FROM orders": (["id"], [(i,) for i in range(50)])})
        result = DatabaseConnector(connection=conn).execute_query(
            "SELECT id FROM orders", max_rows=10
        )
        assert result["row_count"] == 10
        assert result["truncated"] is True

    def test_cursors_are_closed(self, configured):
        conn = FakeConnection({"SELECT 1": (["a"], [(1,)])})
        DatabaseConnector(connection=conn).execute_query("SELECT 1")
        assert all(cursor.closed for cursor in conn.cursors)

    def test_cursor_is_closed_even_when_the_query_fails(self, configured):
        conn = FakeConnection({"SELECT bad": RuntimeError("boom")})
        with pytest.raises(DatabaseError):
            DatabaseConnector(connection=conn).execute_query("SELECT bad")
        assert all(cursor.closed for cursor in conn.cursors)

    def test_close_is_idempotent(self, configured):
        connector = DatabaseConnector(connection=FakeConnection({}))
        connector.close()
        connector.close()


class FakeDb:
    def __init__(self, rows=None, error=None):
        self.rows = rows if rows is not None else [(1,)]
        self.error = error
        self.definition_calls = 0
        self.closed = False

    def get_schema_info(self):
        return "Catalog: main\nSchema: analytics\n\nTables:\n  - orders"

    def get_all_table_definitions(self):
        self.definition_calls += 1
        return "Table: orders\nColumns:\n  - id: bigint\n  - total: double"

    def execute_query(self, sql, max_rows=100):
        if self.error is not None:
            raise self.error
        return {
            "success": True,
            "data": pd.DataFrame(self.rows, columns=["id"]),
            "row_count": len(self.rows),
            "column_count": 1,
            "columns": ["id"],
            "truncated": False,
        }

    def close(self):
        self.closed = True


class TestQueryEngine:
    def build(self, reply="SELECT id FROM orders", db=None):
        return QueryEngine(
            llm=TextToSQLLLM(llm=FakeLLM(reply)),
            db=db if db is not None else FakeDb(),
        )

    def test_successful_pipeline(self):
        result = self.build().process_query("show me order ids")
        assert result["success"] is True
        assert "orders" in result["sql"]
        assert result["results"]["row_count"] == 1

    def test_real_table_definitions_reach_the_model(self):
        """Regression: a placeholder sentence was sent instead of the schema,
        so the model had to guess every column name."""
        llm = FakeLLM("SELECT id FROM orders")
        db = FakeDb()
        QueryEngine(llm=TextToSQLLLM(llm=llm), db=db).process_query("order ids")
        assert "id: bigint" in llm.prompts[0]
        assert "will be retrieved" not in llm.prompts[0]

    def test_schema_is_fetched_once_per_session(self):
        db = FakeDb()
        engine = QueryEngine(llm=TextToSQLLLM(llm=FakeLLM("SELECT id FROM orders")), db=db)
        engine.process_query("first")
        engine.process_query("second")
        assert db.definition_calls == 1

    def test_refresh_schema_forces_a_reload(self):
        db = FakeDb()
        engine = QueryEngine(llm=TextToSQLLLM(llm=FakeLLM("SELECT id FROM orders")), db=db)
        engine.process_query("first")
        engine.refresh_schema()
        engine.process_query("second")
        assert db.definition_calls == 2

    def test_unsafe_sql_is_refused_before_execution(self):
        db = FakeDb()
        engine = QueryEngine(llm=TextToSQLLLM(llm=FakeLLM("DROP TABLE orders")), db=db)
        result = engine.process_query("delete everything")
        assert result["success"] is False
        assert result["error"]

    def test_empty_question_is_rejected(self):
        result = self.build().process_query("   ")
        assert result["success"] is False

    def test_overlong_question_is_rejected(self):
        result = self.build().process_query("x" * 5000)
        assert result["success"] is False
        assert "too long" in result["error"]

    def test_llm_failure_is_reported_not_raised(self):
        engine = QueryEngine(llm=TextToSQLLLM(llm=FakeLLM(error=RuntimeError("boom"))), db=FakeDb())
        result = engine.process_query("anything")
        assert result["success"] is False
        assert "boom" in result["error"]

    def test_database_failure_is_reported_not_raised(self):
        from utils.errors import DatabaseError as DbError

        engine = QueryEngine(
            llm=TextToSQLLLM(llm=FakeLLM("SELECT id FROM orders")),
            db=FakeDb(error=DbError("warehouse asleep")),
        )
        result = engine.process_query("anything")
        assert result["success"] is False
        assert "warehouse asleep" in result["error"]
        assert result["sql"]

    def test_explain_failure_is_returned_as_text(self):
        engine = QueryEngine(llm=TextToSQLLLM(llm=FakeLLM(error=RuntimeError("boom"))), db=FakeDb())
        assert "Could not explain" in engine.explain_generated_sql("SELECT 1")

    def test_close_releases_the_connection(self):
        db = FakeDb()
        QueryEngine(llm=TextToSQLLLM(llm=FakeLLM("SELECT 1")), db=db).close()
        assert db.closed is True
