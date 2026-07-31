"""
SQL validation and formatting.

Checks run against a copy of the query with string literals and comments
blanked out, so a value such as ``WHERE city = 'Update Falls'`` is not
mistaken for an UPDATE statement.
"""
import re
from typing import List, Tuple

from utils.logger import setup_logger

logger = setup_logger(__name__)

_STRING_LITERAL = re.compile(r"'(?:[^']|'')*'")
_LINE_COMMENT = re.compile(r"--[^\n]*")
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


def strip_literals_and_comments(sql_query: str) -> str:
    """Blank out string literals and comments so keyword checks are accurate."""
    without_comments = _BLOCK_COMMENT.sub(" ", _LINE_COMMENT.sub(" ", sql_query))
    return _STRING_LITERAL.sub("''", without_comments)


class SQLValidator:
    """Validation and light formatting for generated SQL."""

    # Statement types that must never be executed by this application.
    DANGEROUS_KEYWORDS = {
        "ALTER",
        "CREATE",
        "DELETE",
        "DROP",
        "EXEC",
        "EXECUTE",
        "GRANT",
        "INSERT",
        "MERGE",
        "REFRESH",
        "REPLACE",
        "REVOKE",
        "TRUNCATE",
        "UPDATE",
    }

    KEYWORDS_ON_NEW_LINE = (
        "SELECT",
        "FROM",
        "WHERE",
        "GROUP BY",
        "ORDER BY",
        "HAVING",
        "LIMIT",
    )

    @staticmethod
    def split_statements(sql_query: str) -> List[str]:
        """Split on semicolons that sit outside string literals and comments."""
        masked = strip_literals_and_comments(sql_query)
        statements = []
        start = 0
        for index, char in enumerate(masked):
            if char == ";":
                chunk = sql_query[start:index].strip()
                if chunk:
                    statements.append(chunk)
                start = index + 1
        tail = sql_query[start:].strip()
        if tail:
            statements.append(tail)
        return statements

    @staticmethod
    def validate_sql(sql_query: str) -> Tuple[bool, str]:
        """
        Validate a query's shape and safety.

        Args:
            sql_query: Query to check.

        Returns:
            ``(is_valid, message)``.
        """
        if not sql_query or not isinstance(sql_query, str) or not sql_query.strip():
            return False, "Query is empty or not a string"

        statements = SQLValidator.split_statements(sql_query)
        if len(statements) > 1:
            return False, "Only a single SQL statement may be executed"
        if not statements:
            return False, "Query is empty or not a string"

        masked = strip_literals_and_comments(statements[0]).strip()
        if not masked:
            return False, "Query is empty or not a string"

        sql_upper = masked.upper()

        if not (sql_upper.startswith("SELECT") or sql_upper.startswith("WITH")):
            return False, "Query must start with SELECT or WITH (CTE)"

        for keyword in sorted(SQLValidator.DANGEROUS_KEYWORDS):
            if re.search(rf"\b{keyword}\b", sql_upper):
                return False, f"Dangerous keyword '{keyword}' detected"

        if masked.count("(") != masked.count(")"):
            return False, "Unbalanced parentheses in query"

        logger.info("SQL validation passed")
        return True, "SQL query is valid"

    @staticmethod
    def format_query(sql_query: str) -> str:
        """
        Normalise whitespace and put major clauses on their own line.

        Only whitespace changes, so the query's meaning is unaffected. String
        literals are protected: they are removed, the formatting is applied,
        and then they are put back.

        Args:
            sql_query: Query to format.

        Returns:
            The reformatted query.
        """
        literals = []

        def _stash(match: "re.Match") -> str:
            literals.append(match.group(0))
            return f"\x00{len(literals) - 1}\x00"

        stashed = _STRING_LITERAL.sub(_stash, sql_query.strip())
        stashed = re.sub(r"\s+", " ", stashed)

        for keyword in SQLValidator.KEYWORDS_ON_NEW_LINE:
            stashed = re.sub(
                rf"\s+{keyword}\s+",
                f"\n{keyword} ",
                stashed,
                flags=re.IGNORECASE,
            )

        for index, literal in enumerate(literals):
            stashed = stashed.replace(f"\x00{index}\x00", literal)

        logger.info("Query formatting completed")
        return stashed.strip()

    # Kept so existing callers and docs that say "optimize" keep working.
    optimize_query = format_query

    @staticmethod
    def get_query_info(sql_query: str) -> dict:
        """
        Summarise the features a query uses.

        Args:
            sql_query: Query to analyse.

        Returns:
            A dictionary of boolean flags plus the referenced table names.
        """
        masked = strip_literals_and_comments(sql_query)
        upper = masked.upper()

        return {
            "has_join": bool(re.search(r"\bJOIN\b", upper)),
            "has_group_by": bool(re.search(r"\bGROUP\s+BY\b", upper)),
            "has_order_by": bool(re.search(r"\bORDER\s+BY\b", upper)),
            "has_limit": bool(re.search(r"\bLIMIT\b", upper)),
            "has_where": bool(re.search(r"\bWHERE\b", upper)),
            "has_aggregation": bool(re.search(r"\b(COUNT|SUM|AVG|MIN|MAX)\s*\(", upper)),
            "has_cte": bool(re.match(r"^\s*WITH\b", upper)),
            "has_union": bool(re.search(r"\bUNION\b", upper)),
            "tables": SQLValidator.referenced_tables(sql_query),
        }

    @staticmethod
    def referenced_tables(sql_query: str) -> List[str]:
        """Return the table names appearing after FROM or JOIN, in order."""
        masked = strip_literals_and_comments(sql_query)
        matches = re.findall(
            r"\b(?:FROM|JOIN)\s+([A-Za-z_][A-Za-z0-9_.]*)",
            masked,
            flags=re.IGNORECASE,
        )
        seen = []
        for name in matches:
            if name.upper() in {"SELECT"}:
                continue
            if name not in seen:
                seen.append(name)
        return seen
