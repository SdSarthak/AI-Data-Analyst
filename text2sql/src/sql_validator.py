"""
SQL validation and optimization module
"""
import re
from typing import Tuple, List
from utils.logger import setup_logger
from utils.errors import SQLValidationError

logger = setup_logger(__name__)


class SQLValidator:
    """
    SQL validation and optimization class
    """
    
    # SQL keywords to validate
    VALID_KEYWORDS = {
        'SELECT', 'FROM', 'WHERE', 'JOIN', 'INNER JOIN', 'LEFT JOIN',
        'RIGHT JOIN', 'FULL OUTER JOIN', 'GROUP BY', 'ORDER BY',
        'LIMIT', 'HAVING', 'AND', 'OR', 'NOT', 'IN', 'BETWEEN',
        'LIKE', 'IS', 'NULL', 'AS', 'DISTINCT', 'COUNT', 'SUM',
        'AVG', 'MIN', 'MAX', 'WITH'
    }
    
    # Dangerous keywords to prevent injection
    DANGEROUS_KEYWORDS = {
        'DROP', 'DELETE', 'TRUNCATE', 'ALTER', 'CREATE', 'INSERT',
        'UPDATE', 'EXEC', 'EXECUTE'
    }
    
    @staticmethod
    def validate_sql(sql_query: str) -> Tuple[bool, str]:
        """
        Validate SQL query syntax and safety
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            Tuple of (is_valid, message)
        """
        if not sql_query or not isinstance(sql_query, str):
            return False, "Query is empty or not a string"
        
        sql_upper = sql_query.strip().upper()
        
        # Check for dangerous keywords
        for keyword in SQLValidator.DANGEROUS_KEYWORDS:
            if re.search(rf'\b{keyword}\b', sql_upper):
                return False, f"Dangerous keyword '{keyword}' detected"
        
        # Check if query starts with SELECT
        if not sql_upper.startswith('SELECT') and not sql_upper.startswith('WITH'):
            return False, "Query must start with SELECT or WITH (CTE)"
        
        # Check for balanced parentheses
        if sql_query.count('(') != sql_query.count(')'):
            return False, "Unbalanced parentheses in query"
        
        # Check for SQL injection patterns
        if SQLValidator._check_injection_patterns(sql_query):
            return False, "Potential SQL injection detected"
        
        logger.info("SQL validation passed")
        return True, "SQL query is valid"
    
    @staticmethod
    def _check_injection_patterns(sql_query: str) -> bool:
        """
        Check for common SQL injection patterns
        
        Args:
            sql_query: SQL query to check
            
        Returns:
            True if injection pattern detected
        """
        injection_patterns = [
            r"('\s*(OR|AND)\s*'1')",
            r"(;\s*(DROP|DELETE|UPDATE))",
            r"(UNION\s+SELECT)",
            r"(-{2}|/\*|\*/)",  # Comments
        ]
        
        sql_upper = sql_query.upper()
        for pattern in injection_patterns:
            if re.search(pattern, sql_upper):
                return True
        return False
    
    @staticmethod
    def optimize_query(sql_query: str) -> str:
        """
        Optimize SQL query (basic optimization)
        
        Args:
            sql_query: SQL query to optimize
            
        Returns:
            Optimized SQL query
        """
        optimized = sql_query.strip()
        
        # Remove extra whitespace
        optimized = re.sub(r'\s+', ' ', optimized)
        
        # Add newlines for readability
        keywords_to_newline = ['SELECT', 'FROM', 'WHERE', 'GROUP BY', 'ORDER BY', 'HAVING', 'LIMIT']
        for keyword in keywords_to_newline:
            optimized = re.sub(
                rf'(\s)({keyword})(\s)',
                rf'\n{keyword} ',
                optimized,
                flags=re.IGNORECASE
            )
        
        logger.info("Query optimization completed")
        return optimized.strip()
    
    @staticmethod
    def get_query_info(sql_query: str) -> dict:
        """
        Extract information from SQL query
        
        Args:
            sql_query: SQL query to analyze
            
        Returns:
            Dictionary with query information
        """
        sql_upper = sql_query.upper()
        
        info = {
            "has_join": bool(re.search(r'\bJOIN\b', sql_upper)),
            "has_group_by": bool(re.search(r'\bGROUP\s+BY\b', sql_upper)),
            "has_order_by": bool(re.search(r'\bORDER\s+BY\b', sql_upper)),
            "has_limit": bool(re.search(r'\bLIMIT\b', sql_upper)),
            "has_where": bool(re.search(r'\bWHERE\b', sql_upper)),
            "has_aggregation": bool(re.search(r'\b(COUNT|SUM|AVG|MIN|MAX)\s*\(', sql_upper)),
            "has_cte": bool(re.search(r'\bWITH\b', sql_upper)),
        }
        
        return info
