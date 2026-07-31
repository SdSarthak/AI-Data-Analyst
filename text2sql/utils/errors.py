"""
Custom exceptions for the application
"""

class TextToSQLError(Exception):
    """Base exception for Text-to-SQL application"""
    pass


class LLMError(TextToSQLError):
    """Exception raised for LLM-related errors"""
    pass


class DatabaseError(TextToSQLError):
    """Exception raised for database connection/query errors"""
    pass


class SQLValidationError(TextToSQLError):
    """Exception raised for SQL validation errors"""
    pass


class AuthenticationError(TextToSQLError):
    """Exception raised for authentication errors"""
    pass


class QueryExecutionError(TextToSQLError):
    """Exception raised for query execution errors"""
    pass
