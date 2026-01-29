"""
Main query engine combining LLM, database, and validation
"""
from typing import Dict, Any, Tuple
from src.llm_engine import TextToSQLLLM
from src.database_connector import DatabaseConnector
from src.sql_validator import SQLValidator
from utils.logger import setup_logger
from utils.errors import QueryExecutionError

logger = setup_logger(__name__)


class QueryEngine:
    """
    Main query engine orchestrating the Text-to-SQL pipeline
    """
    
    def __init__(self):
        """Initialize query engine components"""
        try:
            self.llm = TextToSQLLLM()
            self.db = DatabaseConnector()
            self.validator = SQLValidator()
            logger.info("Query engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize query engine: {str(e)}")
            raise QueryExecutionError(f"Failed to initialize query engine: {str(e)}")
    
    def process_query(self, natural_language_query: str) -> Dict[str, Any]:
        """
        Process natural language query end-to-end
        
        Args:
            natural_language_query: User's query in natural language
            
        Returns:
            Dictionary with results, SQL, and metadata
        """
        try:
            # Step 1: Get schema context
            logger.info("Step 1: Retrieving schema context")
            schema_context = self.db.get_schema_info()
            
            # Step 2: Get table definitions
            logger.info("Step 2: Retrieving table definitions")
            # For now, get definitions for common tables
            table_definitions = "Table definitions will be retrieved from database"
            
            # Step 3: Generate SQL using LLM
            logger.info("Step 3: Generating SQL query")
            sql_query = self.llm.generate_sql(
                natural_language_query,
                schema_context,
                table_definitions
            )
            
            # Step 4: Validate SQL
            logger.info("Step 4: Validating SQL query")
            is_valid, validation_message = self.validator.validate_sql(sql_query)
            
            if not is_valid:
                return {
                    "success": False,
                    "error": validation_message,
                    "sql": sql_query,
                    "natural_query": natural_language_query
                }
            
            # Step 5: Optimize SQL
            logger.info("Step 5: Optimizing SQL query")
            optimized_sql = self.validator.optimize_query(sql_query)
            
            # Step 6: Execute query
            logger.info("Step 6: Executing SQL query")
            result = self.db.execute_query(optimized_sql)
            
            # Step 7: Get query info
            query_info = self.validator.get_query_info(optimized_sql)
            
            return {
                "success": True,
                "sql": optimized_sql,
                "natural_query": natural_language_query,
                "results": result,
                "query_info": query_info
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "natural_query": natural_language_query
            }
    
    def explain_generated_sql(self, sql_query: str) -> str:
        """
        Explain an SQL query
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Explanation text
        """
        try:
            return self.llm.explain_query(sql_query)
        except Exception as e:
            logger.error(f"Error explaining query: {str(e)}")
            return f"Could not explain query: {str(e)}"
    
    def close(self):
        """Clean up resources"""
        self.db.close()
        logger.info("Query engine closed")
