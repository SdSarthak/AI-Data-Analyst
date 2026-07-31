"""
Database connection and query execution module
"""
from typing import List, Dict, Any, Optional
import pandas as pd
from databricks import sql
from config.settings import (
    DATABRICKS_HOST,
    DATABRICKS_HTTP_PATH,
    DATABRICKS_TOKEN,
    DATABRICKS_CATALOG,
    DATABRICKS_SCHEMA
)
from utils.logger import setup_logger
from utils.errors import DatabaseError

logger = setup_logger(__name__)


class DatabaseConnector:
    """
    Database connector for Databricks
    """
    
    def __init__(self):
        """Initialize Databricks connection"""
        self.connection = None
        self.connect()
    
    def connect(self):
        """Establish connection to Databricks"""
        try:
            self.connection = sql.connect(
                host=DATABRICKS_HOST,
                http_path=DATABRICKS_HTTP_PATH,
                authentication="pat",
                token=DATABRICKS_TOKEN
            )
            logger.info("Connected to Databricks successfully")
        except Exception as e:
            logger.error(f"Failed to connect to Databricks: {str(e)}")
            raise DatabaseError(f"Failed to connect to Databricks: {str(e)}")
    
    def get_schema_info(self) -> str:
        """
        Get database schema information
        
        Returns:
            Schema context string
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"USE CATALOG {DATABRICKS_CATALOG}")
            cursor.execute(f"USE SCHEMA {DATABRICKS_SCHEMA}")
            
            cursor.execute("SHOW TABLES")
            tables = cursor.fetchall()
            
            schema_info = f"Catalog: {DATABRICKS_CATALOG}\nSchema: {DATABRICKS_SCHEMA}\n\nTables:\n"
            schema_info += "\n".join([str(table) for table in tables])
            
            cursor.close()
            return schema_info
        except Exception as e:
            logger.error(f"Error getting schema info: {str(e)}")
            raise DatabaseError(f"Failed to get schema info: {str(e)}")
    
    def get_table_definitions(self, table_name: str) -> str:
        """
        Get detailed table structure
        
        Args:
            table_name: Name of the table
            
        Returns:
            Table definition/schema string
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"DESCRIBE TABLE {DATABRICKS_CATALOG}.{DATABRICKS_SCHEMA}.{table_name}")
            columns = cursor.fetchall()
            
            table_def = f"Table: {table_name}\nColumns:\n"
            for col in columns:
                table_def += f"  - {col[0]}: {col[1]}\n"
            
            cursor.close()
            return table_def
        except Exception as e:
            logger.error(f"Error getting table definition for {table_name}: {str(e)}")
            raise DatabaseError(f"Failed to get table definition: {str(e)}")
    
    def execute_query(self, sql_query: str) -> Dict[str, Any]:
        """
        Execute SQL query and return results
        
        Args:
            sql_query: SQL query to execute
            
        Returns:
            Dictionary with query results and metadata
            
        Raises:
            DatabaseError: If query execution fails
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"USE CATALOG {DATABRICKS_CATALOG}")
            cursor.execute(f"USE SCHEMA {DATABRICKS_SCHEMA}")
            cursor.execute(sql_query)
            
            # Get column names
            columns = [desc[0] for desc in cursor.description]
            
            # Fetch results
            results = cursor.fetchall()
            
            # Convert to DataFrame for better display
            df = pd.DataFrame(results, columns=columns)
            
            cursor.close()
            
            return {
                "success": True,
                "data": df,
                "row_count": len(df),
                "column_count": len(columns),
                "columns": columns
            }
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            raise DatabaseError(f"Query execution failed: {str(e)}")
    
    def validate_table_exists(self, table_name: str) -> bool:
        """
        Check if table exists in database
        
        Args:
            table_name: Name of the table
            
        Returns:
            True if table exists, False otherwise
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(f"USE CATALOG {DATABRICKS_CATALOG}")
            cursor.execute(f"USE SCHEMA {DATABRICKS_SCHEMA}")
            cursor.execute(f"SHOW TABLES LIKE '{table_name}'")
            result = cursor.fetchone()
            cursor.close()
            return result is not None
        except Exception as e:
            logger.warning(f"Error validating table existence: {str(e)}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            logger.info("Database connection closed")
