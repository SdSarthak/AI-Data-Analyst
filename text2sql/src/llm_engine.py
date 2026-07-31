"""
LLM Integration module for Text-to-SQL generation
"""
from typing import Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from config.settings import OPENAI_API_KEY, OPENAI_MODEL
from utils.logger import setup_logger
from utils.errors import LLMError

logger = setup_logger(__name__)


class TextToSQLLLM:
    """
    Text-to-SQL LLM integration class using OpenAI GPT-4
    """
    
    def __init__(self):
        """Initialize LLM with OpenAI configuration"""
        try:
            self.llm = ChatOpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL,
                temperature=0.0,  # For deterministic SQL generation
                max_tokens=2000
            )
            logger.info(f"LLM initialized with model: {OPENAI_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise LLMError(f"Failed to initialize LLM: {str(e)}")
    
    def generate_sql(
        self,
        natural_language_query: str,
        schema_context: str,
        table_definitions: str
    ) -> str:
        """
        Generate SQL query from natural language input
        
        Args:
            natural_language_query: User's query in natural language
            schema_context: Database schema information
            table_definitions: Detailed table structure definitions
            
        Returns:
            Generated SQL query
            
        Raises:
            LLMError: If SQL generation fails
        """
        prompt_template = PromptTemplate(
            input_variables=["schema", "tables", "query"],
            template="""You are an expert SQL developer. Convert the following natural language query into a valid SQL query.

Database Schema:
{schema}

Table Definitions:
{tables}

User Query: {query}

Generate only the SQL query, without any explanation or markdown formatting. Ensure the query is:
1. Valid and executable
2. Optimized for performance
3. Using proper JOIN syntax where needed
4. Including appropriate WHERE clauses
5. Using CTEs (Common Table Expressions) for complex queries when appropriate

SQL Query:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(
                schema=schema_context,
                tables=table_definitions,
                query=natural_language_query
            )
            logger.info("SQL generated successfully")
            return result.strip()
        except Exception as e:
            logger.error(f"Error generating SQL: {str(e)}")
            raise LLMError(f"Failed to generate SQL: {str(e)}")
    
    def validate_sql(self, sql_query: str) -> bool:
        """
        Use LLM to validate SQL syntax
        
        Args:
            sql_query: SQL query to validate
            
        Returns:
            True if valid, False otherwise
        """
        prompt_template = PromptTemplate(
            input_variables=["sql"],
            template="""Validate the following SQL query. Respond with only 'VALID' or 'INVALID'.

SQL Query:
{sql}

Response:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(sql=sql_query).strip().upper()
            return result == "VALID"
        except Exception as e:
            logger.warning(f"Error validating SQL: {str(e)}")
            return False
    
    def explain_query(self, sql_query: str) -> str:
        """
        Generate explanation for an SQL query
        
        Args:
            sql_query: SQL query to explain
            
        Returns:
            Explanation of the query
        """
        prompt_template = PromptTemplate(
            input_variables=["sql"],
            template="""Provide a clear and concise explanation of what this SQL query does:

{sql}

Explanation:"""
        )
        
        chain = LLMChain(llm=self.llm, prompt=prompt_template)
        
        try:
            result = chain.run(sql=sql_query)
            return result.strip()
        except Exception as e:
            logger.error(f"Error explaining query: {str(e)}")
            raise LLMError(f"Failed to explain query: {str(e)}")
