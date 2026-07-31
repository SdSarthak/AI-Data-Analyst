"""
Configuration settings for Text-to-SQL LLM application
"""
import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Configuration
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-4')

# Databricks Configuration
DATABRICKS_HOST = os.getenv('DATABRICKS_HOST')
DATABRICKS_HTTP_PATH = os.getenv('DATABRICKS_HTTP_PATH')
DATABRICKS_TOKEN = os.getenv('DATABRICKS_TOKEN')
DATABRICKS_CATALOG = os.getenv('DATABRICKS_CATALOG')
DATABRICKS_SCHEMA = os.getenv('DATABRICKS_SCHEMA')

# Application Configuration
APP_SECRET_KEY = os.getenv('APP_SECRET_KEY', 'your-secret-key-change-in-production')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# AWS Configuration
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_REGION = os.getenv('AWS_REGION', 'us-east-1')

# Session Configuration
SESSION_TIMEOUT_MINUTES = 30
MAX_QUERY_LENGTH = 1000

# SQL Configuration
MAX_ROWS_DISPLAY = 100
ENABLE_QUERY_OPTIMIZATION = True
