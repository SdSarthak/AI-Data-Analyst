# Text-to-SQL LLM Application

A Streamlit-based web application that leverages Large Language Models to convert natural language queries into optimized SQL commands, enabling secure database interaction with Databricks, user authentication, and seamless deployment on AWS EC2.

## Project Overview

This application democratizes data access by allowing non-technical users to query databases using natural language. It uses GPT-4 to generate accurate SQL queries and validates them before execution.

### Key Features

- **Natural Language to SQL**: Convert plain English queries to SQL using GPT-4
- **User Authentication**: Secure login and signup with password hashing
- **Query Management**: History tracking, favorites, and quick analysis templates
- **SQL Validation**: Automatic SQL validation, optimization, and explanation
- **Databricks Integration**: Seamless connection to Databricks warehouses
- **Responsive UI**: Intuitive Streamlit interface with real-time query execution
- **AWS Deployment**: Ready-to-deploy Docker and EC2 configurations
- **Error Handling**: Comprehensive error handling and logging

## Architecture

### Components

1. **Frontend**: Streamlit web application
2. **LLM Engine**: OpenAI GPT-4 integration for SQL generation
3. **Database Connector**: Databricks SQL connector
4. **SQL Validator**: Query validation and optimization
5. **Authentication**: User management and session handling
6. **Query Engine**: Orchestration of the complete pipeline

### Data Flow

1. User enters natural language query
2. System retrieves database schema
3. LLM generates SQL query
4. Validator checks query safety and syntax
5. Query is optimized
6. Query executes on Databricks
7. Results displayed to user

## Prerequisites

- Python 3.10+
- OpenAI API key (GPT-4 access)
- Databricks account with:
  - Host URL
  - HTTP path
  - Personal access token
- Git (for deployment)

## Installation

### 1. Clone Repository

```bash
git clone https://your-repo-url.git
cd text-to-sql-llm
```

### 2. Create Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
cp .env.example .env
# Edit .env with your credentials:
# - OPENAI_API_KEY
# - DATABRICKS_HOST
# - DATABRICKS_HTTP_PATH
# - DATABRICKS_TOKEN
# - DATABRICKS_CATALOG
# - DATABRICKS_SCHEMA
```

### 5. Run Application Locally

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501`

## Configuration

### Environment Variables

Create a `.env` file in the project root:

```env
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Databricks Configuration
DATABRICKS_HOST=https://xyz.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/abc123
DATABRICKS_TOKEN=dapi...
DATABRICKS_CATALOG=main
DATABRICKS_SCHEMA=food_delivery

# Application Configuration
APP_SECRET_KEY=your-secret-key
DEBUG=False
LOG_LEVEL=INFO

# AWS Configuration (for deployment)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
AWS_REGION=us-east-1
```

## Usage

### Quick Start

1. **Login/Sign Up**: Create an account or log in
2. **Enter Query**: Type a natural language query
3. **Generate SQL**: Click "Generate SQL & Execute"
4. **View Results**: See results in a formatted table
5. **Save Favorites**: Add queries to favorites for quick access

### Query Examples

- "Show me the top 10 restaurants by number of orders"
- "What's the average rating for each cuisine type?"
- "Calculate revenue by restaurant for the last 3 months"
- "Find customers who ordered more than 5 times"
- "List all orders from restaurants with 4+ star ratings"

## Deployment

### Local Docker Deployment

```bash
# Build and run with Docker Compose
docker-compose up --build

# Application will be available at http://localhost
```

### AWS EC2 Deployment

1. **Launch EC2 Instance**:
   - Ubuntu 22.04 LTS
   - t3.medium or larger (recommended: t3.large for production)
   - Security group: Allow ports 80, 443, 22

2. **Run Deployment Script**:
   ```bash
   chmod +x deployment/deploy_aws.sh
   ./deployment/deploy_aws.sh
   ```

3. **Configure DNS** (Optional):
   - Point your domain to the EC2 instance IP
   - Run: `sudo certbot --nginx -d your-domain.com`

4. **Verify Deployment**:
   - Check service status: `sudo systemctl status text-to-sql`
   - View logs: `sudo journalctl -u text-to-sql -f`

### HTTPS Configuration

For production, enable HTTPS:

```bash
sudo certbot --nginx -d your-domain.com
```

## Project Structure

```
text-to-sql-llm/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── Dockerfile                      # Docker configuration
├── docker-compose.yml             # Docker Compose configuration
├── nginx.conf                     # Nginx reverse proxy config
│
├── config/
│   └── settings.py               # Application settings
│
├── src/
│   ├── llm_engine.py            # LLM integration (OpenAI GPT-4)
│   ├── database_connector.py     # Databricks connection
│   ├── sql_validator.py          # SQL validation and optimization
│   ├── query_engine.py           # Main query processing engine
│   └── auth.py                   # User authentication & sessions
│
├── utils/
│   ├── logger.py                # Logging configuration
│   └── errors.py                # Custom exceptions
│
├── data/
│   └── schema_examples.sql       # Sample database schema
│
└── deployment/
    └── deploy_aws.sh            # AWS EC2 deployment script
```

## Features in Detail

### 1. Natural Language Query Generation

- Converts natural language to SQL using GPT-4
- Supports complex queries: CTEs, JOINs, aggregations
- Context-aware query generation using database schema

### 2. Query Validation & Optimization

- SQL syntax validation
- Safety checks (prevents dangerous keywords)
- Query optimization (formatting, index hints)
- Injection prevention

### 3. User Authentication

- Secure password hashing with bcrypt
- Session management with timeout
- User-specific query history and favorites

### 4. Query History & Favorites

- Automatic history tracking
- Save favorite queries for quick access
- Timestamp tracking for all queries

### 5. Quick Analysis Templates

- Pre-built analysis templates
- Common business questions
- One-click execution

## API Reference

### QueryEngine

```python
from src.query_engine import QueryEngine

engine = QueryEngine()
result = engine.process_query("Your natural language query")

# result contains:
# - success: bool
# - sql: str (generated SQL)
# - results: dict (execution results)
# - error: str (if failed)
```

### DatabaseConnector

```python
from src.database_connector import DatabaseConnector

db = DatabaseConnector()
results = db.execute_query("SELECT * FROM table")
schema = db.get_schema_info()
```

## Error Handling

The application includes comprehensive error handling:

- **LLMError**: LLM-related errors (API issues, timeouts)
- **DatabaseError**: Database connection or query errors
- **SQLValidationError**: SQL syntax or safety issues
- **AuthenticationError**: User authentication failures
- **QueryExecutionError**: General query processing errors

## Logging

Logs are configured in `config/settings.py`. By default, logs are output to console with configurable levels:

```python
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
```

## Performance Considerations

1. **Query Timeouts**: Default 30 seconds per query
2. **Result Limits**: Maximum 100 rows displayed by default
3. **Session Timeout**: 30 minutes of inactivity
4. **Database Optimization**: Recommended indexes on commonly queried columns

## Security Considerations

- ✅ Secure password hashing with bcrypt
- ✅ SQL injection prevention
- ✅ Dangerous SQL keyword filtering
- ✅ Session-based authentication
- ✅ HTTPS support (production)
- ✅ Environment variable secrets management

## Troubleshooting

### Connection Issues

```bash
# Test Databricks connection
python -c "from src.database_connector import DatabaseConnector; DatabaseConnector()"
```

### API Key Issues

- Verify OPENAI_API_KEY is set correctly
- Check OpenAI account has GPT-4 access
- Verify API key has sufficient quota

### Database Issues

- Test Databricks credentials
- Verify network access to Databricks host
- Check HTTP path is correct

## Future Enhancements

- [ ] Multi-database support (Snowflake, BigQuery, etc.)
- [ ] Query performance metrics and suggestions
- [ ] Advanced filtering and visualization options
- [ ] Query templates and saved dashboards
- [ ] Team collaboration features
- [ ] Usage analytics and audit logs
- [ ] Advanced caching for frequently used queries

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see LICENSE file for details.

## Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Check existing documentation
- Review error logs for detailed information

## Citation

If you use this project in your research or work, please cite:

```bibtex
@software{text_to_sql_llm,
  title={Text-to-SQL LLM Application},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

## Acknowledgments

- OpenAI for GPT-4
- Databricks for SQL warehouse infrastructure
- Streamlit for the web framework
- LangChain for LLM integration tools
