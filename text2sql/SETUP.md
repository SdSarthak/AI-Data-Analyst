# Text-to-SQL LLM - Development Setup Guide

## Local Development Setup

### 1. Prerequisites
- Python 3.10 or higher
- pip (Python package manager)
- Git
- Code editor (VS Code recommended)
- OpenAI API key
- Databricks account

### 2. Environment Setup

```bash
# Clone the repository
git clone <your-repo-url>
cd text-to-sql-llm

# Create virtual environment
python3.10 -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your credentials
# nano .env  (or use your preferred editor)
```

### 4. Running the Application

```bash
# Start Streamlit app
streamlit run app.py

# App will open at http://localhost:8501
```

## Docker Development Setup

### 1. Build Docker Image

```bash
docker build -t text-to-sql-llm:latest .
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

### 3. Access Application

- Application: http://localhost
- Logs: `docker-compose logs -f text-to-sql-app`

## AWS EC2 Deployment

### 1. Launch EC2 Instance

```bash
# Using AWS CLI
aws ec2 run-instances \
  --image-id ami-0c55b159cbfafe1f0 \
  --instance-type t3.medium \
  --key-name your-key-pair \
  --security-groups allow-http-https
```

### 2. Connect to Instance

```bash
ssh -i your-key.pem ubuntu@your-instance-ip
```

### 3. Run Deployment Script

```bash
# Download the repository
git clone <your-repo-url>
cd text-to-sql-llm

# Make script executable and run
chmod +x deployment/deploy_aws.sh
./deployment/deploy_aws.sh

# Configure credentials
nano .env
```

### 4. Verify Deployment

```bash
# Check service status
sudo systemctl status text-to-sql

# View logs
sudo journalctl -u text-to-sql -f

# Check Nginx
sudo systemctl status nginx
```

### 5. Configure Domain & HTTPS

```bash
# Using Let's Encrypt (recommended)
sudo certbot --nginx -d your-domain.com

# Follow the prompts to set up HTTPS
```

## Testing

### Run Tests

```bash
pytest tests/ -v
```

### Run Linting

```bash
pylint src/
flake8 src/
```

### Test LLM Integration

```python
from src.llm_engine import TextToSQLLLM

llm = TextToSQLLLM()
sql = llm.generate_sql(
    "Show top 10 restaurants",
    "schema_context",
    "table_definitions"
)
print(sql)
```

### Test Database Connection

```python
from src.database_connector import DatabaseConnector

db = DatabaseConnector()
schema = db.get_schema_info()
print(schema)
```

## Troubleshooting

### Port Already in Use

```bash
# Find process using port 8501
lsof -i :8501

# Kill the process
kill -9 <PID>
```

### Virtual Environment Issues

```bash
# Recreate virtual environment
rm -rf venv
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Database Connection Issues

1. Check DATABRICKS_HOST is correct
2. Verify DATABRICKS_TOKEN is valid
3. Confirm network access to Databricks
4. Test with: `python src/database_connector.py`

### API Key Issues

1. Verify OPENAI_API_KEY in .env
2. Check key has GPT-4 access
3. Verify key hasn't expired
4. Test with: `python src/llm_engine.py`

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

```bash
# Edit files and test locally
streamlit run app.py
```

### 3. Commit Changes

```bash
git add .
git commit -m "Description of changes"
```

### 4. Push and Create PR

```bash
git push origin feature/your-feature-name
```

## Performance Tuning

### Streamlit Performance

```toml
# .streamlit/config.toml
[client]
# Reduce animations
toolbarMode = "minimal"

[server]
# Enable caching
enableRunOnSave = false
# Increase max upload
maxUploadSize = 50
```

### Database Query Performance

- Create indexes on frequently queried columns
- Use LIMIT in development queries
- Monitor query execution times
- Consider query result caching

## Monitoring

### Log Files

```bash
# Application logs (Docker)
docker-compose logs -f text-to-sql-app

# System logs (EC2)
sudo journalctl -u text-to-sql -f
```

### Health Checks

```bash
# Check application health
curl http://localhost:8501/_stcore/health

# Check Nginx
curl http://localhost/health
```

## Backup and Recovery

### Backup User Data

```bash
# Backup session data (in production, use database)
tar -czf backup_$(date +%Y%m%d).tar.gz /app/data/
```

### Restore Data

```bash
tar -xzf backup_20240101.tar.gz
```

## Security Checklist

- [ ] Change default secret key
- [ ] Use strong database credentials
- [ ] Enable HTTPS in production
- [ ] Set up firewall rules
- [ ] Regular security updates
- [ ] Monitor for SQL injection attempts
- [ ] Implement rate limiting
- [ ] Enable audit logging

## Documentation

- README.md - Main documentation
- SETUP.md - This file (development guide)
- API documentation in docstrings
- Configuration details in config/settings.py
