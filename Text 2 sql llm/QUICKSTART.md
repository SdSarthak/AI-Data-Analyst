# Text-to-SQL LLM Project - Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.10+
- OpenAI API key (GPT-4)
- Databricks account credentials

### Step 1: Setup Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Credentials
```bash
# Copy environment template
cp .env.example .env

# Edit with your credentials:
# - OPENAI_API_KEY: Your OpenAI API key
# - DATABRICKS_HOST: Your Databricks workspace URL
# - DATABRICKS_HTTP_PATH: SQL warehouse HTTP path
# - DATABRICKS_TOKEN: Your personal access token
# - DATABRICKS_CATALOG & DATABRICKS_SCHEMA: Your database
```

### Step 3: Run Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

---

## 📁 Project Structure

```
text-to-sql-llm/
├── app.py                 ← Main Streamlit application
├── requirements.txt       ← Python dependencies
├── .env.example          ← Environment template
├── README.md             ← Full documentation
├── SETUP.md              ← Development setup guide
│
├── src/                  ← Core application modules
│   ├── llm_engine.py     ← LLM integration (GPT-4)
│   ├── database_connector.py  ← Databricks connector
│   ├── sql_validator.py  ← SQL validation & optimization
│   ├── query_engine.py   ← Main orchestration
│   └── auth.py           ← User authentication
│
├── config/               ← Configuration
│   └── settings.py       ← App settings
│
├── utils/                ← Utilities
│   ├── logger.py         ← Logging setup
│   └── errors.py         ← Custom exceptions
│
├── data/                 ← Data and schemas
│   └── schema_examples.sql
│
├── deployment/           ← Deployment scripts
│   └── deploy_aws.sh     ← AWS EC2 deployment
│
└── .streamlit/
    └── config.toml       ← Streamlit config
```

---

## 🎯 Key Features

### ✅ Core Features
- **Natural Language to SQL**: Convert English to SQL using GPT-4
- **Query Validation**: Automatic syntax and safety checking
- **Databricks Integration**: Direct warehouse connection
- **User Authentication**: Secure login with bcrypt hashing
- **Query History**: Track all previous queries
- **Favorites**: Save frequently used queries
- **Quick Analysis**: Pre-built analysis templates

### ✅ Advanced Features
- **SQL Optimization**: Automatic query formatting and optimization
- **Query Explanation**: Get explanations of generated SQL
- **Session Management**: 30-minute session timeout
- **Error Handling**: Comprehensive error messages and logging
- **Responsive UI**: Mobile-friendly Streamlit interface

---

## 🔧 Configuration

### Essential Environment Variables
```env
# OpenAI
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4

# Databricks
DATABRICKS_HOST=https://abc.cloud.databricks.com
DATABRICKS_HTTP_PATH=/sql/1.0/warehouses/xyz
DATABRICKS_TOKEN=dapi...
DATABRICKS_CATALOG=main
DATABRICKS_SCHEMA=food_delivery

# App
APP_SECRET_KEY=your-secret-key
DEBUG=False
```

---

## 📚 Query Examples

Try these natural language queries:

1. **"Show me the top 10 restaurants by number of orders"**
2. **"What's the average rating for each cuisine type?"**
3. **"Calculate revenue by restaurant for the last 3 months"**
4. **"Find customers who ordered more than 5 times"**
5. **"List all orders from restaurants with 4+ star ratings"**

---

## 🐳 Docker Deployment

### Local Docker
```bash
docker-compose up --build
# Access at http://localhost
```

### AWS EC2 Deployment
```bash
chmod +x deployment/deploy_aws.sh
./deployment/deploy_aws.sh
```

---

## 🛡️ Security Features

✅ **Password Security**: bcrypt hashing with 12 rounds  
✅ **SQL Injection Prevention**: Keyword filtering and validation  
✅ **Session Management**: Token-based with timeout  
✅ **HTTPS Support**: Ready for production HTTPS  
✅ **Credential Management**: Environment variable secrets  

---

## 📊 Architecture Overview

```
User Input (Natural Language)
        ↓
    Streamlit UI (Login/Query Interface)
        ↓
    Query Engine Orchestration
        ↓
    ├─→ LLM Engine (GPT-4) → Generate SQL
    ├─→ SQL Validator → Check Safety & Syntax
    ├─→ SQL Optimizer → Format & Optimize
    └─→ Database Connector → Execute on Databricks
        ↓
    Results Display & Storage
        ↓
    User History & Favorites
```

---

## 🚦 Quick Commands

```bash
# Start application
streamlit run app.py

# Run with custom config
streamlit run app.py --server.port 8000

# Check Databricks connection
python -c "from src.database_connector import DatabaseConnector; DatabaseConnector()"

# Test LLM integration
python -c "from src.llm_engine import TextToSQLLLM; TextToSQLLLM()"

# View logs (Docker)
docker-compose logs -f text-to-sql-app

# Deploy to AWS
chmod +x deployment/deploy_aws.sh && ./deployment/deploy_aws.sh
```

---

## ❓ Troubleshooting

### "Connection refused" error
- Verify DATABRICKS_HOST and HTTP_PATH
- Check internet connection
- Ensure DATABRICKS_TOKEN is valid

### "Invalid API key" error
- Verify OPENAI_API_KEY in .env
- Check key has GPT-4 access
- Ensure key hasn't expired

### "Port already in use" (8501)
```bash
lsof -i :8501
kill -9 <PID>
```

### Module not found errors
```bash
pip install -r requirements.txt --force-reinstall
```

---

## 📖 Documentation

- **README.md** - Full project documentation
- **SETUP.md** - Development setup guide
- **Code docstrings** - Function documentation
- **config/settings.py** - Configuration options

---

## 🤝 Contributing

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and test
3. Commit: `git commit -m "description"`
4. Push: `git push origin feature/your-feature`
5. Create Pull Request

---

## 📞 Support

- Check README.md for detailed documentation
- Review error logs for troubleshooting
- Test each component individually
- Verify all credentials before running

---

## 🎓 Learning Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [LangChain Docs](https://python.langchain.com)
- [OpenAI API Docs](https://platform.openai.com/docs)
- [Databricks SQL Docs](https://docs.databricks.com/sql)

---

## 📝 License

MIT License - See LICENSE file

---

## ✨ Next Steps

1. ✅ Install dependencies
2. ✅ Configure .env file
3. ✅ Test database connection
4. ✅ Run application locally
5. ✅ Try sample queries
6. ✅ Deploy to AWS (optional)

**Happy querying! 🚀**
