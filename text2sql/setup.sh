#!/bin/bash

# Text-to-SQL LLM Setup Script for macOS/Linux

echo ""
echo "============================================"
echo " Text-to-SQL LLM Project Setup"
echo "============================================"
echo ""

# Check if Python is installed
PYTHON="${PYTHON:-python3}"
if ! command -v "$PYTHON" &> /dev/null; then
    echo "ERROR: python3 was not found on PATH"
    echo "Install Python 3.10+ from https://www.python.org, or set PYTHON=/path/to/python"
    exit 1
fi

if ! "$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    echo "ERROR: Python 3.10+ is required ($("$PYTHON" --version) found)"
    exit 1
fi

echo "[1/5] Creating virtual environment..."
"$PYTHON" -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment"
    exit 1
fi

echo "[2/5] Activating virtual environment..."
source venv/bin/activate

echo "[3/5] Installing dependencies..."
pip install -r requirements.txt --quiet
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "[4/5] Creating .env file..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "Created .env file - please edit with your credentials"
else
    echo ".env file already exists"
fi

echo "[5/5] Setup complete!"
echo ""
echo "============================================"
echo " Next Steps:"
echo "============================================"
echo ""
echo "1. Edit .env file with your credentials:"
echo "   nano .env"
echo "   # Add:"
echo "   OPENAI_API_KEY=your_api_key"
echo "   DATABRICKS_HOST=your_host"
echo "   DATABRICKS_HTTP_PATH=your_path"
echo "   DATABRICKS_TOKEN=your_token"
echo "   DATABRICKS_CATALOG=main"
echo "   DATABRICKS_SCHEMA=your_schema"
echo ""
echo "2. Run the application:"
echo "   streamlit run app.py"
echo ""
echo "3. Open in browser:"
echo "   http://localhost:8501"
echo ""
echo "============================================"
echo ""
