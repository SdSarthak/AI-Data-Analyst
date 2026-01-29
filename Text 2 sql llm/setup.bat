@echo off
REM Text-to-SQL LLM Setup Script for Windows

echo.
echo ============================================
echo  Text-to-SQL LLM Project Setup
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.10+ from https://www.python.org
    pause
    exit /b 1
)

echo [1/5] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/5] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/5] Installing dependencies...
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo [4/5] Creating .env file...
if not exist .env (
    copy .env.example .env
    echo Created .env file - please edit with your credentials
) else (
    echo .env file already exists
)

echo [5/5] Setup complete!
echo.
echo ============================================
echo  Next Steps:
echo ============================================
echo.
echo 1. Edit .env file with your credentials:
echo    OPENAI_API_KEY=your_api_key
echo    DATABRICKS_HOST=your_host
echo    DATABRICKS_HTTP_PATH=your_path
echo    DATABRICKS_TOKEN=your_token
echo    DATABRICKS_CATALOG=main
echo    DATABRICKS_SCHEMA=your_schema
echo.
echo 2. Run the application:
echo    streamlit run app.py
echo.
echo 3. Open in browser:
echo    http://localhost:8501
echo.
echo ============================================
echo.
pause
