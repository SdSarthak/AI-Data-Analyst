# AI Data Analysis

## Overview
An intelligent data analysis application that combines natural language processing with SQL query generation. Users can ask questions in plain English, and the system automatically generates and executes SQL queries on their data.

## Features
- **Natural Language to SQL**: Convert English questions to SQL queries using AI
- **Streamlit Interface**: User-friendly web interface for data interaction
- **SQLite Integration**: Built-in database support for data storage and querying
- **Real-time Analysis**: Instant query execution and results display
- **AI-Powered Insights**: Uses DeepSeek-R1-Distill-Llama-70B model for query generation

## Technology Stack
- **Frontend**: Streamlit
- **Database**: SQLite3
- **AI Model**: DeepSeek-R1-Distill-Llama-70B via Hugging Face Inference
- **Data Processing**: Pandas

## Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your Hugging Face API key in the code or environment variables

## Usage
1. Run the Streamlit application:
   ```bash
   streamlit run main.py
   ```
2. Upload your CSV data file
3. Ask questions in natural language
4. View automatically generated SQL queries and results

## Example Queries
- "What is the average sales by region?"
- "Show me the top 10 customers by revenue"
- "How many orders were placed last month?"

## Dependencies
- streamlit
- pandas
- sqlite3 (built-in)
- huggingface_hub
- requests

## API Requirements
- Hugging Face API key for model access
- Internet connection for API calls

## File Structure
- `main.py` - Main Streamlit application
- `requirements.txt` - Python dependencies

## Contributing
Feel free to submit issues and enhancement requests!

## License
MIT License
