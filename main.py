import re
import streamlit as st
import pandas as pd
import sqlite3
import time
from huggingface_hub import InferenceClient
import traceback

# --------- Set Up InferenceClient ---------
@st.cache_resource
def get_inference_client():
    try:
        return InferenceClient(
            provider="nebius",
         #   api_key=""  # Replace with your actual API key
        )
    except Exception as e:
        st.error(f"Failed to initialize inference client: {e}")
        return None

# --------- Helper Function: Generate SQL using the Chat Completion API ---------
def generate_sql(natural_query, table_info):
    client = get_inference_client()
    if not client:
        return None
        
    prompt = f"""
Translate the following English question to a valid SQL query.
Table name: data
Table schema: {table_info}

Question: {natural_query}

Return ONLY the SQL query without any explanations or markdown formatting.
"""
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    try:
        completion = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
            messages=messages,
            max_tokens=500,
        )
        generated_sql = completion.choices[0].message.content.strip()
        return generated_sql
    except Exception as e:
        st.error(f"Error during inference: {str(e)}")
        st.error(traceback.format_exc())
        return None

# --------- Function to Clean and Fix Generated SQL ---------
def fix_generated_sql(sql_query):
    if not sql_query:
        return None
        
    # Remove any chain-of-thought text enclosed in <think>...</think>
    sql_query = re.sub(r"<think>.*?</think>", "", sql_query, flags=re.DOTALL)
    
    # Remove markdown code block formatting if present
    sql_query = re.sub(r"```sql\s*|\s*```", "", sql_query, flags=re.DOTALL)
    
    # Find the first valid SELECT statement and discard preceding text
    match = re.search(r"(SELECT|WITH|WITH\s+RECURSIVE)\s+", sql_query, re.IGNORECASE)
    if match:
        sql_query = sql_query[match.start():]
    
    # Replace common placeholders with our actual table name ("data")
    replacements = {
        "table_name": "data",
        "your_table": "data",
        "your_data_name": "data",
        "your_data": "data",
    }
    
    for old, new in replacements.items():
        sql_query = re.sub(r"\b" + old + r"\b", new, sql_query, flags=re.IGNORECASE)
    
    # Replace "COUNT Row" with "COUNT(*)"
    sql_query = re.sub(r"COUNT\s+Row", "COUNT(*)", sql_query, flags=re.IGNORECASE)
    
    return sql_query.strip()

# --------- Function to get table schema information ---------
def get_table_schema(df):
    columns = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            if pd.api.types.is_integer_dtype(dtype):
                col_type = "INTEGER"
            else:
                col_type = "FLOAT"
        elif pd.api.types.is_datetime64_dtype(dtype):
            col_type = "DATETIME"
        else:
            col_type = "TEXT"
        columns.append(f"{col} ({col_type})")
    
    return ", ".join(columns)

# --------- Function to validate SQL query syntax ---------
def validate_sql_query(sql_query):
    try:
        # Connect to an in-memory SQLite database
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        # Try executing the query to check if it is valid
        cursor.execute(sql_query)
        conn.close()
        return True
    except sqlite3.Error as e:
        return False, f"SQL syntax error: {str(e)}"

# --------- Function to execute SQL query ---------
def execute_sql_query(df, query):
    # Validate the query before execution
    is_valid, error_msg = validate_sql_query(query)
    if not is_valid:
        return None, error_msg

    # Create a new connection for each query execution
    conn = sqlite3.connect(":memory:")
    try:
        # Load the dataframe into the new connection
        df.to_sql("data", conn, index=False, if_exists="replace")
        # Execute the query
        result = pd.read_sql_query(query, conn)
        return result, None
    except Exception as e:
        return None, f"Error executing SQL query: {str(e)}"
    finally:
        conn.close()

# --------- Streamlit App Layout ---------
st.set_page_config(page_title="Data Analyst Agent", layout="wide", initial_sidebar_state="expanded")
st.title("🔍 Data Analyst Agent")
st.write("Upload your dataset and ask questions in natural language. The app will convert your question into SQL, execute it, and show the results.")

# ---- Sidebar: Upload Dataset ----
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

# Initialize session state for storing dataframe
if 'df' not in st.session_state:
    st.session_state.df = None

# ---- Main Area: Query Input and Results ----
if uploaded_file is not None:
    try:
        # Only reload the data if the file has changed
        file_details = {"FileName": uploaded_file.name, "FileSize": uploaded_file.size}
        if st.session_state.df is None or st.session_state.get('file_name') != uploaded_file.name:
            with st.spinner("Loading data..."):
                st.session_state.df = pd.read_csv(uploaded_file)
                st.session_state.file_name = uploaded_file.name
        
        df = st.session_state.df
        
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        
        # Display dataset info
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
        with col2:
            st.info(f"Column names: {', '.join(df.columns.tolist())}")
        
        # Get table schema for better SQL generation
        table_schema = get_table_schema(df)
        
        # Query input
        user_query = st.text_area("Enter your data question (in plain English):", height=100)
        col1, col2 = st.columns([1, 5])
        with col1:
            submit_button = st.button("Run Analysis", type="primary")
        
        if submit_button and user_query:
            st.subheader("Processing your query...")
            
            with st.spinner("Generating SQL query from natural language..."):
                sql_query = generate_sql(user_query, table_schema)
            
            if sql_query:
                sql_query_fixed = fix_generated_sql(sql_query)
                
                if sql_query_fixed:
                    st.success("Generated SQL Query:")
                    st.code(sql_query_fixed, language="sql")
                    
                    with st.spinner("Executing query..."):
                        result, error = execute_sql_query(df, sql_query_fixed)
                    
                    if error:
                        st.error(f"Error executing SQL query: {error}")
                    else:
                        st.subheader("Query Results")
                        if result.empty:
                            st.warning("The query returned no results. Try rephrasing your question.")
                        else:
                            st.dataframe(result)
                            
                            # Offer download option for results
                            csv = result.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="query_results.csv",
                                mime="text/csv",
                            )
                            
                            # Visualization suggestion based on result shape
                            if 1 < result.shape[1] <= 10 and 1 < result.shape[0] <= 100:
                                numeric_cols = result.select_dtypes(include=['number']).columns.tolist()
                                if len(numeric_cols) >= 1:
                                    st.subheader("Quick Visualization")
                                    chart_type = st.selectbox(
                                        "Select chart type:",
                                        ["Bar Chart", "Line Chart", "Scatter Plot"]
                                    )
                                    
                                    if chart_type == "Bar Chart" and len(numeric_cols) >= 1:
                                        st.bar_chart(result)
                                    elif chart_type == "Line Chart" and len(numeric_cols) >= 1:
                                        st.line_chart(result)
                                    elif chart_type == "Scatter Plot" and len(numeric_cols) >= 2:
                                        st.scatter_chart(result)
                else:
                    st.error("Failed to extract a valid SQL query from the generated response.")
            else:
                st.error("No SQL query generated. Please try rephrasing your question.")
        elif submit_button:
            st.warning("Please enter a question first.")
    
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        st.error(traceback.format_exc())
else:
    st.info("Please upload a CSV file using the sidebar.")
    
    # Example queries to help users get started
    st.subheader("Example questions you can ask:")
    examples = [
        "What is the average value in column X?",
        "Show me the top 5 rows sorted by column Y",
        "Count the number of rows where column Z is greater than 100",
        "What is the correlation between column A and column B?",
        "Group the data by column C and calculate the sum of column D"
    ]
    
    for example in examples:
        st.markdown(f"- {example}")
