"""
Main Streamlit application for Text-to-SQL LLM
"""
import streamlit as st
import pandas as pd
from datetime import datetime
from src.query_engine import QueryEngine
from src.auth import UserManager, SessionManager
from utils.logger import setup_logger
from utils.errors import TextToSQLError
from config.settings import SESSION_TIMEOUT_MINUTES

logger = setup_logger(__name__)


def initialize_session_state():
    """Initialize Streamlit session state"""
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'session_token' not in st.session_state:
        st.session_state.session_token = None
    if 'query_engine' not in st.session_state:
        st.session_state.query_engine = None


def login_page():
    """Render login/signup page"""
    st.title("🔐 Text-to-SQL LLM Application")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Login")
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login"):
            try:
                UserManager.authenticate(username, password)
                session_token = SessionManager.create_session(username)
                
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.session_token = session_token
                st.session_state.query_engine = QueryEngine()
                
                logger.info(f"User '{username}' logged in")
                st.success("Login successful!")
                st.rerun()
            except TextToSQLError as e:
                st.error(f"Login failed: {str(e)}")
    
    with col2:
        st.subheader("Sign Up")
        new_username = st.text_input("New Username", key="signup_username")
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password", key="confirm_password")
        
        if st.button("Sign Up"):
            if not new_username or not new_email or not new_password:
                st.error("All fields are required")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                try:
                    UserManager.create_user(new_username, new_email, new_password)
                    st.success("Account created successfully! Please log in.")
                except TextToSQLError as e:
                    st.error(f"Sign up failed: {str(e)}")


def main_app():
    """Render main application page"""
    st.set_page_config(page_title="Text-to-SQL LLM", layout="wide")
    
    # Sidebar
    with st.sidebar:
        st.title("📊 Text-to-SQL LLM")
        st.write(f"Welcome, **{st.session_state.username}**!")
        
        if st.button("🚪 Logout"):
            SessionManager.destroy_session(st.session_state.session_token)
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.session_token = None
            st.rerun()
        
        st.divider()
        
        # Menu
        menu = st.radio(
            "Menu",
            ["🔄 Query Generator", "📋 Query History", "⭐ Favorites", "💡 Quick Analysis"]
        )
    
    # Main content
    if menu == "🔄 Query Generator":
        render_query_generator()
    elif menu == "📋 Query History":
        render_query_history()
    elif menu == "⭐ Favorites":
        render_favorites()
    elif menu == "💡 Quick Analysis":
        render_quick_analysis()


def render_query_generator():
    """Render query generator interface"""
    st.header("🔄 Natural Language Query Generator")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_query = st.text_area(
            "Enter your query in natural language:",
            placeholder="e.g., Show me the total orders by restaurant in the last month",
            height=100
        )
    
    with col2:
        st.markdown("### Options")
        show_sql = st.checkbox("Show Generated SQL", value=True)
        optimize = st.checkbox("Optimize Query", value=True)
        explain = st.checkbox("Explain Query", value=False)
    
    if st.button("Generate SQL & Execute", type="primary"):
        if not user_query:
            st.error("Please enter a query")
        else:
            with st.spinner("Processing your query..."):
                try:
                    # Add to history
                    SessionManager.add_to_history(st.session_state.session_token, user_query)
                    
                    # Process query
                    result = st.session_state.query_engine.process_query(user_query)
                    
                    if result["success"]:
                        # Display SQL
                        if show_sql:
                            st.subheader("Generated SQL")
                            st.code(result["sql"], language="sql")
                        
                        # Display results
                        st.subheader("Results")
                        if result["results"]["row_count"] > 0:
                            st.dataframe(result["results"]["data"], use_container_width=True)
                            st.info(f"Total rows: {result['results']['row_count']} | Columns: {result['results']['column_count']}")
                        else:
                            st.warning("Query returned no results")
                        
                        # Display explanation
                        if explain:
                            st.subheader("Query Explanation")
                            explanation = st.session_state.query_engine.explain_generated_sql(result["sql"])
                            st.write(explanation)
                        
                        # Add to favorites button
                        if st.button("⭐ Add to Favorites"):
                            SessionManager.add_to_favorites(st.session_state.session_token, result["sql"])
                            st.success("Added to favorites!")
                    else:
                        st.error(f"Query processing failed: {result['error']}")
                        if "sql" in result:
                            st.code(result["sql"], language="sql")
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    logger.error(f"Error in query generator: {str(e)}")


def render_query_history():
    """Render query history"""
    st.header("📋 Query History")
    
    session_data = SessionManager.get_session_data(st.session_state.session_token)
    if session_data and session_data["query_history"]:
        for i, item in enumerate(reversed(session_data["query_history"]), 1):
            with st.expander(f"{i}. {item['query'][:50]}... ({item['timestamp'].strftime('%Y-%m-%d %H:%M')})"):
                st.text(item['query'])
                if st.button("Use Again", key=f"reuse_{i}"):
                    st.session_state.user_query = item['query']
    else:
        st.info("No query history yet")


def render_favorites():
    """Render favorites"""
    st.header("⭐ Favorite Queries")
    
    session_data = SessionManager.get_session_data(st.session_state.session_token)
    if session_data and session_data["favorites"]:
        for i, item in enumerate(session_data["favorites"], 1):
            with st.expander(f"{i}. {item['query'][:50]}... (Added: {item['added_at'].strftime('%Y-%m-%d %H:%M')})"):
                st.code(item['query'], language="sql")
                if st.button("Execute", key=f"execute_fav_{i}"):
                    with st.spinner("Executing query..."):
                        try:
                            result = st.session_state.query_engine.db.execute_query(item['query'])
                            if result["success"]:
                                st.dataframe(result["data"], use_container_width=True)
                        except Exception as e:
                            st.error(f"Error executing query: {str(e)}")
    else:
        st.info("No favorite queries yet")


def render_quick_analysis():
    """Render quick analysis templates"""
    st.header("💡 Quick Analysis Templates")
    
    st.info("Use pre-built analysis templates for common queries")
    
    analysis_options = {
        "Top Restaurants by Orders": "Show the top 10 restaurants with the most orders",
        "Revenue Analysis": "Calculate total revenue by restaurant for each month",
        "Customer Segments": "Identify customer segments by order frequency and total spending",
        "Order Trends": "Show order trends over time with growth metrics",
        "Review Analysis": "Analyze average ratings by restaurant and food category"
    }
    
    selected = st.selectbox("Select an analysis:", list(analysis_options.keys()))
    
    if st.button("Run Analysis"):
        query = analysis_options[selected]
        st.info(f"Running: {query}")
        
        with st.spinner("Processing..."):
            try:
                SessionManager.add_to_history(st.session_state.session_token, query)
                result = st.session_state.query_engine.process_query(query)
                
                if result["success"]:
                    st.subheader("Analysis Results")
                    st.dataframe(result["results"]["data"], use_container_width=True)
                    
                    with st.expander("View Generated SQL"):
                        st.code(result["sql"], language="sql")
                else:
                    st.error(f"Analysis failed: {result['error']}")
            except Exception as e:
                st.error(f"Error running analysis: {str(e)}")


def main():
    """Main application entry point"""
    initialize_session_state()
    
    if not st.session_state.authenticated:
        login_page()
    else:
        main_app()


if __name__ == "__main__":
    main()
