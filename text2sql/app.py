"""
Main Streamlit application for the Text-to-SQL LLM.

Run from this directory so the ``src``, ``config`` and ``utils`` packages
resolve:

    cd text2sql && streamlit run app.py
"""
import streamlit as st

from config.settings import MAX_QUERY_LENGTH, missing_settings
from src.auth import UserManager, SessionManager
from src.query_engine import QueryEngine
from src.sql_validator import SQLValidator
from utils.errors import TextToSQLError
from utils.logger import setup_logger

logger = setup_logger(__name__)

QUICK_ANALYSES = {
    "Top restaurants by orders": "Show the top 10 restaurants with the most orders",
    "Revenue analysis": "Calculate total revenue by restaurant for each month",
    "Customer segments": "Identify customer segments by order frequency and total spending",
    "Order trends": "Show order trends over time with growth metrics",
    "Review analysis": "Analyse average ratings by restaurant and food category",
}


def initialize_session_state():
    """Populate the Streamlit session state keys the app relies on."""
    defaults = {
        "authenticated": False,
        "username": None,
        "session_token": None,
        "query_engine": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def get_query_engine():
    """
    Return the session's query engine, building it on first use.

    Building is deferred until after login so a configuration problem is
    reported inside the app rather than crashing it at import time.
    """
    if st.session_state.query_engine is None:
        st.session_state.query_engine = QueryEngine()
    return st.session_state.query_engine


def login_page():
    """Render the login and signup forms."""
    st.title("Text-to-SQL LLM Application")

    absent = missing_settings()
    if absent:
        st.warning(
            "These settings are missing, so queries will fail until they are "
            f"provided: {', '.join(absent)}. Copy `.env.example` to `.env`."
        )

    login_tab, signup_tab = st.tabs(["Log in", "Sign up"])

    with login_tab:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in")

        if submitted:
            try:
                UserManager.authenticate(username, password)
            except TextToSQLError as exc:
                st.error(f"Login failed: {exc}")
            else:
                st.session_state.authenticated = True
                st.session_state.username = username.strip()
                st.session_state.session_token = SessionManager.create_session(username.strip())
                st.session_state.query_engine = None
                logger.info("User '%s' logged in", username)
                st.rerun()

    with signup_tab:
        with st.form("signup_form"):
            new_username = st.text_input("Username", key="signup_username")
            new_email = st.text_input("Email", key="signup_email")
            new_password = st.text_input("Password", type="password", key="signup_password")
            confirm_password = st.text_input(
                "Confirm password", type="password", key="confirm_password"
            )
            submitted = st.form_submit_button("Create account")

        if submitted:
            if new_password != confirm_password:
                st.error("Passwords do not match")
            else:
                try:
                    UserManager.create_user(new_username, new_email, new_password)
                except TextToSQLError as exc:
                    st.error(f"Sign up failed: {exc}")
                else:
                    st.success("Account created. Switch to the Log in tab.")


def main_app():
    """Render the authenticated application."""
    with st.sidebar:
        st.title("Text-to-SQL LLM")
        st.write(f"Signed in as **{st.session_state.username}**")

        if st.button("Log out"):
            SessionManager.destroy_session(st.session_state.session_token)
            engine = st.session_state.query_engine
            if engine is not None:
                engine.close()
            st.session_state.authenticated = False
            st.session_state.username = None
            st.session_state.session_token = None
            st.session_state.query_engine = None
            st.rerun()

        st.divider()
        menu = st.radio(
            "Menu",
            ["Query generator", "Query history", "Favourites", "Quick analysis"],
        )

    if SessionManager.get_session_data(st.session_state.session_token) is None:
        st.warning("Your session expired. Please log in again.")
        st.session_state.authenticated = False
        st.rerun()
        return

    if menu == "Query generator":
        render_query_generator()
    elif menu == "Query history":
        render_query_history()
    elif menu == "Favourites":
        render_favorites()
    else:
        render_quick_analysis()


def run_question(question: str, explain: bool = False, show_sql: bool = True):
    """Send a question through the pipeline and render the outcome."""
    try:
        engine = get_query_engine()
    except TextToSQLError as exc:
        st.error(str(exc))
        return

    SessionManager.add_to_history(st.session_state.session_token, question)

    with st.spinner("Processing your query..."):
        result = engine.process_query(question)

    if not result["success"]:
        st.error(f"Query processing failed: {result['error']}")
        if result.get("sql"):
            st.code(result["sql"], language="sql")
        return

    if show_sql:
        st.subheader("Generated SQL")
        st.code(result["sql"], language="sql")

    st.subheader("Results")
    results = result["results"]
    if results["row_count"] > 0:
        st.dataframe(results["data"], use_container_width=True)
        caption = f"Rows: {results['row_count']} | Columns: {results['column_count']}"
        if results.get("truncated"):
            caption += " (truncated for display)"
        st.caption(caption)
    else:
        st.warning("Query returned no results")

    if explain:
        st.subheader("Query explanation")
        with st.spinner("Explaining..."):
            st.write(engine.explain_generated_sql(result["sql"]))

    if st.button("Add to favourites", key=f"fav_{hash(result['sql'])}"):
        if SessionManager.add_to_favorites(st.session_state.session_token, result["sql"]):
            st.success("Added to favourites")
        else:
            st.info("Already in favourites")


def render_query_generator():
    """Render the free-form query interface."""
    st.header("Natural language query generator")

    col1, col2 = st.columns([2, 1])
    with col1:
        user_query = st.text_area(
            "Enter your query in natural language:",
            placeholder="e.g. Show me the total orders by restaurant in the last month",
            height=100,
            max_chars=MAX_QUERY_LENGTH,
        )
    with col2:
        st.markdown("### Options")
        show_sql = st.checkbox("Show generated SQL", value=True)
        explain = st.checkbox("Explain query", value=False)

    if st.button("Generate SQL and execute", type="primary"):
        if not user_query.strip():
            st.error("Please enter a query")
        else:
            run_question(user_query, explain=explain, show_sql=show_sql)


def render_query_history():
    """Render the session's question history."""
    st.header("Query history")

    session_data = SessionManager.get_session_data(st.session_state.session_token)
    history = session_data["query_history"] if session_data else []
    if not history:
        st.info("No query history yet")
        return

    for i, item in enumerate(reversed(history), 1):
        label = f"{i}. {item['query'][:60]} ({item['timestamp']:%Y-%m-%d %H:%M})"
        with st.expander(label):
            st.text(item["query"])
            if st.button("Run again", key=f"reuse_{i}"):
                run_question(item["query"])


def render_favorites():
    """Render saved queries and allow re-running them."""
    st.header("Favourite queries")

    session_data = SessionManager.get_session_data(st.session_state.session_token)
    favorites = session_data["favorites"] if session_data else []
    if not favorites:
        st.info("No favourite queries yet")
        return

    for i, item in enumerate(favorites, 1):
        label = f"{i}. {item['query'][:60]} (added {item['added_at']:%Y-%m-%d %H:%M})"
        with st.expander(label):
            st.code(item["query"], language="sql")
            if st.button("Execute", key=f"execute_fav_{i}"):
                execute_saved_query(item["query"])


def execute_saved_query(sql_query: str):
    """
    Re-run a saved SQL statement.

    The statement is re-validated first: a favourite is stored text and must
    not be trusted just because it was valid when it was saved.
    """
    is_valid, message = SQLValidator.validate_sql(sql_query)
    if not is_valid:
        st.error(f"Refused to run this query: {message}")
        return

    try:
        engine = get_query_engine()
        with st.spinner("Executing query..."):
            result = engine.db.execute_query(sql_query)
    except TextToSQLError as exc:
        st.error(f"Error executing query: {exc}")
        return

    if result["row_count"] > 0:
        st.dataframe(result["data"], use_container_width=True)
    else:
        st.warning("Query returned no results")


def render_quick_analysis():
    """Render the pre-written analysis templates."""
    st.header("Quick analysis templates")
    st.info("Pre-built questions for common analyses.")

    selected = st.selectbox("Select an analysis:", list(QUICK_ANALYSES))
    question = QUICK_ANALYSES[selected]
    st.caption(f"Question sent to the model: {question}")

    if st.button("Run analysis"):
        run_question(question)


def main():
    """Application entry point."""
    # set_page_config must be the first Streamlit call in a run, so it goes
    # here rather than inside the authenticated branch.
    st.set_page_config(page_title="Text-to-SQL LLM", layout="wide")
    initialize_session_state()

    if st.session_state.authenticated:
        main_app()
    else:
        login_page()


if __name__ == "__main__":
    main()
